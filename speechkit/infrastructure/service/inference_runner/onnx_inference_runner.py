import io
import uuid

import librosa
import numpy as np
import structlog
import torch
import transformers  # type: ignore[import-untyped]
from optimum import onnxruntime  # type: ignore[import-untyped]

from speechkit.domain.service import inference_runner


logger = structlog.get_logger(__name__)


class OnnxInferenceRunnerService(
    inference_runner.AbstractInferenceRunner[onnxruntime.ORTModelForSpeechSeq2Seq, transformers.WhisperProcessor],
):
    """
    A service for running inference using ONNX Runtime with a Whisper processor.

    This class implements the AbstractInferenceRunner interface to process audio files,
    split them into manageable segments, run inference using ONNX Runtime, and combine
    the transcription results.
    """

    async def _get_audio_in_converted_format(self, task_id: uuid.UUID) -> np.ndarray:
        """
        Load audio file data from file system and preprocess it to be compatible with inference runner.

        Args:
            task_id: The UUID identifying the task to process

        Returns:
            np.ndarray: Audio data as a numpy array with proper sample rate

        Raises:
            InferenceRunnerError: If there isn't exactly one file associated with the task ID

        """
        file_metadata_list = await self._file_system_repository.get_metadata_by_filter(task_id=task_id)
        if len(file_metadata_list) != 1:
            msg = 'Expected exactly one file to preprocess'
            raise inference_runner.InferenceRunnerError(msg)
        file, file_metadata = await self._file_system_repository.get(file_metadata_list[0].id)
        audio_as_numpy_array, _ = librosa.load(
            io.BytesIO(file),
            sr=self.TARGET_SAMPLING_RATE,
            mono=True,
        )
        if len(audio_as_numpy_array) < 1:
            msg = 'Audio file appears to be empty or corrupted'
            raise inference_runner.InferenceRunnerError(msg)
        return audio_as_numpy_array

    def _split_audio_data_to_segments(self, audio_data: np.ndarray) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Split audio data into overlapping segments of manageable length.

        Args:
            audio_data: The full audio data as a numpy array

        Returns:
            list[torch.Tensor]: A list of audio segments as torch tensors

        """
        max_length_samples = self.MAX_CHUNK_LENGTH_SECONDS * self.TARGET_SAMPLING_RATE
        overlap_samples = self.MAX_OVERLAP_SECONDS * self.TARGET_SAMPLING_RATE
        segments = []
        start = 0
        while start < len(audio_data):
            end = min(start + max_length_samples, len(audio_data))
            segments.append(audio_data[start:end])
            start = start + max_length_samples - overlap_samples
            if end == len(audio_data):
                break

        segments_with_attention_mask = []
        for segment in segments:
            processor_result = self._processor(
                segment,
                sampling_rate=self.TARGET_SAMPLING_RATE,
                return_tensors='pt',
                chunk_length_s=self.MAX_CHUNK_LENGTH_SECONDS,
                stride_length_s=self.MAX_OVERLAP_SECONDS,
            )
            input_features = processor_result.input_features
            # Create attention mask - for audio input features, typically all values are attended to
            # so we create a tensor of ones with the same shape as the input_features first dimension
            attention_mask = torch.ones(input_features.shape[0], dtype=torch.long)
            segments_with_attention_mask.append((input_features, attention_mask))
        return segments_with_attention_mask

    def _run_inference(self, segments: list[tuple[torch.Tensor, torch.Tensor]]) -> list[str]:
        """
        Run inference on audio segments using the ONNX Runtime model.

        Args:
            segments: List of audio segments as torch tensors

        Returns:
            list[str]: List of transcription texts for each segment

        """
        return [
            self._model.generate(
                segment,
                attention_mask=attention_mask,
                max_length=448,  # максимальная длина генерации
                num_beams=5,
                do_sample=False,
                use_cache=True,  # использование кэширования для ускорения
            )
            for segment, attention_mask in segments
        ]

    def _decode(self, transcriptions: list[str]) -> list[str]:
        """
        Decode transcription tokens into text.

        Args:
            transcriptions: List of transcription tokens for audio segments

        Returns:
            list[str]: List of decoded transcription texts for each segment

        """
        return [
            self._processor.batch_decode(transcription, skip_special_tokens=True)[0]
            for transcription in transcriptions
        ]

    def _postprocess_transcriptions(self, transcriptions: list[str]) -> str:
        """
        Join transcription parts into a single string.

        Args:
            transcriptions: List of transcription texts for audio segments

        Returns:
            str: The combined transcription text

        """
        return ' '.join(transcriptions)

    async def run(self, task_id: uuid.UUID) -> str:
        """
        Get file for task and run inference with its content.

        Args:
            task_id: The UUID identifying the task to process

        Returns:
            str: The complete transcription text for the audio file

        """
        logger.debug('Starting inference process')
        file_data = await self._get_audio_in_converted_format(task_id)
        logger.debug('Audio file is converted')
        segments_with_attention_mask = self._split_audio_data_to_segments(audio_data=file_data)
        logger.debug('Audio file is split into segments')
        transcriptions_for_segments = self._run_inference(segments_with_attention_mask)
        logger.debug('Inference is done')
        decoded_transcriptions_for_segments = self._decode(transcriptions_for_segments)
        logger.debug('Transcriptions are decoded')
        return self._postprocess_transcriptions(decoded_transcriptions_for_segments)
