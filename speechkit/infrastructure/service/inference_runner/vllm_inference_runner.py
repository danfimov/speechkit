import io
import uuid

import librosa
import numpy as np
import structlog
import vllm  # type: ignore[import-not-found]

from speechkit.domain.service import inference_runner


logger = structlog.get_logger(__name__)


class VllmInferenceRunnerService(
    inference_runner.AbstractInferenceRunner[vllm.LLM, None],
):
    """
    A service for running inference using VLLM.

    This class implements the AbstractInferenceRunner interface to process audio files,
    split them into manageable segments, run inference using VLLM, and combine
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
            ValueError: If there isn't exactly one file associated with the task ID

        """
        file_metadata_list = await self._file_system_repository.get_metadata_by_filter(task_id=task_id)
        if len(file_metadata_list) != 1:
            msg = 'Expected exactly one file to preprocess'
            raise ValueError(msg)
        file, file_metadata = await self._file_system_repository.get(file_metadata_list[0].id)
        audio_as_numpy_array, _ = librosa.load(
            io.BytesIO(file),
            sr=self.TARGET_SAMPLING_RATE,
            mono=True,
        )
        return audio_as_numpy_array

    def _split_audio_data_to_segments(self, audio_data: np.ndarray) -> list[np.ndarray]:
        """
        Split audio data into overlapping segments of manageable length.

        Args:
            audio_data: The full audio data as a numpy array

        Returns:
            list[np.ndarray]: A list of audio segments as numpy arrays

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
        return segments

    def _run_inference(self, segments: list[np.ndarray]) -> list[str]:
        """
        Run inference on audio segments using the VLLM model.

        Args:
            segments: List of audio segments as numpy arrays

        Returns:
            list[str]: List of transcription texts for each segment

        """
        batch_inputs = [
            {
                'prompt': '<|startoftranscript|>',
                'multi_modal_data': {
                    'audio': [(segment, 16000)],
                },
            }
            for segment in segments
        ]

        sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=256)
        outputs = self._model.generate(
            batch_inputs,
            sampling_params=sampling_params,
        )
        return [output.outputs[0].text.strip() for output in outputs]

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
        segments = self._split_audio_data_to_segments(audio_data=file_data)
        logger.debug('Audio file is split into segments')
        transcriptions_for_segments = self._run_inference(segments)
        logger.debug('Inference is done, transcriptions are decoded')
        return self._postprocess_transcriptions(transcriptions_for_segments)
