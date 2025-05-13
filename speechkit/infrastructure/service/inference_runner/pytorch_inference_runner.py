import io
import logging
import uuid

import librosa
import numpy as np
import torch
import transformers  # type: ignore[import-untyped]

from speechkit.domain.service import inference_runner


logger = logging.getLogger(__name__)


class PytorchInferenceRunnerService(
    inference_runner.AbstractInferenceRunner[
        transformers.WhisperForConditionalGeneration,
        transformers.WhisperProcessor,
    ],
):
    async def _get_audio_in_converted_format(self, task_id: uuid.UUID) -> np.ndarray:
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

    def _split_audio_data_to_segments(self, audio_data: np.ndarray) -> list[torch.Tensor]:
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

        tensors = []
        for segment in segments:
            tensor = self._processor(
                segment,
                sampling_rate=self.TARGET_SAMPLING_RATE,
                return_tensors='pt',
                chunk_length_s=self.MAX_CHUNK_LENGTH_SECONDS,
                stride_length_s=self.MAX_OVERLAP_SECONDS,
            ).input_features
            tensors.append(tensor)
        return tensors

    def _run_inference(self, segments: list[torch.Tensor]) -> list[str]:
        return [self._model.generate(segment) for segment in segments]

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
        file_data = await self._get_audio_in_converted_format(task_id)
        segments = self._split_audio_data_to_segments(audio_data=file_data)
        transcriptions_for_segments = self._run_inference(segments)
        return self._postprocess_transcriptions(transcriptions_for_segments)
