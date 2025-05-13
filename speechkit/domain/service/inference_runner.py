import abc
import typing as tp
import uuid

from speechkit.domain.repository import file_system


class InferenceRunnerError(Exception):
    pass


_TypeModel = tp.TypeVar('_TypeModel', bound=tp.Any)
_TypeProcessor = tp.TypeVar('_TypeProcessor', bound=tp.Any)


class AbstractInferenceRunner(tp.Generic[_TypeModel, _TypeProcessor]):
    TARGET_SAMPLING_RATE: int = 16000  # optimal for Whisper
    MAX_CHUNK_LENGTH_SECONDS: int = 20
    MAX_OVERLAP_SECONDS: int = 5

    def __init__(
        self,
        model: _TypeModel,
        processor: _TypeProcessor,
        file_system_repository: file_system.FileSystemRepository,
    ) -> None:
        self._model = model
        self._processor = processor
        self._file_system_repository = file_system_repository

    @abc.abstractmethod
    async def run(self, task_id: uuid.UUID) -> str:
        """
        Get file for task and run inference with its content.

        Args:
            task_id: The UUID identifying the task to process

        Returns:
            str: The complete transcription text for the audio file

        """
        ...
