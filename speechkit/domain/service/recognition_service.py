import abc

from speechkit.domain.repository import file_system, task


class RecognitionServiceError(Exception):
    pass


class AbstractRecognitionService(abc.ABC):
    def __init__(
        self,
        file_system_repository: file_system.FileSystemRepository,
        task_repository: task.AbstractTaskRepository,
    ) -> None:
        self.file_system_repository = file_system_repository
        self.task_repository = task_repository

    @abc.abstractmethod
    async def act(
        self,
        model_name: str,
        audio_file_content: bytes,
        audio_file_name: str | None = None,
        audio_file_content_type: str | None = None,
    ) -> str: ...
