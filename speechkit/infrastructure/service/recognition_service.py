from taskiq.abc import broker as taskiq_broker

from speechkit.domain.repository import file_system, task
from speechkit.domain.service import recognition_service


class TaskiqRecognitionService(recognition_service.AbstractRecognitionService):
    def __init__(
        self,
        file_system_repository: file_system.FileSystemRepository,
        task_repository: task.AbstractTaskRepository,
        broker: taskiq_broker.AsyncBroker,
    ) -> None:
        super().__init__(file_system_repository, task_repository)
        self.broker = broker

    async def act(
        self,
        model_name: str,  # noqa: ARG002
        audio_file_content: bytes,
        audio_file_name: str | None = None,
        audio_file_content_type: str | None = None,
    ) -> str:
        """
        Recognizes the audio file and returns the transcribed text.

        In case of an error, returns the error text.
        """
        from speechkit.broker.tasks.inference import run_inference

        task_id = await self.task_repository.create()
        await self.file_system_repository.save(
            file_content=audio_file_content,
            file_name=audio_file_name,
            content_type=audio_file_content_type,
            task_id=task_id,
        )
        task = await run_inference.kiq(task_id)
        task_result = await task.wait_result()
        if task_result.error:
            raise recognition_service.RecognitionServiceError(task_result.error)
        return str(task_result.return_value)
