import uuid

from dishka.integrations.taskiq import FromDishka, inject
from taskiq.brokers.shared_broker import async_shared_broker

from speechkit.domain.dto import task_status
from speechkit.domain.repository import task
from speechkit.domain.service import inference_runner


@async_shared_broker.task
@inject(patch_module=False)
async def run_inference(
    task_id: uuid.UUID,
    task_repository: FromDishka[task.AbstractTaskRepository],
    service: FromDishka[inference_runner.AbstractInferenceRunner],
) -> str:
    try:
        result_text = await service.run(task_id=task_id)
    except inference_runner.InferenceRunnerError:
        await task_repository.update(
            task_id=task_id,
            status=task_status.TaskStatus.FAILED,
        )
        raise
    return result_text
