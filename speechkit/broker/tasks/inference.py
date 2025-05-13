import typing as tp
import uuid

import taskiq

from speechkit import dependencies
from speechkit.domain.dto import task_status
from speechkit.domain.repository import task
from speechkit.domain.service import inference_runner


async def run_inference(
    task_id: uuid.UUID,
    task_repository: tp.Annotated[task.AbstractTaskRepository, taskiq.TaskiqDepends(dependencies.get_task_repository)],
    service: tp.Annotated[
        inference_runner.AbstractInferenceRunner,
        taskiq.TaskiqDepends(dependencies.get_inference_runner),
    ],
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
