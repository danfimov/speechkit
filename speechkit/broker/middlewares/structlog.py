import typing as tp

import structlog
from taskiq.abc.middleware import TaskiqMiddleware
from taskiq.message import TaskiqMessage


class StructlogMiddleware(TaskiqMiddleware):
    def pre_execute(
        self,
        message: TaskiqMessage,
    ) -> (
        TaskiqMessage | tp.Coroutine[tp.Any, tp.Any, TaskiqMessage]
    ):
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            taskiq_task_id=message.task_id, taskiq_task_name=message.task_name,
        )
        return message
