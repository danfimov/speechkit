import abc
import types
import uuid

from speechkit.domain.dto import task, task_status


class TaskRepositoryError(Exception):
    pass


class AbstractTaskRepository(abc.ABC):
    @abc.abstractmethod
    async def create(
        self,
    ) -> uuid.UUID: ...

    @abc.abstractmethod
    async def update(
        self,
        task_id: uuid.UUID,
        status: task_status.TaskStatus,
        result: str | types.EllipsisType = ...,
    ) -> None: ...

    @abc.abstractmethod
    async def get(
        self,
        task_id: uuid.UUID,
    ) -> task.Task: ...
