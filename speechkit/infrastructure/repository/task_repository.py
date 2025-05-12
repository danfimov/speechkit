import types
import uuid

import sqlalchemy as sa

from speechkit.domain.dto import task as task_dto
from speechkit.domain.dto import task_status
from speechkit.domain.repository import task
from speechkit.infrastructure.database import schemas, session_provider


class PostgresTaskRepository(task.AbstractTaskRepository):
    def __init__(self, session_provider: session_provider.AsyncPostgresSessionProvider) -> None:
        self._session_provider = session_provider

    async def create(self) -> uuid.UUID:
        query_task_creation = (
            sa.insert(schemas.Task).values(status=task_status.TaskStatus.CREATED.value).returning(schemas.Task.id)
        )
        async with self._session_provider.session() as session:
            result = await session.execute(query_task_creation)
            return result.scalar_one()

    async def update(
        self,
        task_id: uuid.UUID,
        status: task_status.TaskStatus,
        result: str | types.EllipsisType = ...,
    ) -> None:
        values: dict[str, int | str] = {'status': status.value}
        if not isinstance(result, types.EllipsisType):
            values['result'] = result
        query_task_update = sa.update(schemas.Task).values(**values).where(schemas.Task.id == task_id)
        async with self._session_provider.session() as session:
            await session.execute(query_task_update)

    async def get(
        self,
        task_id: uuid.UUID,
    ) -> task_dto.Task:
        query = sa.select(schemas.Task).where(schemas.Task.id == task_id)
        async with self._session_provider.session() as session:
            result = await session.execute(query)
            task_from_db = result.scalar_one_or_none()
            if task_from_db is None:
                msg = f'Task {task_id} not found'
                raise task.TaskRepositoryError(msg)
            return task_dto.Task(
                id=task_from_db.id,
                status=task_status.TaskStatus(task_from_db.status),
                result=task_from_db.result,
            )
