import uuid

import pydantic

from speechkit.domain.dto import task_status


class Task(pydantic.BaseModel):
    id: uuid.UUID
    status: task_status.TaskStatus
    result: str
