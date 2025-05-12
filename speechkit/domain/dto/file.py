import uuid

import pydantic


class FileMetadata(pydantic.BaseModel):
    id: uuid.UUID
    filename: str | None
    content_type: str | None
    task_id: uuid.UUID


class AudioFile(pydantic.BaseModel):
    content: bytes
