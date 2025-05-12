import abc
import uuid

from speechkit.domain.dto import file


class AbstractMetadataRepository(abc.ABC):
    @abc.abstractmethod
    async def save(
        self,
        file_id: uuid.UUID,
        task_id: uuid.UUID,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> None: ...

    @abc.abstractmethod
    async def get(self, file_id: uuid.UUID) -> file.FileMetadata: ...

    @abc.abstractmethod
    async def get_by_filter(self, task_id: uuid.UUID) -> list[file.FileMetadata]: ...

    @abc.abstractmethod
    async def delete(self, file_id: uuid.UUID) -> None: ...
