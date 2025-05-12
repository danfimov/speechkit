import abc
import uuid


class FileContentRepositoryError(Exception):
    pass


class AbstractFileContentRepository(abc.ABC):
    @abc.abstractmethod
    async def save(self, file_id: uuid.UUID, file_content: bytes) -> None: ...

    @abc.abstractmethod
    async def get(self, file_id: uuid.UUID) -> bytes: ...

    @abc.abstractmethod
    async def delete(self, file_id: uuid.UUID) -> None: ...
