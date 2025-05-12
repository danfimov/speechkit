import uuid

from speechkit.domain.dto import file
from speechkit.domain.repository import file_content as file_content_repo
from speechkit.domain.repository import file_metadata


class FIleSystemError(Exception):
    pass


class FileSystemRepository:
    """
    Class for managing files.

    Combines direct file storage with metadata management.
    """

    def __init__(
        self,
        file_content_repository: file_content_repo.AbstractFileContentRepository,
        metadata_repository: file_metadata.AbstractMetadataRepository,
    ) -> None:
        self._file_content_repository = file_content_repository
        self._metadata_repository = metadata_repository

    async def save(
        self,
        file_content: bytes,
        task_id: uuid.UUID,
        file_name: str | None = None,
        content_type: str | None = None,
    ) -> uuid.UUID:
        """Save file and returns its unique identifier."""
        file_id = uuid.uuid4()
        await self._file_content_repository.save(file_id, file_content)
        await self._metadata_repository.save(
            file_id=file_id,
            task_id=task_id,
            filename=file_name,
            content_type=content_type,
        )
        return file_id

    async def get(self, file_id: uuid.UUID) -> tuple[bytes, file.FileMetadata]:
        """Get file content and metadata."""
        metadata = await self._metadata_repository.get(file_id)
        if not metadata:
            msg = f'Object with ID {file_id} not found'
            raise FIleSystemError(msg)
        content = await self._file_content_repository.get(file_id)
        return content, metadata

    async def get_metadata_by_filter(
        self,
        task_id: uuid.UUID,
    ) -> list[file.FileMetadata]:
        """Get file metadata by filter."""
        return await self._metadata_repository.get_by_filter(task_id=task_id)

    async def delete(self, file_id: uuid.UUID) -> None:
        """Delete file from storage."""
        await self._metadata_repository.delete(file_id)
        await self._file_content_repository.delete(file_id)
