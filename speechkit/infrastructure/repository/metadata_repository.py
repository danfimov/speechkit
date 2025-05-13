import uuid

import sqlalchemy as sa

from speechkit.domain.dto import file
from speechkit.domain.repository import file_metadata
from speechkit.infrastructure.database import schemas, session_provider


class PostgresMetadataRepository(file_metadata.AbstractMetadataRepository):
    def __init__(self, session_provider: session_provider.AsyncPostgresSessionProvider) -> None:
        self._session_provider = session_provider

    async def save(
        self,
        file_id: uuid.UUID,
        task_id: uuid.UUID,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> None:
        if isinstance(filename, str) and len(filename) > 255:  # noqa: PLR2004
            filename = filename[:252] + '...'
        query = sa.insert(schemas.FileMetadata).values(
            id=file_id,
            filename=filename,
            content_type=content_type,
            task_id=task_id,
        )
        async with self._session_provider.session() as session:
            await session.execute(query)

    async def get(self, file_id: uuid.UUID) -> file.FileMetadata:
        query = sa.select(schemas.FileMetadata).where(schemas.FileMetadata.id == file_id).limit(1)
        async with self._session_provider.session() as session:
            result = await session.execute(query)
            file_metadata = result.scalar_one_or_none()
            if file_metadata is None:
                msg = f'Object with ID {file_id} not found in metadata repository'
                raise KeyError(msg)
            return file.FileMetadata(
                id=file_metadata.id,
                filename=file_metadata.filename,
                content_type=file_metadata.content_type,
                task_id=file_metadata.task_id,
            )

    async def get_by_filter(self, task_id: uuid.UUID) -> list[file.FileMetadata]:
        query = sa.select(schemas.FileMetadata).where(schemas.FileMetadata.task_id == task_id)
        async with self._session_provider.session() as session:
            result = await session.execute(query)
            metadata_items = result.scalars()
            return [
                file.FileMetadata(
                    id=metadata_item.id,
                    filename=metadata_item.filename,
                    content_type=metadata_item.content_type,
                    task_id=metadata_item.task_id,
                )
                for metadata_item in metadata_items
            ]

    async def delete(self, file_id: uuid.UUID) -> None:
        query = sa.delete(schemas.FileMetadata).where(schemas.FileMetadata.id == file_id)
        async with self._session_provider.session() as session:
            await session.execute(query)
