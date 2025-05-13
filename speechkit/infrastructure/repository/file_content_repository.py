import http
import uuid

import aiohttp
import aiohttp_s3_client
import structlog

from speechkit.domain.repository import file_content as file_content_repo


logger = structlog.get_logger(__name__)


class S3FileContentRepository(file_content_repo.AbstractFileContentRepository):
    def __init__(
        self,
        aws_access_key: str,
        aws_secret_key: str,
        aws_region: str,
        aws_s3_bucket: str,
        aws_host: str,
    ) -> None:
        self._session = aiohttp.ClientSession(
            raise_for_status=True,
            connector_owner=False,
            connector=aiohttp.TCPConnector(
                ssl=False,
                limit=10,  # лимитируем количество одновременных соединений
            ),
        )
        self._client = aiohttp_s3_client.S3Client(
            url=aws_host,
            session=self._session,
            access_key_id=aws_access_key,
            secret_access_key=aws_secret_key,
            region=aws_region,
        )
        self._bucket = aws_s3_bucket

    async def save(self, file_id: uuid.UUID, file_content: bytes) -> None:
        async with self._client.put(
            object_name=f'{self._bucket}/{file_id.hex}',
            data=file_content,
        ) as resp:
            if resp.status == http.HTTPStatus.OK:
                logger.info('File %s saved to S3', file_id)
            else:
                logger.error('Failed to save file %s to S3', file_id)
                msg = f'Failed to save file {file_id} to S3'
                raise file_content_repo.FileContentRepositoryError(msg)

    async def get(self, file_id: uuid.UUID) -> bytes:
        async with self._client.get(object_name=f'{self._bucket}/{file_id.hex}') as resp:
            if resp.status == http.HTTPStatus.OK:
                return await resp.read()
            logger.error('Failed to get file %s from S3', file_id)
            msg = f'Failed to get file {file_id} from S3'
            raise file_content_repo.FileContentRepositoryError(msg)

    async def delete(self, file_id: uuid.UUID) -> None:
        async with self._client.delete(object_name=f'{self._bucket}/{file_id.hex}') as resp:
            if resp.status == http.HTTPStatus.OK:
                logger.info('File %s deleted from S3', file_id)
            else:
                logger.warning('File %s not found on S3 during deletion', file_id)
