import asyncio
import logging
import os
import pathlib

import aiohttp
import aiohttp_s3_client
import click
import uvloop
from tochka_infrastructure import logs

from speechkit import dependencies
from speechkit.infrastructure import settings


logger = logging.getLogger(__name__)


class Uploader:
    def __init__(self, settings: settings.Settings) -> None:
        self._session = aiohttp.ClientSession(
            raise_for_status=True,
            connector_owner=False,
            connector=aiohttp.TCPConnector(ssl=False),
        )
        self._client = aiohttp_s3_client.S3Client(
            url=settings.s3.endpoint_url,
            session=self._session,
            access_key_id=settings.s3.access_key,
            secret_access_key=settings.s3.secret_key,
            region=settings.s3.region,
        )
        self._bucket = settings.s3_bucket_models

    async def _upload_file(self, object_name: str, file_path: pathlib.Path) -> None:
        file_size = file_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # More than 100MB
            await self._client.put_file_multipart(
                object_name=object_name,
                file_path=file_path,
                workers_count=8,
            )
        else:
            # For smaller files, use regular put_file
            await self._client.put_file(
                object_name=object_name,
                file_path=file_path,
            )
        logger.info('File %s uploaded to %s', file_path, object_name)

    async def upload_directory(self, directory: pathlib.Path, destination_prefix: str) -> None:
        logger.info('Uploading directory %s to %s', directory, destination_prefix)
        tasks = []
        for root, _dirs, files in os.walk(directory):
            for filename in files:
                file_path = pathlib.Path(root) / filename
                relative_path = os.path.relpath(file_path, directory)
                tasks.append(
                    self._upload_file(
                        object_name=f'{self._bucket}/{destination_prefix}/{relative_path}',
                        file_path=file_path,
                    ),
                )
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        if self._session.connector:
            await self._session.connector.close()
        await self._session.close()


async def upload_file(directory: pathlib.Path, destination_prefix: str) -> None:
    settings = dependencies.get_settings()
    logs.configure_logging(settings.log_config_path, settings.log_level)
    service = Uploader(settings=settings)
    try:
        await service.upload_directory(directory, destination_prefix)
    finally:
        await service.stop()


@click.command()
@click.argument(
    'directory',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    default='./data/whisper-model-onnx',
)
@click.argument('destination_prefix', type=str, default='whisper-model-large-ru-onnx')
def command(directory: pathlib.Path, destination_prefix: str) -> None:
    """Upload the contents of the directory to S3 storage with needed prefix."""
    uvloop.run(upload_file(directory, destination_prefix))


if __name__ == '__main__':
    command()
