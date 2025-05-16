import asyncio
import os
import pathlib

import aiohttp
import aiohttp_s3_client
import click
import structlog
import uvloop

from speechkit import dependencies
from speechkit.cli.commands.root import cli
from speechkit.infrastructure import logs
from speechkit.infrastructure.settings import Settings


logger = structlog.get_logger('speechkit.cli')


class Uploader:
    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str,
        bucket: str,
    ) -> None:
        self._session = aiohttp.ClientSession(
            raise_for_status=True,
            connector_owner=False,
            connector=aiohttp.TCPConnector(ssl=False),
        )
        self._client = aiohttp_s3_client.S3Client(
            url=endpoint_url,
            session=self._session,
            access_key_id=access_key,
            secret_access_key=secret_key,
            region=region,
        )
        self._bucket = bucket

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
    settings = await dependencies.container.get(Settings)
    logs.configure_logging()
    service = Uploader(
        endpoint_url=settings.s3.endpoint_url,
        access_key=settings.s3.access_key,
        secret_key=settings.s3.secret_key,
        region=settings.s3.region,
        bucket=settings.s3_bucket_models,
    )
    try:
        await service.upload_directory(directory, destination_prefix)
    finally:
        await service.stop()


@cli.command('upload-file')
@click.argument(
    'directory',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
    default='./data/whisper-model-onnx',
)
@click.argument('destination_prefix', type=str, default='whisper-model-large-ru-onnx')
def upload_file_command(directory: pathlib.Path, destination_prefix: str) -> None:
    """Upload the contents of the directory to S3 storage with needed prefix."""
    uvloop.run(upload_file(directory, destination_prefix))
