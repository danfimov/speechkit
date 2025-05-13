import asyncio
import pathlib

import aiohttp
import aiohttp_s3_client
import anyio
import structlog
import torch
import transformers  # type: ignore[import-untyped]
from optimum import onnxruntime  # type: ignore[import-untyped]

from speechkit.domain.service import model_downloader


logger = structlog.get_logger(__name__)


class HuggingFaceModelDownloader(model_downloader.AbstractModelDownloaderService):
    """Можно использовать только локально, так как в сети Точки не будет доступа к хосту."""

    async def act(self, model_name: str, destination_path: pathlib.Path) -> None:
        if model_name == 'openai/whisper-tiny.en':
            transformers.WhisperForConditionalGeneration.from_pretrained(
                'openai/whisper-tiny.en',
                cache_dir=destination_path,
                torch_dtype=torch.float32,
                local_files_only=False,
                force_download=True,
            )
        elif model_name == 'openai/whisper-tiny.en-onnx':
            model = onnxruntime.ORTModelForSpeechSeq2Seq.from_pretrained(
                'openai/whisper-tiny.en',
                export=True,
                force_download=True,
            )
            model.save_pretrained(destination_path)
        elif model_name == 'antony66/whisper-large-v3-russian':
            transformers.WhisperForConditionalGeneration.from_pretrained(
                'antony66/whisper-large-v3-russian',
                cache_dir=destination_path,
                torch_dtype=torch.float32,
                local_files_only=False,
                force_download=True,
            )
        elif model_name == 'antony66/whisper-large-v3-russian-onnx':
            model = onnxruntime.ORTModelForSpeechSeq2Seq.from_pretrained(
                'antony66/whisper-large-v3-russian',
                export=True,
                force_download=True,
            )
            model.save_pretrained(destination_path)
        else:
            msg = f'Model {model_name} not found'
            raise RuntimeError(msg)


class S3ModelDownloader(model_downloader.AbstractModelDownloaderService):
    """For internal usage in Tochka network."""

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
            connector=aiohttp.TCPConnector(ssl=False),
        )
        self._client = aiohttp_s3_client.S3Client(
            url=aws_host,
            session=self._session,
            access_key_id=aws_access_key,
            secret_access_key=aws_secret_key,
            region=aws_region,
        )
        self._bucket = aws_s3_bucket

    async def _download_file(self, s3_key: str, destination_path: pathlib.Path, file_size: int) -> None:
        relative_path = s3_key.split('/', 1)[1] if '/' in s3_key else s3_key
        file_path = destination_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if file_size > 100 * 1024 * 1024:  # More than 100MB
                logger.info(
                    'Start parallel download of file %s with size %.2f Mb',
                    file_path,
                    file_size / (1024 * 1024),
                )
                await self._client.get_file_parallel(
                    object_name=f'{self._bucket}/{s3_key}',
                    file_path=file_path,
                    workers_count=10,
                )
            else:
                logger.info(
                    'Start download of file %s with size %.2f Mb',
                    file_path,
                    file_size / (1024 * 1024),
                )
                async with (
                    self._client.get(f'{self._bucket}/{s3_key}') as resp,
                    await anyio.open_file(file_path, 'wb') as f,
                ):
                    await f.write(await resp.read())
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            msg = 'Error during file download'
            raise model_downloader.ModelDownloaderError(msg) from e
        else:
            logger.info('Downloaded %s to %s', s3_key, file_path)

    async def _download_directory(self, prefix: str, destination_path: pathlib.Path) -> None:
        logger.info('Will load %s to %s', prefix, destination_path)
        tasks = []
        async for result, _prefixes in self._client.list_objects_v2(bucket=self._bucket, prefix=prefix):
            for file in result:
                tasks.append(self._download_file(file.key, destination_path, file_size=file.size))  # noqa: PERF401
        await asyncio.gather(*tasks)

    async def act(self, model_name: str, destination_path: pathlib.Path) -> None:
        if model_name == 'openai/whisper-tiny.en':
            await self._download_directory('whisper-model-tiny-en/', destination_path)
        elif model_name == 'openai/whisper-tiny.en-onnx':
            await self._download_directory('whisper-model-tiny-en-onnx/', destination_path)
        elif model_name == 'antony66/whisper-large-v3-russian':
            await self._download_directory('whisper-model-large-ru/', destination_path)
        elif model_name == 'antony66/whisper-large-v3-russian-onnx':
            await self._download_directory('whisper-model-large-ru-onnx/', destination_path)
        else:
            msg = f'Model {model_name} not found'
            raise RuntimeError(msg)

    async def close(self) -> None:
        logger.info('Closing connections in S3ModelDownloader')
        if self._session.connector:
            await self._session.connector.close()
        await self._session.close()
