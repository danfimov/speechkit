import pathlib
from typing import assert_never
from urllib.parse import quote

import onnxruntime as ort  # type: ignore[import-untyped]
import psutil
import structlog
import taskiq
import taskiq_aio_pika
import taskiq_pipelines
import torch
import transformers  # type: ignore[import-untyped]
from dishka import Provider, Scope, make_async_container, make_container
from dishka.integrations.fastapi import FastapiProvider
from dishka.integrations.taskiq import TaskiqProvider, setup_dishka
from optimum import onnxruntime  # type: ignore[import-untyped]
from prometheus_async.aio.web import MetricsHTTPServer
from taskiq.brokers.shared_broker import async_shared_broker
from taskiq_pg.asyncpg import AsyncpgResultBackend

from speechkit.domain.repository import file_content, file_metadata, file_system, task
from speechkit.domain.service import inference_runner, model_downloader, recognition_service
from speechkit.infrastructure import logs, metrics
from speechkit.infrastructure.database import session_provider
from speechkit.infrastructure.repository import (
    file_content_repository,
    metadata_repository,
    task_repository,
)
from speechkit.infrastructure.service import recognition_service as taskiq_recognition_service
from speechkit.infrastructure.settings import Settings


logger = structlog.get_logger(__name__)


def provide_settings() -> Settings:
    return Settings()


async def provide_metrics_server(settings: Settings) -> MetricsHTTPServer | None:
    if settings.metrics_port:
        return await metrics.start_metric_server(port=settings.metrics_port)
    return None


async def provide_session_provider(settings: Settings) -> session_provider.AsyncPostgresSessionProvider:
    return session_provider.AsyncPostgresSessionProvider(
        connection_settings=settings.postgres,
    )


def provide_broker(settings: Settings) -> taskiq.AsyncBroker:
    from speechkit.broker import middlewares

    broker: taskiq.AsyncBroker
    if settings.broker_type == 'amqp' and settings.amqp:
        broker = taskiq_aio_pika.AioPikaBroker(
            url=settings.amqp.get_url(),
        ).with_result_backend(
            AsyncpgResultBackend(
                f'postgresql://{settings.postgres.user}:{quote(settings.postgres.password.get_secret_value())}@{settings.postgres.host}:{settings.postgres.port}/{settings.postgres.database}',
            ),
        )
    else:
        broker = taskiq.InMemoryBroker()
    broker.add_middlewares(
        taskiq_pipelines.PipelineMiddleware(),
        # TODO: Enable it with tweaks. Probably related to https://github.com/taskiq-python/taskiq/issues/397 or https://github.com/taskiq-python/taskiq/issues/173
        # middlewares.PrometheusMiddleware(),
        middlewares.StructlogMiddleware(),
    )
    async_shared_broker.default_broker(broker)

    @broker.on_event(taskiq.TaskiqEvents.WORKER_STARTUP)
    async def startup(_: taskiq.TaskiqState) -> None:
        logs.configure_logging(
            log_level=settings.log_level,
            json_log=settings.environment not in ('local', 'unknown', 'development'),
        )
        # await container.get(MetricsHTTPServer | None)
        await container.get(inference_runner.AbstractInferenceRunner)

    setup_dishka(container=container, broker=broker)
    return broker


async def provide_model_downloader(settings: Settings) -> model_downloader.AbstractModelDownloaderService:
    from speechkit.infrastructure.service.model_downloader import HuggingFaceModelDownloader, S3ModelDownloader

    if settings.downloader_type == 'huggingface':
        return HuggingFaceModelDownloader()
    return S3ModelDownloader(
        aws_access_key=settings.s3.access_key,
        aws_secret_key=settings.s3.secret_key,
        aws_region=settings.s3.region,
        aws_s3_bucket=settings.s3_bucket_models,
        aws_host=settings.s3.endpoint_url,
    )


async def provide_whisper_processor(
    settings: Settings,
    model_downloader: model_downloader.AbstractModelDownloaderService,
) -> transformers.WhisperProcessor:
    cache_directory = pathlib.Path('data/whisper-model')
    if not cache_directory.exists():
        cache_directory.mkdir(parents=True)
        await model_downloader.act(
            model_name=settings.model_name,
            destination_path=cache_directory,
        )
    return transformers.WhisperProcessor.from_pretrained(
        pretrained_model_name_or_path=settings.model_name,
        cache_dir=cache_directory,
        local_files_only=True,
    )


async def provide_whisper_model_for_pytorch(
    settings: Settings,
    model_downloader: model_downloader.AbstractModelDownloaderService,
) -> transformers.WhisperForConditionalGeneration:
    cache_directory = pathlib.Path('data/whisper-model')
    if not cache_directory.exists():
        cache_directory.mkdir(parents=True)
        await model_downloader.act(
            model_name=settings.model_name,
            destination_path=cache_directory,
        )
    torch.set_num_threads(4)
    return transformers.WhisperForConditionalGeneration.from_pretrained(
        settings.model_name,
        cache_dir=cache_directory,
        device_map='auto',
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )


async def provide_whisper_model_for_onnx(
    settings: Settings,
    model_downloader: model_downloader.AbstractModelDownloaderService,
) -> onnxruntime.ORTModelForSpeechSeq2Seq:
    logger.info('Bootstrapping whisper model (onnx runtime)')

    cache_directory = pathlib.Path('data/whisper-model-onnx')
    if not cache_directory.exists():
        logger.info('Loading cache for whisper model (onnx runtime)')
        cache_directory.mkdir(parents=True)
        await model_downloader.act(
            model_name=f'{settings.model_name}-onnx',
            destination_path=cache_directory,
        )

    if not settings.intra_op_num_threads or not settings.inter_op_num_threads:
        physical_cpu = psutil.cpu_count(logical=False) or 4
        settings.intra_op_num_threads = physical_cpu // 2
        settings.inter_op_num_threads = settings.intra_op_num_threads // 2

    logger.info(
        'Will use %s intra-op threads and %s inter-op threads',
        settings.intra_op_num_threads,
        settings.inter_op_num_threads,
    )

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = settings.intra_op_num_threads  # Количество потоков для операций
    session_options.inter_op_num_threads = settings.inter_op_num_threads  # Количество потоков между операциями
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    _whisper_model_for_onnx = onnxruntime.ORTModelForSpeechSeq2Seq.from_pretrained(  # pyright: ignore  # noqa: PGH003
        cache_directory,
        provider='CPUExecutionProvider',
        provider_options={
            'intra_op_num_threads': settings.intra_op_num_threads,
            'inter_op_num_threads': settings.inter_op_num_threads,
        },
        session_options=session_options,
        use_io_binding=True,  # для оптимизации производительности на CPU
        local_files_only=True,
    )
    logger.info('Whisper model (onnx runtime) bootstrapped')
    return _whisper_model_for_onnx


# TODO: maybe remove this later
async def provide_whisper_model_for_vllm() -> 'vllm.LLM':  # type: ignore[name-defined]  # noqa: F821
    import dataclasses

    import vllm  # type: ignore[import-not-found]

    logger.info('Bootstrapping whisper model (vllm runtime)')
    engine_args = vllm.EngineArgs(
        model='antony66/whisper-large-v3-russian',
        max_model_len=448,
        max_num_seqs=5,
        download_dir='data/whisper-model-vllm',
        trust_remote_code=True,
        device='cpu',
    )
    _whisper_model_for_vllm = vllm.LLM(**dataclasses.asdict(engine_args))
    logger.info('Whisper model (vllm runtime) bootstrapped')
    return _whisper_model_for_vllm


async def provide_metadata_repository(
    session_provider: session_provider.AsyncPostgresSessionProvider,
) -> file_metadata.AbstractMetadataRepository:
    return metadata_repository.PostgresMetadataRepository(session_provider=session_provider)


async def provide_file_repository(settings: Settings) -> file_content.AbstractFileContentRepository:
    return file_content_repository.S3FileContentRepository(
        aws_access_key=settings.s3.access_key,
        aws_secret_key=settings.s3.secret_key,
        aws_region=settings.s3.region,
        aws_s3_bucket=settings.s3_bucket_files,
        aws_host=settings.s3.endpoint_url,
    )


async def provide_file_system_repository(
    file_repository: file_content.AbstractFileContentRepository,
    metadata_repository: file_metadata.AbstractMetadataRepository,
) -> file_system.FileSystemRepository:
    return file_system.FileSystemRepository(
        file_content_repository=file_repository,
        metadata_repository=metadata_repository,
    )


async def provide_recognition_service(
    file_system_repository: file_system.FileSystemRepository,
    task_repository: task.AbstractTaskRepository,
    broker: taskiq.AsyncBroker,
) -> recognition_service.AbstractRecognitionService:
    return taskiq_recognition_service.TaskiqRecognitionService(
        file_system_repository=file_system_repository,
        task_repository=task_repository,
        broker=broker,
    )


async def provide_task_repository(
    session_provider: session_provider.AsyncPostgresSessionProvider,
) -> task.AbstractTaskRepository:
    return task_repository.PostgresTaskRepository(
        session_provider=session_provider,
    )


async def provide_inference_runner(  # noqa: PLR0913
    settings: Settings,
    whisper_model_for_onnx: onnxruntime.ORTModelForSpeechSeq2Seq,
    whisper_model_for_pytorch: transformers.WhisperForConditionalGeneration,
    whisper_model_for_vllm: transformers.WhisperForConditionalGeneration,
    whisper_processor: transformers.WhisperProcessor,
    file_system_repository: file_system.FileSystemRepository,
) -> inference_runner.AbstractInferenceRunner:
    match settings.inference_type:
        case 'onnx':
            from speechkit.infrastructure.service.inference_runner import onnx_inference_runner
            return onnx_inference_runner.OnnxInferenceRunnerService(
                model=whisper_model_for_onnx,
                processor=whisper_processor,
                file_system_repository=file_system_repository,
            )
        case 'pytorch':
            from speechkit.infrastructure.service.inference_runner import pytorch_inference_runner
            return pytorch_inference_runner.PyTorchInferenceRunnerService(  # type: ignore[no-any-return, attr-defined]
                model=whisper_model_for_pytorch,
                processor=whisper_processor,
                file_system_repository=file_system_repository,
            )
        case 'vllm':
            from speechkit.infrastructure.service.inference_runner import vllm_inference_runner
            return vllm_inference_runner.VLLMInferenceRunnerService(  # type: ignore[no-any-return, attr-defined]
                model=whisper_model_for_vllm,
                processor=whisper_processor,
                file_system_repository=file_system_repository,
            )
        case _:
            assert_never(settings.inference_type)


sync_provider = Provider(scope=Scope.APP)
sync_provider.provide(provide_settings)
sync_provider.provide(provide_broker)

provider = Provider(scope=Scope.APP)
provider.provide(provide_settings)
provider.provide(provide_metrics_server)
provider.provide(provide_broker)
provider.provide(provide_session_provider)
provider.provide(provide_model_downloader)
provider.provide(provide_whisper_processor)
provider.provide(provide_whisper_model_for_pytorch)
provider.provide(provide_whisper_model_for_onnx)
provider.provide(provide_metadata_repository)
provider.provide(provide_file_repository)
provider.provide(provide_file_system_repository)
provider.provide(provide_recognition_service)
provider.provide(provide_task_repository)
provider.provide(provide_inference_runner)

container = make_async_container(
    provider,
    FastapiProvider(),
    TaskiqProvider(),
)
sync_container = make_container(
    sync_provider,
    TaskiqProvider(),
)
