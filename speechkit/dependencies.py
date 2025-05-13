import contextlib
import dataclasses
import pathlib
from urllib.parse import quote

import onnxruntime as ort  # type: ignore[import-untyped]
import prometheus_async.aio.web
import psutil
import structlog
import taskiq
import taskiq.abc.broker
import taskiq_aio_pika
import taskiq_fastapi
import taskiq_pipelines
import torch
import transformers  # type: ignore[import-untyped]
from optimum import onnxruntime  # type: ignore[import-untyped]
from taskiq_pg.asyncpg import AsyncpgResultBackend

from speechkit.domain.repository import authentication, file_content, file_metadata, file_system, task
from speechkit.domain.service import inference_runner, model_downloader, recognition_service
from speechkit.infrastructure import logs, metrics, settings
from speechkit.infrastructure.database import session_provider
from speechkit.infrastructure.repository import (
    auth_storage,
    file_content_repository,
    metadata_repository,
    task_repository,
)
from speechkit.infrastructure.service import recognition_service as taskiq_recognition_service


logger = structlog.get_logger(__name__)
_exit_stack = contextlib.AsyncExitStack()

_settings: settings.Settings | None = None
_metrics_server: prometheus_async.aio.web.MetricsHTTPServer | None = None

# Tasks
_broker: taskiq.AsyncBroker | None = None
_task_repository: task.AbstractTaskRepository | None = None

# Auth
_auth_repository: authentication.AbstractAuthenticationRepository | None = None

# Services
_model_downloader: model_downloader.AbstractModelDownloaderService | None = None
_recognition_service: recognition_service.AbstractRecognitionService | None = None


# Inference
_inference_runner: inference_runner.AbstractInferenceRunner | None = None
_whisper_processor: transformers.WhisperProcessor | None = None
_whisper_model_for_pytorch: transformers.WhisperForConditionalGeneration | None = None
_whisper_model_for_onnx: onnxruntime.ORTModelForSpeechSeq2Seq | None = None
_whisper_model_for_vllm = None

# Storage for files
_file_repository: file_content.AbstractFileContentRepository | None = None
_metadata_repository: file_metadata.AbstractMetadataRepository | None = None
_file_system_repository: file_system.FileSystemRepository | None = None

_session_provider: session_provider.AsyncPostgresSessionProvider | None = None


def get_settings() -> settings.Settings:
    global _settings
    if _settings is None:
        _settings = settings.Settings()
    return _settings


async def get_metrics_server() -> prometheus_async.aio.web.MetricsHTTPServer | None:
    global _metrics_server
    if _metrics_server is None:
        settings = get_settings()
        if settings.metrics_port:
            _metrics_server = await metrics.start_metric_server(
                port=settings.metrics_port,
            )
            _exit_stack.push_async_callback(_metrics_server.close)
    return _metrics_server



def get_session_provider() -> session_provider.AsyncPostgresSessionProvider:
    global _session_provider
    if _session_provider is None:
        settings = get_settings()
        _session_provider = session_provider.AsyncPostgresSessionProvider(
            connection_settings=settings.postgres,
        )
    return _session_provider


def get_broker() -> taskiq.AsyncBroker:
    from speechkit.broker import middlewares, tasks

    global _broker
    if _broker is None:
        settings = get_settings()
        if settings.broker_type == 'amqp' and settings.amqp:
            logger.info('Use AioPikaBroker')
            _broker = taskiq_aio_pika.AioPikaBroker(
                    url=settings.amqp.get_url(),
                ).with_result_backend(
                    AsyncpgResultBackend(
                        f'postgresql://{settings.postgres.user}:{quote(settings.postgres.password.get_secret_value())}@{settings.postgres.host}:{settings.postgres.port}/{settings.postgres.database}',
                    ),
                ).with_middlewares(
                    middlewares.PrometheusMiddleware(),
                    middlewares.StructlogMiddleware(),
                )
        else:
            logger.info('Use InMemoryBroker')
            _broker = taskiq.InMemoryBroker()
        _broker.add_middlewares(taskiq_pipelines.PipelineMiddleware())
        _broker.register_task(tasks.run_inference)
        if settings.app_name == 'api':  # TODO: Fix errors with FastAPI when running worker
            taskiq_fastapi.init(_broker, 'speechkit.api.__main__:get_app')

        @_broker.on_event(taskiq.TaskiqEvents.WORKER_STARTUP)
        async def startup(_: taskiq.TaskiqState) -> None:
            logs.configure_logging(
                log_level=settings.log_level,
                json_log=settings.environment not in ('local', 'unknown', 'development'),
            )
            await get_metrics_server()
            await preconfigure_inference_runner()

        _exit_stack.push_async_callback(_broker.shutdown)
        logger.info('Broker initialized')
    return _broker


def get_model_downloader() -> model_downloader.AbstractModelDownloaderService:
    from speechkit.infrastructure.service.model_downloader import HuggingFaceModelDownloader, S3ModelDownloader

    global _model_downloader
    if _model_downloader is None:
        settings = get_settings()
        if settings.downloader_type == 'huggingface':
            _model_downloader = HuggingFaceModelDownloader()
        else:
            _model_downloader = S3ModelDownloader(
                aws_access_key=settings.s3.access_key,
                aws_secret_key=settings.s3.secret_key,
                aws_region=settings.s3.region,
                aws_s3_bucket=settings.s3_bucket_models,
                aws_host=settings.s3.endpoint_url,
            )
        _exit_stack.push_async_callback(_model_downloader.close)
    return _model_downloader


async def get_whisper_processor() -> transformers.WhisperProcessor:
    global _whisper_processor
    if _whisper_processor is None:
        logger.info('Bootstrapping whisper processor')
        settings = get_settings()
        cache_directory = pathlib.Path('data/whisper-model')
        if not cache_directory.exists():
            logger.info('Loading cache for whisper processor')
            cache_directory.mkdir(parents=True)
            await get_model_downloader().act(
                model_name=settings.model_name,
                destination_path=cache_directory,
            )
        processor = transformers.WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path=settings.model_name,
            cache_dir=cache_directory,
            local_files_only=True,
        )
        # can be tuple if pass_unused_kwargs=True
        _whisper_processor = processor[0] if isinstance(processor, tuple) else processor
        logger.info('Whisper processor bootstrapped')
    return _whisper_processor


async def get_whisper_model_for_pytorch() -> transformers.WhisperForConditionalGeneration:
    global _whisper_model_for_pytorch
    if _whisper_model_for_pytorch is None:
        logger.info('Bootstrapping whisper model (pytorch runtime)')
        settings = get_settings()
        cache_directory = pathlib.Path('data/whisper-model')
        if not cache_directory.exists():
            cache_directory.mkdir(parents=True)
            await get_model_downloader().act(
                model_name=f'{settings.model_name}',
                destination_path=cache_directory,
            )

        torch.set_num_threads(4)  # ограничение количества потоков
        _whisper_model_for_pytorch = transformers.WhisperForConditionalGeneration.from_pretrained(
            settings.model_name,
            cache_dir=cache_directory,
            device_map='auto',  # автоматическое распределение по доступной памяти
            torch_dtype=torch.float32,  # использование float32 вместо float64
            low_cpu_mem_usage=True,  # оптимизация использования памяти
            local_files_only=True,
        )
        logger.info('Whisper model (pytorch runtime) bootstrapped')
    return _whisper_model_for_pytorch


async def get_whisper_model_for_onnx() -> onnxruntime.ORTModelForSpeechSeq2Seq:
    global _whisper_model_for_onnx
    if _whisper_model_for_onnx is None:
        settings = get_settings()
        logger.info('Bootstrapping whisper model (onnx runtime)')

        cache_directory = pathlib.Path('data/whisper-model-onnx')
        if not cache_directory.exists():
            logger.info('Loading cache for whisper model (onnx runtime)')
            cache_directory.mkdir(parents=True)
            await get_model_downloader().act(
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


async def get_whisper_model_for_vllm():  # type: ignore[no-untyped-def]
    import vllm  # type: ignore[import-not-found]

    global _whisper_model_for_vllm
    if _whisper_model_for_vllm is None:
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


def get_metadata_repository() -> file_metadata.AbstractMetadataRepository:
    global _metadata_repository
    if _metadata_repository is None:
        _metadata_repository = metadata_repository.PostgresMetadataRepository(
            session_provider=get_session_provider(),
        )
        logger.info('Metadata repository loaded')
    return _metadata_repository


async def get_file_repository() -> file_content.AbstractFileContentRepository:
    global _file_repository
    if _file_repository is None:
        logger.info('Bootstrapping file repository')
        settings = get_settings()
        _file_repository = file_content_repository.S3FileContentRepository(
            aws_access_key=settings.s3.access_key,
            aws_secret_key=settings.s3.secret_key,
            aws_region=settings.s3.region,
            aws_s3_bucket=settings.s3_bucket_files,
            aws_host=settings.s3.endpoint_url,
        )
        logger.info('File repository bootstrapped')
    return _file_repository


async def get_file_system_repository() -> file_system.FileSystemRepository:
    global _file_system_repository
    if _file_system_repository is None:
        _file_system_repository = file_system.FileSystemRepository(
            file_content_repository=await get_file_repository(),
            metadata_repository=get_metadata_repository(),
        )
        logger.info('File system repository loaded')
    return _file_system_repository


def get_auth_repository() -> authentication.AbstractAuthenticationRepository:
    global _auth_repository
    if _auth_repository is None:
        _auth_repository = auth_storage.PostgresAuthenticationRepository(
            session_provider=get_session_provider(),
        )
        logger.info('Authentication repository loaded')
    return _auth_repository


async def get_inference_runner() -> inference_runner.AbstractInferenceRunner:
    global _inference_runner
    if _inference_runner is None:
        settings = get_settings()
        if settings.inference_type == 'pytorch':
            from speechkit.infrastructure.service.inference_runner import pytorch_inference_runner

            _inference_runner = pytorch_inference_runner.PytorchInferenceRunnerService(
                model=await get_whisper_model_for_pytorch(),
                processor=await get_whisper_processor(),
                file_system_repository=await get_file_system_repository(),
            )
        elif settings.inference_type == 'onnx':
            from speechkit.infrastructure.service.inference_runner import onnx_inference_runner


            _inference_runner = onnx_inference_runner.OnnxInferenceRunnerService(
                model=await get_whisper_model_for_onnx(),
                processor=await get_whisper_processor(),
                file_system_repository=await get_file_system_repository(),
            )
        else:
            from speechkit.infrastructure.service.inference_runner import vllm_inference_runner


            _inference_runner = vllm_inference_runner.VllmInferenceRunnerService(
                model=await get_whisper_model_for_vllm(),  # type: ignore[no-untyped-call]
                processor=None,
                file_system_repository=await get_file_system_repository(),
            )
    return _inference_runner


def get_task_repository() -> task.AbstractTaskRepository:
    global _task_repository
    if _task_repository is None:
        _task_repository = task_repository.PostgresTaskRepository(
            session_provider=get_session_provider(),
        )
    return _task_repository


async def get_recognition_service() -> recognition_service.AbstractRecognitionService:
    global _recognition_service
    if _recognition_service is None:
        _recognition_service = taskiq_recognition_service.TaskiqRecognitionService(
            file_system_repository=await get_file_system_repository(),
            task_repository=get_task_repository(),
            broker=get_broker(),
        )
    return _recognition_service


async def preconfigure_inference_runner() -> None:
    """Load model and processor into memory at application start, to avoid loading impact on the first requests."""
    settings = get_settings()
    if settings.inference_type == 'pytorch':
        await get_whisper_model_for_pytorch()
        await get_whisper_processor()
    elif settings.inference_type == 'onnx':
        await get_whisper_model_for_onnx()
        await get_whisper_processor()
    else:
        await get_whisper_model_for_vllm()  # type: ignore[no-untyped-call]
    logger.info('Inference runner preconfigured')


async def stop_application() -> None:
    logger.info('Stopping application')
    await _exit_stack.aclose()
    logger.info('Application stopped')
