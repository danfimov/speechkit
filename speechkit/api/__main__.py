import contextlib
import typing as tp

import asgi_correlation_id
import fastapi
import sentry_sdk
from sentry_sdk.integrations import logging as sentry_logging
from starlette_exporter import PrometheusMiddleware

from speechkit import dependencies
from speechkit.api import middlewares
from speechkit.api.routers import recognition, system


@contextlib.asynccontextmanager
async def lifespan(_: fastapi.FastAPI) -> tp.AsyncGenerator[None, None]:
    settings = dependencies.get_settings()
    if settings.sentry is not None:
        sentry_sdk.init(
            dsn=str(settings.sentry.dsn),
            integrations=[
                sentry_logging.LoggingIntegration(
                    level=settings.sentry.logging_level,  # type: ignore[arg-type]
                    event_level=settings.sentry.logging_event_level,  # type: ignore[arg-type]
                ),
            ],
            environment=settings.environment,
            release=settings.app_version,
            ignore_errors=[],
            sample_rate=settings.sentry.sample_rate,
            traces_sample_rate=settings.sentry.traces_sample_rate,
        )
    await dependencies.get_broker().startup()
    await dependencies.get_metrics_server()
    yield
    await dependencies.stop_application()


def get_app() -> fastapi.FastAPI:
    settings = dependencies.get_settings()
    # TODO: rewrite with structlog
    # logs.configure_logging(
    #     path_to_log_config=settings.log_config_path,
    #     root_level=settings.log_level,
    # )

    app = fastapi.FastAPI(
        title='Speechkit',
        summary='API for speech-to-text model',
        version=settings.app_version,
        environment=settings.environment,
        docs_url='/docs',
        redoc_url='/redoc',
        lifespan=lifespan,
    )
    app.include_router(router=recognition.router, prefix='/api/v1')
    app.include_router(router=system.router)

    app.add_middleware(asgi_correlation_id.CorrelationIdMiddleware)
    app.add_middleware(
        PrometheusMiddleware,
        app_name='speechkit',
        prefix='http_server',
        group_paths=True,
        skip_paths=['/liveness', '/readiness', '/docs', '/redoc', '/openapi.json', '/'],
    )

    if settings.enable_auth:
        app.add_middleware(
            middlewares.AuthMiddleware,  # type: ignore[attr-defined]
            auth_repository=dependencies.get_auth_repository(),
            public_paths=['/liveness', '/readiness', '/docs', '/openapi.json'],
        )
    return app
