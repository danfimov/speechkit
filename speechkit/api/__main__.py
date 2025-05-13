import contextlib
import typing as tp

import asgi_correlation_id
import fastapi
import sentry_sdk
from dishka.integrations.fastapi import setup_dishka
from prometheus_async.aio.web import MetricsHTTPServer
from sentry_sdk.integrations import logging as sentry_logging
from starlette_exporter import PrometheusMiddleware
from taskiq import AsyncBroker

from speechkit import dependencies
from speechkit.api import middlewares
from speechkit.api.routers import recognition, system
from speechkit.infrastructure import logs
from speechkit.infrastructure.settings import Settings


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> tp.AsyncGenerator[None, None]:
    settings = dependencies.sync_container.get(Settings)

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
    broker = await app.state.dishka_container.get(AsyncBroker)

    await broker.startup()
    await app.state.dishka_container.get(MetricsHTTPServer | None)

    yield
    await app.state.dishka_container.close()


def get_app() -> fastapi.FastAPI:
    settings = dependencies.sync_container.get(Settings)
    logs.configure_logging(
        log_level=settings.log_level,
        json_log=settings.environment not in ('local', 'unknown', 'development'),
    )
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
    # TODO: Implement auth middleware
    # if settings.enable_auth:
    #     app.add_middleware(
    #         middlewares.AuthMiddleware,
    #         auth_repository=dependencies.get_auth_repository(),
    #         public_paths=['/liveness', '/readiness', '/docs', '/openapi.json'],
    #     )
    app.add_middleware(
        PrometheusMiddleware,
        app_name='speechkit',
        prefix='http_server',
        group_paths=True,
        skip_paths=['/liveness', '/readiness', '/docs', '/redoc', '/openapi.json', '/'],
    )
    app.add_middleware(middlewares.StructLogMiddleware)
    app.add_middleware(asgi_correlation_id.CorrelationIdMiddleware)
    setup_dishka(container=dependencies.container, app=app)
    return app
