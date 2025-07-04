import time
import typing as tp

import structlog
from asgi_correlation_id import correlation_id
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from uvicorn.protocols.utils import get_path_with_query_string


logger = structlog.get_logger(__name__)


class AccessInfo(tp.TypedDict, total=False):
    status_code: int
    start_time: float


class StructLogMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # If the request is not an HTTP request, we don't need to do anything special
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=correlation_id.get())

        info = AccessInfo()

        # Inner send function
        async def inner_send(message: tp.MutableMapping[str, tp.Any]) -> None:
            if message['type'] == 'http.response.start':
                info['status_code'] = message['status']
            await send(message)

        try:
            info['start_time'] = time.perf_counter_ns()
            await self.app(scope, receive, inner_send)
        except Exception as e:
            logger.exception(
                'An unhandled exception was caught by last resort middleware',
                exception_class=e.__class__.__name__,
                exc_info=e,
                stack_info=True,
            )
            info['status_code'] = 500
            response = JSONResponse(
                status_code=500,
                content={
                    'error': 'Internal Server Error',
                    'message': 'An unexpected error occurred.',
                },
            )
            await response(scope, receive, send)
        finally:
            process_time = time.perf_counter_ns() - info['start_time']
            client_host, client_port = scope['client']
            http_method = scope['method']
            http_version = scope['http_version']
            url = get_path_with_query_string(scope)  # type: ignore[arg-type]

            if scope['path'] not in {'/', '/readiness', '/liveness', '/docs', '/redoc', '/openapi.json'}:
                # Recreate the Uvicorn access log format, but add all parameters as structured information
                logger.info(
                    f'{client_host}:{client_port} - "{http_method} {scope["path"]} HTTP/{http_version}" {info["status_code"]}',  # noqa: E501, G004
                    http={
                        'url': str(url),
                        'status_code': info['status_code'],
                        'method': http_method,
                        'request_id': correlation_id.get(),
                        'version': http_version,
                    },
                    network={'client': {'ip': client_host, 'port': client_port}},
                    duration=process_time,
                )

