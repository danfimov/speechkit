import logging
import os
from ssl import SSLContext

from aiohttp import hdrs, web
from prometheus_async.aio.web import MetricsHTTPServer, _choose_generator
from prometheus_client import REGISTRY, CollectorRegistry, multiprocess


logger = logging.getLogger(__name__)


def server_stats(request: web.Request) -> web.Response:
    """
    Copy of prometheus_async.aio.web.server_stats with modifications.

    Added support for multiprocess mode, https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn.

    Returns web.Response with text representation of metrics.
    """
    if os.environ.get('PROMETHEUS_MULTIPROC_DIR'):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
    else:
        registry = REGISTRY

    generate, content_type = _choose_generator(request.headers.get(hdrs.ACCEPT))

    rsp = web.Response(body=generate(registry))
    # This is set separately because aiohttp complains about `;` in
    # content_type thinking it means there's also a charset.
    # cf. https://github.com/aio-libs/aiohttp/issues/2197
    rsp.content_type = content_type

    return rsp


async def get_ping_response(request: web.Request) -> web.Response:
    logger.debug('Pong for %s', request.remote)
    return web.Response(text='200 VERY OK')


async def get_metric_response(request: web.Request) -> web.Response:
    logger.debug('Show metrics for %s', request.remote)
    return server_stats(request)


async def start_metric_server(
    *,
    addr: str = '',
    port: int = 8001,
    ssl_ctx: SSLContext | None = None,
) -> MetricsHTTPServer:
    """
    Start an HTTP(S) server on *addr*:*port*.

    If *ssl_ctx* is set, use TLS.

    :param str addr: Interface to listen on. Leaving empty will listen on all
        interfaces.
    :param int port: Port to listen on.
    :param ssl.SSLContext ssl_ctx: TLS settings

    :rtype: MetricsHTTPServer
    """
    app = web.Application()
    app.router.add_get('/metrics', get_metric_response)
    app.router.add_get('/ping', get_ping_response)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, addr, port, ssl_context=ssl_ctx)
    await site.start()

    ms = MetricsHTTPServer.from_server(
        runner=runner, app=app, https=ssl_ctx is not None,
    )

    logger.info('Metric server run on %s port', port)
    return ms


__all__ = ['start_metric_server']
