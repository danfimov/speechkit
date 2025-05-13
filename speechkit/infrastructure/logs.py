import logging

import structlog


_LOGS_CONFIGURED = False

def _drop_color_message_key(
    _: structlog.types.WrappedLogger,
    __: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """
    Processor for dropping the key `color_message` from the event dict if it exists.

    Uvicorn logs the message a second time in the extra `color_message`, but we don't
    need it.
    """
    event_dict.pop('color_message', None)
    return event_dict


def configure_logging(log_level: str = 'INFO', json_log: bool = False) -> None:
    global _LOGS_CONFIGURED  # noqa: PLW0603
    if _LOGS_CONFIGURED:
        return

    timestamper = structlog.processors.TimeStamper(fmt='iso', utc=True)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ExtraAdder(),
        _drop_color_message_key,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]
    if json_log:
        # Format the exception only for JSON logs, as we want to pretty-print them when
        # using the ConsoleRenderer
        shared_processors.append(structlog.processors.format_exc_info)

    structlog.configure(
        processors= [
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    log_renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_log else structlog.dev.ConsoleRenderer()
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # These run ONLY on `logging` entries that do NOT originate within
        # structlog.
        foreign_pre_chain=shared_processors,
        # These run on ALL entries after the pre_chain is done.
        processors=[
            # Remove _record & _from_structlog.
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            log_renderer,
        ],
    )
    # Reconfigure the root logger to use our structlog formatter, effectively emitting the logs via structlog
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

    for _log in ['uvicorn', 'uvicorn.error']:
        # Make sure the logs are handled by the root logger
        logging.getLogger(_log).handlers.clear()
        logging.getLogger(_log).propagate = True

    # Uvicorn logs are re-emitted with more context. We effectively silence them here
    logging.getLogger('uvicorn.access').handlers.clear()
    logging.getLogger('uvicorn.access').propagate = False

    _LOGS_CONFIGURED = True


__all__ = [
    'configure_logging',
]
