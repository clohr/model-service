import logging  # type: ignore

import structlog  # type: ignore

from server.config import Config

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging(log_level=None):
    """
    Configure structlog to play nice with the stdlib according to these directions:
    http://www.structlog.org/en/stable/standard-library.html#rendering-using-structlog-based-formatters-within-logging
    """
    config = Config()

    shared_processors = [
        create_pennsieve_log_context(config),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Prepare structlog events for the stdlib formatter
    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Aggregate stdlib and structlog and render to JSON
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]

    if log_level is not None:
        env_log_level = LOG_LEVELS.get(log_level.lower().strip(), log_level)
    elif config.log_level is not None:
        env_log_level = LOG_LEVELS.get(
            config.log_level.lower().strip(), config.log_level
        )
    else:
        env_log_level = root_logger.getEffectiveLevel()

    root_logger.setLevel(env_log_level)

    logging.getLogger("connexion.operations.abstract").addFilter(
        IgnoreApiKeyQueryParameter()
    )

    return root_logger


def create_pennsieve_log_context(config: Config):
    """
    Return a structlog processor that conforms to Pennsieve logging
    standards.

    Take all items in the current event dict and nest them under the "pennsieve"
    key.

    https://pennsieve.atlassian.net/wiki/spaces/PLAT/pages/3407908/Logging+Standards+and+Collection#LoggingStandardsandCollection-LoggingFormat
    """

    def inner(logger, name, event_dict):
        event = event_dict.pop("event", None)
        record = event_dict.pop("_record", None)  # Stdlib log message
        exc_info = event_dict.pop("exc_info", None)

        event_dict["service_name"] = config.service_name
        event_dict["environment_name"] = config.environment

        new_event_dict = {"pennsieve": event_dict}

        if event is not None:
            new_event_dict["message"] = event

        if record is not None:
            new_event_dict["_record"] = record

        if exc_info is not None:
            new_event_dict["exc_info"] = exc_info

        new_event_dict["log_level"] = name.upper()

        return new_event_dict

    return inner


class IgnoreApiKeyQueryParameter(logging.Filter):
    """
    Connexion logs an error message because the `api_key` query parameter is not
    stripped by the gateway.

    This logging filter drops all these messages.

    This is a stdlib filter instead of a structlog filter because structlog
    cannot drop messages in `foreign_pre_chain`. See
    https://github.com/hynek/structlog/issues/113
    """

    def filter(self, record):
        return (
            record.getMessage()
            != "Function argument 'api_key' not defined in specification"
        )
