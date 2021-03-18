import logging

import structlog


"""
These small tests do not assert anything due to the difficulty of hooking into
Python's logging with py.test, but can be used to verify the format of log
messages by running

    py.test tests/tests_logging.py -s

"""


def test_logging(client):
    logging.getLogger("stdlog").warning("warning, warning")
    structlog.get_logger("structlog").warning("difficulties", datasetId=123)


def test_error_logging():
    try:
        {}.pop(1)
    except KeyError as e:
        logging.getLogger("stdlib").error("error", exc_info=True)
