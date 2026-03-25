import json
import logging
import sys
import time
from contextlib import contextmanager

# Structured JSON logger
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)
logger.addHandler(_handler)
logger.propagate = False


def _log(level: str, event: str, **kwargs):
    record = {"level": level, "event": event, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kwargs}
    logger.info(json.dumps(record))


def info(event: str, **kwargs):
    _log("INFO", event, **kwargs)


def warning(event: str, **kwargs):
    _log("WARNING", event, **kwargs)


def error(event: str, **kwargs):
    _log("ERROR", event, **kwargs)


@contextmanager
def timer(event: str, **kwargs):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = round(time.perf_counter() - start, 3)
        info(event, duration_s=elapsed, **kwargs)
