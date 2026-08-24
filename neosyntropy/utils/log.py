import logging

logger = logging.getLogger("neosyntropy")
logger.setLevel(logging.INFO)

def log_debug(msg: str, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def log_info(msg: str, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def log_warning(msg: str, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def log_error(msg: str, *args, **kwargs):
    logger.error(msg, *args, **kwargs)
