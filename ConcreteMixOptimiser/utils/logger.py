import datetime
import logging
from typing import Optional

from ConcreteMixOptimiser.utils.constants import LOG_DIR


def get_logger(logger_name: Optional[str] = None, stream: bool = True):
    """
    init the logger, give it proper format, log them both in terminal stream and file
    """
    logging.basicConfig(
        format="%(name)s: %(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "CLIENT: %(name)s | %(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )
    if not logger.hasHandlers() and stream:
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.INFO)
        logger.addHandler(stdout_handler)

    # add file handler
    # logs name will be based current date and time, get a string of it
    filename = datetime.datetime.now().strftime("%Y-%m-%d")

    file_handler = logging.FileHandler(LOG_DIR / f"{filename}.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger
