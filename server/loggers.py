import logging
import os
import sys
import warnings


def init_logger(logger_name, level=logging.INFO) -> logging.Logger:
    if not sys.warnoptions:
        warnings.simplefilter('ignore')

    curr_pid = os.getpid()
    log_format = (
        f'({curr_pid}): '
        '[%(levelname)s]: '
        '%(asctime)s - '
        '%(name)s - '
        '%(message)s')
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(log_format)
    ch.setFormatter(formatter)
    _logger.addHandler(ch)
    # TODO: add file handlers
    return _logger


logger = init_logger('APP')
