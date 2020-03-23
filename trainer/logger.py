import os
import logging


def get_logger(job_dir):
    dashes = '-' * 75
    logger = logging.getLogger(job_dir.replace('/', '_'))
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    log_file_path = os.path.join(job_dir, 'log' + '.txt')
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(dashes + job_dir.split('/')[-1] + ': %(asctime)s' + dashes + '\n%(message)s\n')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger
