"""Logger wrapper for PyTorch experiments.

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import logging

try:
    import ujson as json
except ImportError:
    import json


class LoggerBase(object):
    """Base class for logger
    """
    def __init__(self, **kwargs):
        self.level = kwargs.get('level', logging.INFO)
        self.logger = self.set_logger(**kwargs)

    def set_logger(self, **kwargs):
        """set a logger
        """
        raise NotImplementedError

    def log(self, msg, level=None):
        """log a message.

        Arg:
            msg (str): a message string to logging.
            level (int): a logging level of the message
        """
        raise NotImplementedError

    def pretty_log_dict(self, msg_dict, level=None):
        """log a message from a dictionary object.

        Arg:
            msg_dict (dict): a dictionary to logging.
            prefix (str): a prefix of the message to logging.
            level (int): a logging level of the message.
        """
        self.log(json.dumps(msg_dict, indent=4))

    def log_dict(self, msg_dict, prefix='', pretty=False, level=None):
        """log a message from a dictionary object.

        Arg:
            msg_dict (dict): a dictionary to logging.
            prefix (str): a prefix of the message to logging.
            level (int): a logging level of the message.
        """
        raise NotImplementedError

    def report(self, msg_dict, prefix='', level=None):
        """report a message for a scalar graph.

        Arg:
            msg_dict (dict): a dictionary to logging.
                msg_dict should have property "step".
            level (int): a logging level of the message.
        """
        raise NotImplementedError

    def update_tracker(self, tracker_data, keys=None):
        return

    def insert_to_tracker(self, tracker_data, keys=None):
        return


class PythonLogger(LoggerBase):
    """a logger with Python ``'logging'`` libraray.
    """
    def set_logger(self, name=None, level=None, fmt=None, datefmt=None):
        logger = logging.getLogger(name)
        if level is None:
            level = self.level
        logger.setLevel(level)

        # create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # create formatter and add it to the handlers
        if not fmt:
            # fmt = '[%(asctime)s] [%(module)s:%(funcName)s:%(lineno)d] %(message)s'
            fmt = '[%(asctime)s] %(message)s'
        if not datefmt:
            datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger

    def log(self, msg, level=None):
        if level is None:
            level = self.level
        self.logger.log(level, msg)

    def log_dict(self, msg_dict, prefix='Report @step: ', pretty=False, level=None):
        if 'step' in msg_dict:
            step = msg_dict.pop('step')
            prefix = '{}{:.2f} '.format(prefix, step)
        if pretty:
            msg_dict = json.dumps(msg_dict, indent=4)
        self.log('{}{}'.format(prefix, msg_dict), level=level)

    def report(self, msg_dict, prefix='Report @step', pretty=False, level=None):
        self.log_dict(msg_dict, prefix, pretty=pretty, level=level)
