# -*- coding: utf-8 -*-
import os
import os.path as op
import yaml
import logging.config

from .text_files import read
from ..config import LOG_LEVEL

MODULE_NAME = __name__.split('.')[0]


def setup_logging(log_config_file=op.join(op.dirname(__file__), 'logger.yml'),
                  log_default_level=LOG_LEVEL,
                  env_key=MODULE_NAME.upper() + '_LOG_CFG'):
    """Setup logging configuration."""
    path = log_config_file
    value = os.getenv(env_key, None)
    if value:
        path = value

    if op.exists(path):
        log_cfg = yaml.load(read(path).format(MODULE_NAME))
        logging.config.dictConfig(log_cfg)
        #print('Started logging using config file {0}.'.format(path))
    else:
        logging.basicConfig(level=log_default_level)
        #print('Started default logging. Could not find config file '
        #      'in {0}.'.format(path))
    log = logging.getLogger(__name__)
    log.debug('Start logging.')
