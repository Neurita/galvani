#!/usr/bin/env python

"""
Install the packages you have listed in the requirements file you input as
first argument.
"""

from   __future__ import (absolute_import, division, print_function,
                          unicode_literals)

import sys
import fileinput
import subprocess
import logging
import uuid
from   pip.req import parse_requirements

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_requirements(*args):
    """Parse all requirements files given and return a list of the dependencies"""
    if len(args) < 1:
        raise ValueError('Some arguments were expected, but received {} arguments.'.format(len(args)))

    install_deps = []
    try:
        for fpath in args:
            install_deps.extend([str(d.req) for d in parse_requirements(fpath, session=uuid.uuid1())])
    except:
        log.exception('Error reading {} file looking for dependencies.'.format(fpath))
        raise

    return [dep for dep in install_deps if dep != 'None']


if __name__ == '__main__':

    if not len(sys.argv):
        log.error('Usage: ./install_deps.py <list of requirements files>')
        exit(-1)

    req_filepaths = []
    for line in fileinput.input():
        req_filepaths = sys.argv[1:]

    try:
        deps = get_requirements(*req_filepaths)
    except:
        log.exception('Error reading files {}.'.format(req_filepaths))
        exit(-1)
    else:
        if not len(deps):
            log.error('After parsing the given files: {}, could not get any '
                      'valid requirement.'.format(req_filepaths))
            exit(-1)

    try:
        for dep_name in deps:
            cmd = "pip install '{0}'".format(dep_name)
            print('#', cmd)
            subprocess.check_call(cmd, shell=True)
    except:
        log.exception('Error installing {}'.format(dep_name))
        exit(-1)
    else:
        log.info('Requirements: {} successfully installed.'.format(deps))
