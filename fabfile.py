#!/usr/bin/env python

"""
Install the packages you have listed in the requirements file you input as
first argument.
"""
from   __future__ import (absolute_import, division, print_function, unicode_literals)

try:
    from invoke     import task
    from invoke     import run as local
except:
    from fabric.api import task, local

import os
import os.path    as     op
import shutil

from   glob       import glob
from   setuptools import find_packages
from   pip.req    import parse_requirements

# Get version without importing, which avoids dependency issues
module_name    = find_packages(exclude=['tests'])[0]
version_pyfile = op.join(module_name, 'version.py')
exec(compile(open(version_pyfile).read(), version_pyfile, 'exec'))

# get current dir
CWD = op.realpath(op.curdir)

#ignore dirs
IGNORE = ['.git', '.idea']


def get_requirements(*args):
    """Parse all requirements files given and return a list of the dependencies"""
    install_deps = []
    try:
        for fpath in args:
            install_deps.extend([str(d.req or d.url) for d in parse_requirements(fpath)])
    except:
        print('Error reading {} file looking for dependencies.'.format(fpath))

    return [dep for dep in install_deps if dep != 'None']


def recursive_glob(base_directory, regex=None):
    """Uses glob to find all files that match the regex in base_directory.

    @param base_directory: string

    @param regex: string

    @return: set
    """
    if regex is None:
        regex = ''

    files = glob(os.path.join(base_directory, regex))
    for path, dirlist, filelist in os.walk(base_directory):
        for ignored in IGNORE:
            try:
                dirlist.remove(ignored)
            except:
                pass

        for dir_name in dirlist:
            files.extend(glob(os.path.join(path, dir_name, regex)))

    return files


def recursive_remove(work_dir=CWD, regex='*'):
    [os.remove(fn) for fn in recursive_glob(work_dir, regex)]


def recursive_rmtrees(work_dir=CWD, regex='*'):
    [shutil.rmtree(fn, ignore_errors=True) for fn in recursive_glob(work_dir, regex)]


@task
def install_deps(req_filepaths = ['requirements.txt']):
    # for line in fileinput.input():
    deps = get_requirements(*req_filepaths)

    try:
        for dep_name in deps:
            cmd = "pip install '{0}'".format(dep_name)
            print('#', cmd)
            local(cmd)
    except:
        print('Error installing {}'.format(dep_name))


@task
def version():
    print(__version__)


@task
def install():
    clean()
    install_deps()
    local('python setup.py install')


@task
def develop():
    clean()
    install_deps()
    local('python setup.py develop')


@task
def clean(work_dir=CWD):
    clean_build(work_dir)
    clean_pyc(work_dir)


@task
def clean_build(work_dir=CWD):
    shutil.rmtree('build', ignore_errors=True)
    shutil.rmtree('dist', ignore_errors=True)
    shutil.rmtree('.eggs', ignore_errors=True)
    recursive_rmtrees(work_dir, '__pycache__')
    recursive_rmtrees(work_dir, '*.egg-info')
    recursive_rmtrees(work_dir, '*.egg')
    recursive_rmtrees(work_dir, '.ipynb_checkpoints')


@task
def clean_pyc(work_dir=CWD):
    recursive_remove(work_dir, '*.pyc')
    recursive_remove(work_dir, '*.pyo')
    recursive_remove(work_dir, '*~')


@task
def lint():
    local('flake8 ' + module_name + ' test')


@task
def test(filepath=''):
    if filepath:
        if not op.exists(filepath):
            print('Error: could not find file {}'.format(filepath))
            exit(-1)
        cmd = 'python setup.py test -a ' + filepath
    else:
        cmd = 'python setup.py test'

    local(cmd)


@task
def test_all():
    local('tox')


@task
def coverage():
    local('coverage local --source ' + module_name + ' setup.py test')
    local('coverage report -m')
    local('coverage html')
    local('open htmlcov/index.html')


@task
def docs(doc_type='html'):
    os.remove(op.join('docs', module_name + '.rst'))
    os.remove(op.join('docs', 'modules.rst'))
    local('sphinx-apidoc -o docs/ ' + module_name)
    os.chdir('docs')
    local('make clean')
    local('make ' + doc_type)
    os.chdir(CWD)
    local('open docs/_build/html/index.html')


@task
def release():
    clean()
    local('pip install -U pip setuptools twine wheel')
    local('python setup.py sdist bdist_wheel')
    #local('python setup.py bdist_wheel upload')
    local('twine upload dist/*')


@task
def sdist():
    clean()
    local('python setup.py sdist')
    local('python setup.py bdist_wheel upload')
    print(os.listdir('dist'))
