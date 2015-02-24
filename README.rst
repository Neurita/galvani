.. -*- mode: rst -*-

luigi
=====

Brain functional MRI analysis tools

Named after the Luigi Aloisio Galvani (1737 – 1798) was an Italian physician, physicist and philosopher. He discovered that the muscles of dead frogs legs twitched when struck by an electrical spark.This was one of the first forays into the study of bioelectricity, a field that still studies the electrical patterns and signals of the nervous system. The beginning of Galvani's experiments with bioelectricity has a popular legend which says that Galvani was slowly skinning a frog at a table where he had been conducting experiments with static electricity by rubbing frog skin. The observation made Galvani the first investigator to appreciate the relationship between electricity and animation — or life.

.. image:: https://secure.travis-ci.org/neurita/luigi.png?branch=master
    :target: https://travis-ci.org/neurita/luigi

.. image:: https://coveralls.io/repos/neurita/luigi/badge.png
    :target: https://coveralls.io/r/neurita/luigi


Dependencies
============

Please see the requirements.txt and pip_requirements.txt file.

Install
=======

This package uses setuptools and Makefiles. 

I've made a workaround to deal with build dependencies of some requirements.
So there are two requirements files: requirements.txt and pip-requirements.txt.
The requirements.txt dependencies must be installed one by one, with::

    make install_deps

The following command will install everything with all dependencies::

    make install
    
If you already have the dependencies listed in requirements.txt installed, 
to install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

You can also install it in development mode with::

    make develop


Development
===========

Code
----

Github
~~~~~~

You can check the latest sources with the command::

    git clone https://www.github.com/neurita/luigi.git

or if you have write privileges::

    git clone git@github.com:neurita/luigi.git

If you are going to create patches for this project, create a branch for it 
from the master branch.

The stable releases are tagged in the repository.


Testing
-------

TODO
