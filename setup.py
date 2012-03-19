#!/usr/bin/env python
import ez_setup
ez_setup.use_setuptools()


from setuptools import setup

setup(name='clinco',
      version='0.01',
      description='clinical components',
      author='Brent Pedersen',
      author_email='bpederse@gmail.com',
      license='MIT',
      url='https://github.com/brentp/clinical-components',
      packages=['clinco', 'clinco.tests'],
      install_requires=['numpy', 'scipy', 'sklearn', 'toolshed'],
      scripts=['clinco/clinco'],
      long_description=open('README.md').read(),
      classifiers=["Topic :: Scientific/Engineering :: Bio-Informatics"],
 )
