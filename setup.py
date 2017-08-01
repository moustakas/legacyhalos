# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='LSLGA',
    url='https://github.com/moustakas/legacyhalos',
    version='untagged',
    author='John Moustakas',
    author_email='jmoustakas@siena.edu',
    #packages=[],
    license=license,
    description='Study of baryons and dark matter halos in Legacy Survey imaging.',
    long_description=readme,
    #package_data={},
    #scripts=,
    #include_package_data=True,
    #install_requires=['numpy']
)
