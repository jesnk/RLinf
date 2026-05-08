from setuptools import setup, find_packages

setup(
    name='liberopro',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['libero'],
    description='LIBERO-PRO compatibility shim: redirects bddl_files/init_files to position-perturbation set.',
)
