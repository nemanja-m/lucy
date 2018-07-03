import io
import os
import re

from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


VERSION = find_version('lucy', '__init__.py')
LONG_DESCRIPTION = read('README.md')

setup_info = dict(
    name='lucy-bot',
    version=VERSION,
    author='Nemanja Milicevic',
    author_email='the.nemanja.milicevic@gmail.com',
    url='https://github.com/nemanja-m/lucy',
    description='Chat bot based on Key-Value Memory Networks',
    long_description=LONG_DESCRIPTION,
    license='MIT',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'lucy-train = lucy.train:main',
            'lucy-chat = lucy.chat:main'
        ]
    },
    include_package_data=True,
    install_requires=[
        'numpy',
        'revtok',
        'six',
        'torch==0.4.0',
        'torchtext==0.3.0',
        'tqdm',
    ],
    dependency_links=[
        'git+https://github.com/pytorch/text.git@master#egg=torchtext-0.3.0'
    ]
)

setup(**setup_info)
