# -*- coding: utf-8 -*-
from setuptools import setup

packages = ['itk_transformer_nlp']
package_data = {'': ['*']}
install_requires = ['datasets>=1.18.4,<2.0.0',
                    'torch>=1.10.2,<2.0.0',
                    'transformers>=4.17.0,<5.0.0']

with open('README.md', 'r', encoding='utf-8') as readme:
    long_description = readme.read()

setup_kwargs = {
    'name': 'itk-transformer-nlp',
    'version': '0.1.0',
    'description': 'A package for learning NLP with Transformers',
    'long_description': long_description,
    'author': 'nyekibence',
    'author_email': 'nyeki.bence96@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/nyekibence/ITK-Transformer-NLP',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<4.0',
}

setup(**setup_kwargs)
