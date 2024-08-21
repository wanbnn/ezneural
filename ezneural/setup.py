from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='ezneural',
    version='0.1.0',
    description='A library for creating neural networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Wanderson Cavalcante',
    author_email='wanbnn@outlook.com.br',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)