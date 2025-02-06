from setuptools import setup, find_packages

setup(
    name='fef',
    version='0.1.0',
    author='Lorenzo Rosset, Alessandra Carbone, Aurélien Decelle, Beatriz Seoane',
    maintainer='Lorenzo Rosset',
    author_email='rosset.lorenzo@gmail.com',
    description='Fast and Functional (F&F) structured data generators',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DsysDML/FeF',
    packages=find_packages(include=['fef', 'fef.*']),
    include_package_data=True,
    package_data={
        "fef": ["*.sh"],  # Include all `.sh` files in the `annadca` package
    },
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'fef=fef.cli:main',
        ],
    },
    install_requires=[
        'annadca>=0.1.0',
    ],
)