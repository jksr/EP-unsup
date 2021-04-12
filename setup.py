code_name = "epunsup"
readable_name = "EP-unsup"


from setuptools import setup, find_packages

setup(
    name=code_name,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author="Wei Tian",
    author_email="jksr.tw@gmail.com",
    description=readable_name,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url=f'https://github.com/jksr/{readable_name}',
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    packages=find_packages(exclude=('doc',)),
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.tsv', '*.csv', '*.fa', '*Snakefile', '*ipynb']
    },
    #install_requires=['pandas>=1.0',
    #                  'numpy',
    #                  'seaborn==0.10',
    #                  'matplotlib',
    #                  'papermill',
    #                  'dnaio',
    #                  'pysam'],
    entry_points={
    #    'console_scripts': ['enhscr=EnhScr.__main__:main',
    #                        'enhscr-internal=EnhScr._enhscr_internal_cli_:internal_main'],
    }
)
