import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blast_ct",
    version="0.1",
    author="Miguel Monteiro",
    author_email="miguel.monteiro@imperial.ac.uk",
    description="Automatic segmentation of Traumatic Brain Injury (TBI) in Head CT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biomedia-mira/blast_ct",
    packages=['blast_ct', 'blast_ct.models', 'blast_ct.nifti', 'blast_ct.trainer'],
    package_data={'': ['examples/saved_models/*.pt', 'examples/config.json', 'README.md']},
    entry_points={
        'console_scripts': [
            'blast-ct = blast_ct.console_tool:console_tool'
        ]
    },
    install_requires=[
        'scipy',
        'numpy',
        'pandas',
        'SimpleITK',
        'torch',
        'tensorboard'
    ],
    python_requires='>=3.6',
)
