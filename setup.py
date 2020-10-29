import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blast_ct",
    version="0.1.1",
    author="Miguel Monteiro",
    author_email="miguel.monteiro@imperial.ac.uk",
    description="Automatic segmentation of Traumatic Brain Injury (TBI) in Head CT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biomedia-mira/blast_ct",
    packages=['blast_ct', 'blast_ct.models', 'blast_ct.nifti', 'blast_ct.trainer'],
    package_data={'': ['data/saved_models/*.pt', 'data/config.json', 'README.md']},
    entry_points={
        'console_scripts': [
            'blast-ct = blast_ct.console_tool:console_tool',
            'blast-ct-inference = blast_ct.inference:inference',
            'blast-ct-train = blast_ct.train:train'
        ]
    },
    install_requires=[
        'scipy',
        'numpy',
        'pandas',
        'SimpleITK==1.2.4',
        'torch',
        'tensorboard'
    ],
    python_requires='>=3.6',
)
