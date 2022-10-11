import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blast_ct",
    version="2.0.0",
    author="Miguel Monteiro",
    author_email="miguel.monteiro@imperial.ac.uk",
    description="Automatic segmentation of Traumatic Brain Injury (TBI) in Head CT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biomedia-mira/blast_ct",
    packages=['blast_ct', 'blast_ct.models', 'blast_ct.nifti', 'blast_ct.trainer', 'blast_ct.localisation'],
    package_data={'': ['data/saved_models/*.torch_model', 'data/config.json', 'README.md',
                       'data/localisation_files/*.nii.gz',
                       'data/localisation_files/atlas_labels.csv']},
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
        'SimpleITK==1.2.0',
        'torch',
        'tensorboard'
    ],
    python_requires='>=3.6',
    setup_requires=['setuptools_scm']

)
