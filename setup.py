import setuptools

setuptools.setup(name='scProjection',
    version='0.22',
    description='Projection and Deconvolution using deep heirarchical and generative neural network.',
    url='https://github.com/ucdavis/quonlab/tree/master/development/deconvAllen',
    author='Nelson Johansen, Gerald Quon',
    author_email='njjohansen@ucdavis.edu, gquon@ucdavis.edu',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'tensorflow',
        'tensorflow_probability',
        'sklearn-learn',
        'numpy'
      ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6")
