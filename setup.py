from setuptools import setup

setup(
    name="ampy",
    version="0.1.0",
    python_requires='>=3.10',
    install_requires=[
        "astropy",
        "emcee",
        "numpy",
        "pandas",
        "scipy",
    ],
    extras_require={
        "plot": ["matplotlib", "arviz", "corner"],
    },
    author='Dylan Dutton',
    description='Afterglow Modeling in Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/astrodyl/skynet-api',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
