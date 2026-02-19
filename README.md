# AMPy

AMPy (Afterglow Modeling in Python) is a complete inference package for modeling Gamma-ray burst afterglows. AMPy’s modular design allows easy integration of arbitrary third-party afterglow and extinction models, enabling use beyond the builtin generalized forward-shock implementations. The framework supports integrated-flux calculations for arbitrary bandpasses, spectral-index computation, and multi-band fitting over broadband light curves and spectra.

It interfaces seamlessly with an MCMC sampling backend, providing flexible prior definitions, likelihood customization, and full posterior analysis tools. By combining physical realism with statistical rigor, AMPy serves as an end-to-end toolkit for parameter estimation, model testing, and interpreting the environments and physics of GRBs.

Documentation can be found here: https://ampy-docs.readthedocs.io/en/latest/quickstart.html

# Installation
The code is not published to PyPI yet, so please install it from source:

```
git clone https://github.com/astrodyl/ampy.git
cd ampy
pip install -e .
```

# Quickstart
Create an ``AMPy`` instance from a run configuration and perform MCMC sampling:

```Python
from ampy import AMPy

# Create AMPy instance from run configuration
ampy = AMPy.from_toml("path/to/config/run.toml")

# Run MCMC and obtain the most likely parameters
best = ampy.run_mcmc(
    nwalkers=100,        # Number of walkers
    iterations=1000,     # Number of production iterations
    burn=1000,           # Number of burn-in iterations
    sampler="tempered",  # Parallel tempered sampler
    ntemps=10,           # Number of temperatures
)

ampy.generate_products("output/directory")
```

The returned ``best`` dictionary contains the maximum-posterior parameter values from the inference run.
