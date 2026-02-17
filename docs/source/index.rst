.. AMPy documentation master file, created by
   sphinx-quickstart on Fri Feb  6 13:54:09 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AMPy
====

AMPy (**A**\fterglow **M**\odeling in **Py**\thon) is a complete inference package
for modeling Gamma-ray burst afterglows. AMPy’s modular design allows easy
integration of arbitrary third-party afterglow and extinction models, enabling
use beyond the builtin generalized forward-shock implementations. The framework
supports integrated-flux calculations for arbitrary bandpasses, spectral-index
computation, and multi-band fitting over broadband light curves and spectra.

It interfaces seamlessly with an MCMC sampling backend, providing flexible
prior definitions, likelihood customization, and full posterior analysis tools.
By combining physical realism with statistical rigor, AMPy serves as an
end-to-end toolkit for parameter estimation, model testing, and interpreting
the environments and physics of GRBs.

Installation
============

The code is not published to PyPI yet, so please install it from source:

.. code-block:: bash

   git clone https://github.com/astrodyl/ampy.git
   cd ampy
   pip install -e .

.. toctree::
   :maxdepth: 1
   :caption: Contents

   quickstart
   configs/index
   inference/index
   plotting/index