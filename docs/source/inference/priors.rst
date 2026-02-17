Priors
======

AMPy uses Bayesian inference to estimate model parameters from observational
data. In this framework, each parameter is assigned a *prior distribution*
that encodes physically motivated constraints or prior knowledge about the
parameter before considering the data.

All priors in AMPy follow a common interface and return **log-densities**
(:math:`\log p(\theta)`) so that they can be combined directly with the
log-likelihood during MCMC sampling:

.. math::

   \log p(\theta \mid D) \propto \log \mathcal{L}(D \mid \theta)
   + \sum_i \log p_i(\theta_i)

Priors in AMPy serve two purposes:

* Constrain parameters to physically meaningful ranges
* Encode external knowledge

Each prior provides:

* A normalized probability density
* A log-density evaluation used during inference
* A sampling method for initializing MCMC walkers
* Serialization for configuration files

.. automodule:: ampy.inference.priors
   :members:
   :exclude-members: Prior, prior_factory
   :undoc-members: False
   :show-inheritance: