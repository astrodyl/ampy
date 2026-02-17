Samplers
========

AMPy uses Markov Chain Monte Carlo (MCMC) sampling to estimate posterior
distributions of model parameters given observational data. Samplers explore
the parameter space according to the posterior probability:

.. math::

   \log p(\theta \mid D) \propto \log \mathcal{L}(D \mid \theta)
   + \sum_i \log p_i(\theta_i)

AMPy provides high-level sampler classes that wrap well-established MCMC
implementations while integrating tightly with the AMPy modeling framework.
These samplers handle parameter transformations, priors, and model evaluation
automatically, so users typically interact with them only through the
:class:`ampy.AMPy` interface or configuration files.

Two samplers are currently available:

* :class:`~ampy.inference.samplers.EnsembleSampler` — affine-invariant ensemble
  sampler based on ``emcee``
* :class:`~ampy.inference.samplers.PTSampler` — parallel-tempered ensemble
  sampler based on ``ptemcee``

Choosing a sampler
------------------

**EnsembleSampler**
   Efficient for moderately correlated posteriors and typical afterglow
   parameter spaces.

**PTSampler**
   Recommended for highly multimodal or degenerate posteriors. Parallel
   tempering allows chains to move between separated probability regions
   at the cost of additional computation.

Configuration
-------------

The sampler is selected in the inference configuration:

.. code-block:: toml

   [inference]
   sampler     = "ensemble"
   n_walkers   = 100
   burn_length = 1000
   run_length  = 2000

For parallel tempering:

.. code-block:: toml

   [inference]
   sampler     = "tempered"
   n_walkers   = 100
   n_temps     = 10
   burn_length = 1000
   run_length  = 2000


.. automodule:: ampy.inference.samplers
   :members:
   :undoc-members: False
   :show-inheritance: