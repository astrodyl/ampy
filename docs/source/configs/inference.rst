Inference configuration
=======================

The *inference configuration* controls how AMPy performs Bayesian sampling.
It specifies which sampler to use and the run parameters (walkers, burn-in,
and production length). This file is intentionally separate from the
modeling configuration to keep model definitions and inference settings
independent and reproducible.

Schema
------

The inference configuration is defined under the ``[inference]`` table.

Required fields
~~~~~~~~~~~~~~~

``sampler``
  Sampler type. Must be one of:

  * ``"ensemble"`` — Uses ``emcee.EnsembleSampler``.
  * ``"tempered"`` — Uses the parallel-tempered ``ptemcee``. Recommended for
    most GRB models.

``n_walkers``
  Number of walkers per temperature (or total walkers for the ensemble sampler).

``burn_length``
  Number of burn-in steps.

``run_length``
  Number of production steps after burn-in.

Optional fields
~~~~~~~~~~~~~~~

``n_workers``
  Number of worker threads/processes used to parallelize likelihood evaluations.
  If omitted, the inference will be executed as a single process. Increasing
  the number of workers is only recommended if the posterior calculation is
  expensive to calculate (e.g., more than ` second). The built-in AMPy models
  take only fractions of a milli-second, and thus should only be run with a
  single worker.

Conditional fields
~~~~~~~~~~~~~~~~~~

``n_temps``
  Number of temperatures used by the parallel-tempered sampler.

  This field is **required** when ``sampler = "tempered"`` and must not be
  provided (or is ignored) when ``sampler = "ensemble"``.

Example
-------

.. code-block:: toml

   [inference]
   sampler     = "tempered"
   n_walkers   = 100
   burn_length = 1_000
   run_length  = 1_000
   n_temps     = 10
   # n_workers = 1

Notes
-----

* For parallel tempering, the total number of walkers is typically
  ``n_temps * n_walkers``.
* AMPy records the fully-resolved inference settings in the run outputs
  (e.g., report and metadata) to support reproducibility.
