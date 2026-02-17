Quickstart
==========

This quickstart demonstrates the typical AMPy workflow:

1. Load a run configuration
2. Perform MCMC inference
3. Generate standard plots and outputs

Running AMPy requires a run configuration file (``run.toml``) that specifies
the input data, modeling configuration, inference settings, and output
location. See :doc:`configs/run` for details.

Load and run inference
----------------------

Create an :class:`~ampy.AMPy` instance from a run configuration and perform
MCMC sampling:

.. code-block:: python

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

The returned ``best`` dictionary contains the maximum-posterior parameter
values from the inference run.

Generate outputs
----------------

AMPy provides several built-in data products and plots. After inference,
these can be generated directly from the :class:`~ampy.AMPy` instance either
using ``ampy.generate_products(output_dir)`` as in the above example, or
individually like the example below.

.. code-block:: python

   from pathlib import Path

   output_dir = Path("output/directory")

   # JSON summary report
   ampy.summary(output_dir / "report.json")

   # Light curve (brightness vs. time)
   ampy.light_curve(path=output_dir / "light_curve.pdf")

   # Spectral plot (spectral breaks and indices)
   ampy.spectral_plot(path=output_dir / "spectral_plot.pdf")

   # Density profile (density vs. blast radius)
   ampy.density_profile(path=output_dir / "density_profile.pdf")

   # Corner plot (posterior distributions)
   ampy.corner_plot(path=output_dir / "corner.pdf")

The plotting routines automatically use the best-fit parameters from the most
recent inference run.

If it is desirable to view the plot immediately rather than save it to a file,
simply omit ``path``.

.. code-block:: python

   fig, ax = ampy.light_curve()
   plt.show()

Next steps
----------

* Learn about configuration files: :doc:`configs/index`
* Learn about the inference routine: :doc:`inference/index`
* Explore available plots: :doc:`plotting/index`
