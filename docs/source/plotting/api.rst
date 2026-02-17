.. _plotting_api:

API reference
=============

The plotting defaults are set up such that the user is not expected to modify
them in order to get publication-ready plots. However, because there is no
"right way" to make a figure, it is impossible to create a set of defaults
that satisfy everyone. These defaults are made accessible so that users can
easily modify the figures in a single location, if desired.

These defaults are provided as immutable dataclasses and are intended to:

* Ensure consistent visual styling across plots
* Provide sensible, publication-ready defaults
* Allow users to override behavior in a controlled and explicit way

.. automodule:: ampy.defaults.plotting
   :members:
   :undoc-members: False
   :show-inheritance:
