Run configuration
=================

The *run configuration* is the top-level entry point used to execute AMPy.
It is intentionally small: it specifies where to find the observational
data, where to find the modeling and inference configurations, and where to
write outputs.

Schema
------

The run configuration contains the following sections:

``[input]``
  Specifies the observational input file. AMPy currently supports CSV
  observational inputs.


``[modeling]``
  Path to the modeling configuration (plugins, models, parameters).

``[inference]``
  Path to the inference configuration (sampler settings, run lengths, etc.).

``[output]``
  Output location for generated data products (plots and reports).

Example
-------

.. code-block:: toml

   [input]
   path = "C:/path/to/input.csv"

   [modeling]
   path = "C:/path/to/modeling.toml"

   [inference]
   path = "C:/path/to/inference.toml"

   [output]
   directory = "C:/directory/to/save/results"

Notes
-----

* Paths may be absolute or relative. Relative paths are resolved relative to
  the location of the run configuration file.
* Forward slashes are recommended for paths on Windows to avoid escaping.
