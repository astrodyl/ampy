Plugin configuration
====================

AMPy uses a plugin system to apply additional modeling components during
inference, such as dust extinction, host-galaxy contributions, calibration
offsets, or custom user-defined effects.

Plugins are configured using standalone TOML files. Each plugin instance
declares which plugin class to use, which model it wraps, and the parameters
passed to that model.

This page describes the **generic plugin configuration format**, independent
of any specific plugin implementation.

Overview
--------

A plugin configuration defines:

* Which plugin class is used
* A user-defined name for that plugin instance
* Which model the plugin wraps
* A set of parameters passed to the model

The same plugin class may be used multiple times in a single run, provided
each instance has a unique name.

Required fields
---------------

Each plugin configuration file must define the following top-level fields:

``plugin``
  The name of the plugin class to use. The class must be defined in
  ``ampy.modeling.plugins.py``.

``name``
  A user-defined identifier for this plugin instance. This name must be
  unique within a run and is used internally by AMPy to associate parameters
  with the correct plugin.

``model``
  A fully-qualified import path pointing to the model used by the plugin.
  This may refer to a function, class, or pre-instantiated object.

Example:

.. code-block:: toml

   plugin = "DustPlugin"
   name   = "source_frame_dust"
   model  = "ampy.modeling.models.builtin.source_dust_extinction_model"

Parameter blocks
----------------

Plugin parameters are specified using ``[[eval]]`` and, optionally,
``[[init]]`` blocks.

Each block defines a single parameter passed to the model.

``[[eval]]``
  Parameters passed each time the model is evaluated. These may be fixed
  values or inferred via priors.

``[[init]]``
  Parameters passed only during model initialization. These are only used when
  the model is a class.

If the model is a **function**, only ``[[eval]]`` blocks should be defined.
If the model is a **class**, both ``[[init]]`` and ``[[eval]]`` may be used.

The user does not need to specify whether the model is a function or class;
AMPy determines this automatically. Additionally, if the model is a class and
all of the ``[[init]]`` parameters are fixed, then the AMPy will instantiate
the class once before running the inference and call the object with the
sampled ``[[eval]]`` parameters.

Parameter definitions
---------------------

Each ``[[init]]`` or ``[[eval]]`` block defines exactly one parameter.

Supported fields include:

``name``
  The parameter name. This must match the corresponding argument name
  expected by the model.

``value``
  A fixed value for the parameter.

``prior``
  A prior specification for parameters inferred during Bayesian sampling.
  The prior format is shared across all configuration files.

Only one of ``value`` or ``prior`` should be specified for a given parameter.

Example plugin configuration
----------------------------

The following example shows a complete plugin configuration using a dust
plugin applied in the source frame:

.. code-block:: toml

   plugin = "DustPlugin"
   name   = "source_frame_dust"
   model  = "ampy.modeling.models.builtin.source_dust_extinction_model"

   [[eval]]  # Redshift
       name  = "z"
       value = 2.198

   [[eval]]  # Total-to-selective extinction
       name  = "Rv"
       value = 3.1

   [[eval]]  # Color excess E(B - V)
       name  = "Ebv"
       prior = { type="uniform", lower=0.0, upper=0.5, start={ guess=0.1, sigma=0.1 } }

In this example:

* The plugin class ``DustPlugin`` is used
* The plugin instance is named ``source_frame_dust``
* The model is specified by its import path
* Two parameters are fixed
* One parameter is inferred during sampling

Multiple instances of the same plugin
-------------------------------------

The same plugin class may be used multiple times in a single run. For example,
a dust plugin may be applied both in the source frame and in the Milky Way.

In such cases, each plugin instance must have a distinct ``name`` field, even
if the ``plugin`` field is the same. AMPy uses the ``name`` field internally
to map parameters to the correct plugin instance.

User-defined plugins
--------------------

Users may provide custom plugin implementations, provided they conform to the
AMPy plugin interface defined in ``ampy.modeling.plugins.PluginBase``.

Custom plugins are configured using the same TOML format described here.
Only the ``plugin`` and ``model`` fields change to reference the user-defined
implementation.