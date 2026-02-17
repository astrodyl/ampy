Modeling configuration
======================

The *modeling configuration* defines which modeling components are applied
during an AMPy run. It is composed of an ordered list of plugins, each of
which points to a separate plugin configuration file.

This file acts as a **registry**: it specifies *what* components are included
in the model and *in what order* they are applied, without defining the
details of each component.

Overview
--------

The modeling configuration consists of a list of ``[[plugins]]`` entries.
Each entry corresponds to a single plugin instance and references an
external plugin configuration file.

Plugins may be enabled or disabled without being removed from the file,
allowing users to easily toggle modeling components during experimentation.

Schema
------

Each ``[[plugins]]`` entry supports the following fields:

``enabled``
  Boolean flag indicating whether the plugin is active. If set to ``false``,
  the plugin is ignored during the run.

``include``
  Path to a plugin configuration file. This file defines the plugin class,
  model, and associated parameters.

Plugins are processed **in the order they appear** in the file.

Example
-------

.. code-block:: toml

   # Base afterglow model
   [[plugins]]
   enabled = true
   include = "C:/Projects/configs/registry/plugins/afterglow.toml"

   # Source-frame dust extinction
   [[plugins]]
   enabled = true
   include = "C:/Projects/configs/registry/plugins/dust_sf.toml"

   # Host galaxy contribution
   [[plugins]]
   enabled = false
   include = "C:/Projects/configs/registry/plugins/host_galaxy.toml"

   # Milky Way dust extinction
   [[plugins]]
   enabled = true
   include = "C:/Projects/configs/registry/plugins/dust_mw.toml"

   # Calibration offsets
   [[plugins]]
   enabled = true
   include = "C:/Projects/configs/registry/plugins/calibration.toml"

   # Chi-squared "slop" term
   [[plugins]]
   enabled = true
   include = "C:/Projects/configs/registry/plugins/chisquared.toml"

Notes
-----

* Plugin configuration files may be reused across runs and referenced by
  multiple modeling configurations.
* Paths may be absolute or relative. Relative paths are resolved relative to
  the location of the modeling configuration file.
* Disabling a plugin via ``enabled = false`` preserves the configuration for
  later use without requiring edits to the file structure.

See also
--------

* :doc:`plugins` — detailed description of plugin configuration files
* :doc:`inference` — inference configuration