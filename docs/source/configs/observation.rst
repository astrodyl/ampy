Observational input (CSV)
=========================

AMPy ingests observational data from a single CSV file. This file is the
authoritative source for the dataset used during modeling and inference.

The input format is designed to be flexible while remaining explicit and
machine-readable: each measurement is represented by a row, and the meaning
of each row is determined by its declared data type.

Row types
---------

AMPy is capable of modeling three types of measurements, each of which are
specified via the ``ValueType`` column.

``SpectralFlux``
  A flux density measurement at a single observing frequency.

``IntegratedFlux``
  A measurement integrated over a frequency/energy band. This row type
  typically specifies a lower and upper bound for the integration range
  along with the integrated flux value.

``SpectralIndex``
  An observed spectral index (e.g., from an X-ray spectral fit).

Column reference
----------------

This section documents the core columns used to define observational
measurements in the input CSV file. All values are parsed and internally
converted to a consistent unit system when modeling using :mod:`astropy.units`.

Units and internal consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All ``Units`` columns are parsed using Astropy’s unit system. As a result:

* Unit strings **must** correspond to valid Astropy units
* Values are converted internally to consistent units before modeling
* Users are free to mix units across rows and data types

This design prioritizes flexibility while maintaining physical correctness and
numerical consistency.

Time-related columns
~~~~~~~~~~~~~~~~~~~~

``Time``
  The time at which the measurement was made.

``TimeUnits``
  Units associated with the ``Time`` column.

``TimeLower`` and ``TimeUpper``
  Optional lower and upper bounds on the observation time.

  These columns are **used for plotting purposes only** (e.g., error bars or
  time intervals) and are **not used during inference**. AMPy always evaluates
  models at the value specified in ``Time``.

Value-related columns
~~~~~~~~~~~~~~~~~~~~~

``Value``
  The measured quantity.

  Depending on ``ValueType``, this may represent a spectral flux density, an
  integrated flux, or a spectral index. This is the primary quantity modeled
  by AMPy.

``ValueUnits``
  Units associated with the ``Value`` column.

  The unit string must be compatible with the corresponding ``ValueType`` and
  must be a valid Astropy unit (e.g., ``"mJy"``, ``"Jy"``, ``"erg cm-2 s-1"``,
  etc.). In the case of Spectral Indices, this column may be omitted.

``ValueLower`` and ``ValueUpper``
  Uncertainty bounds on the measured value.

  These represent the lower and upper uncertainties on ``Value`` such that the
  measurement is interpreted as: ``Value +/- ValueUpper/ValueLower``. AMPy
  only models a symmetric uncertainty, so the average of the uncertainties
  is used.

``ValueType``
  One of ``SpectralFlux``, ``IntegratedFlux``, or ``SpectralIndex``.

Wave-related columns
~~~~~~~~~~~~~~~~~~~~

AMPy uses the term *wave* to generically refer to either a wavelength or a
frequency. Thanks to the flexible unit system, users may specify values in
units of length (wavelength) or inverse time (frequency), and AMPy will
automatically convert them internally using :mod:`astropy.units`.

``Wave``
  Specifies the effective observing wavelength or frequency at which the
  spectral flux is measured. This column is ignored for ``IntegratedFlux``
  and ``SpectralIndex`` data.

``WaveLower`` and ``WaveUpper``
  Specifies the range over which the ``IntegratedFlux`` or ``SpectralIndex``
  quantity is measured.

``WaveUnits``
  Units associated with ``Wave``, ``WaveLower``, and ``WaveUpper``.

  The unit string must correspond to a valid Astropy unit and may represent either:

  * a frequency (e.g., ``"Hz"``, ``"s-1"``, etc.)
  * a wavelength (e.g., ``"cm"``, ``"Angstrom"``, ``"micron"``, etc.)

Grouping columns
~~~~~~~~~~~~~~~~

AMPy supports several optional grouping and control columns that allow users
to associate subsets of the data with specific modeling or plotting behavior.
These columns are designed to support flexible, reproducible workflows.

``Band``
~~~~~~~~
  Band or filter identifier associated with the observation.

  This column is:

  * **Used for plotting only**
  * **Not used during modeling or inference**

  AMPy uses the ``Band`` column to group and color-code data when generating
  light curves and other visualizations. The supported values, and their
  associated colors, are described in :ref:`plotting_api`.

``CalGroup``
~~~~~~~~~~~~
  Calibration grouping label.

  This column defines subsets of the data that share a common calibration offset.
  Rows with the same ``CalGroup`` value are assigned the same calibration
  parameter during modeling. Typical use cases include filter-dependent
  calibration uncertainties.

  Behavior:

  * Rows with the same non-null ``CalGroup`` value share a calibration offset
  * Rows with a blank or missing ``CalGroup`` are **not** modeled with a
    calibration offset
  * The specific calibration model is defined by the corresponding plugin

  The value may be any string; only equality between rows matters.

``HostGroup``
~~~~~~~~~~~~~
  HostGalaxy grouping label.

  This column defines subsets of the data that share a common host galaxy
  contribution. Rows with the same ``HostGroup`` value are assigned the same
  host galaxy parameter during modeling.

``SlopGroup``
~~~~~~~~~~~~~
  Slop (extra variance) grouping label.

  Slop is used to model unaccounted-for variance in the likelihood (e.g., an
  additional term in the chi-squared calculation). This column allows different
  subsets of the data to have independent slop parameters.

  Behavior:

  * Rows with the same ``SlopGroup`` value share the same slop parameter
  * If the ``SlopGroup`` column is present, slop is modeled per group
  * If the ``SlopGroup`` column is **absent**, AMPy models a **single global slop**
    across the entire dataset

  As with ``CalGroup``, the specific slop model is defined by the corresponding
  plugin.

``Include``
~~~~~~~~~~~
  Model inclusion flag.

  This column controls whether a given data point is included in the inference.
  This column is intended to support exploratory workflows by allowing users to
  temporarily exclude data points without deleting or modifying the underlying
  dataset.

  Accepted values:

  * ``1`` — include the data point in modeling
  * ``0`` — exclude the data point from modeling

  Behavior:

  * If the ``Include`` column is **absent**, all data points are included
  * Excluded data points (``Include = 0``) are plotted in a muted or greyed-out
    style to indicate that they were not used during inference

See also
--------

* :doc:`run`
* :doc:`modeling`
* :doc:`inference`
* :doc:`plugins`
