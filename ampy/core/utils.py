import math
import os
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass

from ampy.core.structs import ScaleType


# TODO: validation for readers


def save_plot_unique(filename_base, ext, directory):
    """
    Save a matplotlib plot to disk, adding a suffix if the file exists.

    Parameters
    ----------
    filename_base: str
        base name without extension

    ext: str
        file extension (default 'png')

    directory: str
        directory to save in (default current directory)
    """
    i = 0
    while True:
        filename = f"{filename_base}.{ext}" if i == 0 else f"{filename_base}_{i}.{ext}"
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            plt.savefig(filepath, dpi=1200)
            return
        i += 1


class CSVReader:
    """
    Reads an input CSV.

    Parameters
    ----------
    path : str | Path
        Location to the csv file.

    Attributes
    ----------
    df : pd.DataFrame
        Pandas representation of the CSV file.
    """
    _req_headers = (
        'Time', 'TimeUnits', 'Value', 'ValueType',
        'ValueLower', 'ValueUpper', 'ValueUnits'
    )

    def __init__(self, path: str | Path):
        df = pd.read_csv(path)
        df_sorted = df.sort_values(by='Time')
        df_sorted = df_sorted.reset_index(drop=True)
        self.df = df_sorted

    @staticmethod
    def sort(df):
        """ Sort the dataframe object. """
        return df.sort_values(by='Time').reset_index(drop=True)

    def rows(self):
        """ Return the rows of the CSV file. """
        return self.df.itertuples(name='Observation')


class TOMLReader:
    """
    Reads an input TOML.

    Parameters
    ----------
    path : str | Path
        The path to the TOML file.

    Attributes
    ----------
    data : dict
        The file data.
    """
    def __init__(self, path: str | Path):
        self.path = path
        self.data = self.read()

    def read(self) -> dict:
        """
        Opens the TOML file at ``path``.

        Returns
        -------
        dict
            A dictionary of TOML data.
        """
        with open(self.path, "rb") as f:
            return tomllib.load(f)

    @staticmethod
    def validate_value(name: str, value, expected_type) -> None:
        """

        Raises
        ------
        TypeError
            If encounters unexpected value.
        """
        if not isinstance(value, expected_type):
            raise TypeError(f'Received unexpected value for {name}.')

    def get_section(self, section: str, optional: bool = False) -> dict | None:
        """
        Checks that the section exists and returns it if it does.

        Parameters
        ----------
        section : str
            The TOML section name.

        optional : bool, optional, default=False
            If ``True``, does not raise an exception if section does not
            exist. Instead, returns ``None``.

        Returns
        -------
        dict
            The section dictionary.

        Raises
        ------
        ValueError
            If the section does not exist and optional is ``False``.
        """
        if (data := self.data.get(section, None)) is None and not optional:
            raise ValueError(
                f'{self.path} does not contain a {section} section.'
            )

        return data


class MCMCSettingsReader(TOMLReader):
    """
    Reader for MCMC Settings TOML config file.

    Attributes
    ----------
    num_walkers : float
        The number of walkers.

    burn_length : float
        The number of iterations to burn.

    run_length : float
        The number of iterations to run.
    """
    def __init__(self, path: str | Path, live_dangerously: bool = False):
        """
        Parameters
        ----------
        live_dangerously : bool, optional
            If ``True``, skips the validation process.
        """
        super().__init__(path)

        if not live_dangerously:
            self.validate()

        sampler = self.data.get('sampler')
        self.name = sampler.get('Name')
        self.num_walkers = sampler.get('num_walkers')
        self.burn_length = sampler.get('burn_length')
        self.run_length = sampler.get('run_length')
        self.ntemps = sampler.get('ntemps')
        self.workers = sampler.get('workers')

    def validate(self) -> None:
        """ Validates that the MCMC settings file is valid. """
        self.validate_sampler()
        self.validate_model()

    def validate_sampler(self) -> None:
        """ Validates that the sampler section is valid. """
        sampler = self.get_section('sampler')

        self.validate_value('burn_length', sampler.get('burn_length'), int)
        self.validate_value('run_length',  sampler.get('run_length'),  int)
        self.validate_value('num_walkers', sampler.get('num_walkers'), int)

    def validate_model(self) -> None:
        """ Validates that the model section is valid. """
        self.validate_value('model', self.get_section('model'), str)


# <editor-fold desc="Math">
def chi_squared(
    f: np.ndarray[float],
    y: np.ndarray[float],
    e: np.ndarray[float],
    s = None,
) -> float:
    """
    Calculates the chi-squared value.

    Parameters
    ----------
    f : np.ndarray of float
        The predicted values.

    y : np.ndarray of float
        The observed values.

    e : np.ndarray of float
        The uncertainty in the observed values.

    s : float or np.ndarray of float, optional
        The slop parameter.

    Returns
    -------
    float or Quantity
        The chi-squared value.
    """
    if s is None:
        return np.sum(((y - f) / e) ** 2)

    return chi_squared_eff(f, y, e, s)


def chi_squared_eff(
    f: np.ndarray[float],
    y: np.ndarray[float],
    e: np.ndarray[float],
    s,
) -> float:
    """
    When the slop parameter, `s`, is provided, the
    chi-squared calculation accounts for additional
    unknown variances. When `s > 0`, the effective
    uncertainties increase, decreasing the penalty
    for model-data mismatches but adding a penalty
    for increasing `s` through the normalization term.

    Parameters
    ----------
    f : np.ndarray of float
        The modeled values.

    y : np.ndarray of float
        The observed values.

    e : np.ndarray of float
        The uncertainty in the observed values.

    s : float or np.ndarray of float
        The log slop parameter.

    Returns
    -------
    float
        The effective chi-squared value.
    """
    # Convert slop to linear space
    s_lin_avg = f * (10**s - 10**-s) / 2

    # Combine the slop and data uncertainties
    sig = np.sqrt(s_lin_avg ** 2 + e ** 2)

    # return chi-squared effective
    return np.sum(2 * np.log(sig) + ((y - f) / sig) ** 2)


def to_scale(value: float, from_s: str | ScaleType,
             to_s: str | ScaleType) -> float:
    """
    Converts ``value`` from ``from_s`` to ``to_s``.

    Parameters
    ----------
    value : float
        The value to convert.

    from_s : str or ScaleType
        The scale of ``value``.

    to_s : str or ScaleType
        The new scale type.

    Returns
    -------
    float
        The converted value.
    """
    if isinstance(from_s, str):
        from_s = ScaleType(from_s)

    if isinstance(to_s, str):
        to_s = ScaleType(to_s)

    if from_s == to_s:
        return value

    match to_s:
        case ScaleType.LOG:
            return to_log(value, from_s)
        case ScaleType.LN:
            return to_ln(value, from_s)
        case ScaleType.LINEAR:
            return to_linear(value, from_s)


def to_log(value: float, scale: ScaleType) -> float:
    """
    Converts ``value`` from ``scale`` to log10.

    Parameters
    ----------
    value : float
        The value to convert.

    scale : ScaleType
        The scale of ``value``.

    Returns
    -------
    float
        The converted value.
    """
    match scale:
        case ScaleType.LOG:
            return value
        case ScaleType.LN:
            return value / math.log(10)
        case ScaleType.LINEAR:
            return math.log10(value)


def to_ln(value: float, scale: ScaleType) -> float:
    """
    Converts ``value`` from ``scale`` to log.

    Parameters
    ----------
    value : float
        The value to convert.

    scale : ScaleType
        The scale of ``value``.

    Returns
    -------
    float
        The converted value.
    """
    match scale:
        case ScaleType.LN:
            return value
        case ScaleType.LOG:
            return value * math.log(10)
        case ScaleType.LINEAR:
            return math.log(value)


def to_linear(value: float, scale: ScaleType) -> float:
    """
    Converts ``value`` from ``scale`` to linear.

    Parameters
    ----------
    value : float
        The value to convert.

    scale : ScaleType
        The scale of ``value``.

    Returns
    -------
    float
        The converted value.
    """
    match scale:
        case ScaleType.LINEAR:
            return value
        case ScaleType.LOG:
            return 10 ** value
        case ScaleType.LN:
            return math.exp(value)
# </editor-fold>


# <editor-fold desc="Project Navigation">
def get_project_parent_path() -> Path:
    """
    Returns the project's parent directory.

    Returns
    -------
    Path
        The project's parent directory.
    """
    return get_project_root().parent


def get_project_root() -> Path:
    """
    Returns the project root directory.

    Returns
    -------
    Path
        The project root directory.
    """
    return Path(__file__).parent.parent.parent


def get_models_path() -> Path:
    """
    Returns the project models directory.

    Returns
    -------
    Path
        The project models directory.
    """
    return get_project_root() / 'ampy' / 'models'


def get_results_path() -> Path:
    """
    Returns the project results directory.

    Returns
    -------
    Path
        The project results directory.
    """
    return get_project_root() / 'ampy' / 'results'


def get_resource_path() -> Path:
    """
    Returns the project resource directory.

    Returns
    -------
    Path
        The project resource directory.
    """
    return get_project_root() / 'ampy' / 'resources'


def get_event_path(category: str, event: str) -> Path:
    """
    Returns the path to the event's resource directory.

    Parameters
    ----------
    category : str

    event : str
        Name of event directory in resources

    Returns
    -------
    Path
        The event resource path.
    """
    return get_resource_path() / category / event


def get_input_csv_path(category: str, event: str) -> Path:
    """
    Returns the path to the input csv file.

    Parameters
    ----------
    category : str

    event : str
        Name of event directory in resources

    Notes
    -----
    The input CSV file should be named the same as the event directory.

    Returns
    -------
    Path
        The input CSV file path.
    """
    return get_event_path(category, event) / f'{event}.csv'


def get_mcmc_settings_path() -> Path:
    """ Returns the path to the MCMC settings. """
    return get_project_root() / 'ampy' / 'mcmc' / 'settings.toml'


def get_boosted_fireball_path():
    """ Returns the path to the boosted fireball root directory. """
    return get_models_path() / 'boosted'


def get_hydro_sim_table_path() -> Path:
    """
    Returns the path to the hydrodynamic simulation table.

    Returns
    -------
    Path
        The input hydrodynamic simulation table.
    """
    return get_project_root() / 'rsrcs' / 'hydro_sim_new.h5'
#</editor-fold>