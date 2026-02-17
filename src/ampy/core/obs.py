import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u

from ampy.core.structs import DataType
from ampy.core.structs import IntegratedFlux, SpectralFlux, SpectralIndex


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

    def __init__(self, path: str | Path, sort=True):
        df = pd.read_csv(path)
        if sort:
            self.df = self._sort(df)
        else:
            self.df = df

    @staticmethod
    def _sort(df):
        """ Sort the dataframe object. """
        quantities = [t * u.Unit(unit) for t, unit in zip(df["Time"], df["TimeUnits"])]
        times_in_seconds = [q.to(u.s).value for q in quantities]
        df["Time_sec"] = times_in_seconds
        return df.sort_values("Time_sec").reset_index(drop=True)

    def rows(self):
        """ Return the rows of the CSV file. """
        return self.df.itertuples(name='Observation')


class ObsArray:
    """
    Container for storing `Observation` data as arrays.

    Useful for MCMC fitting when speed is important. All arrays have the same
    size even if a particular array is only applicable to a subset of the
    observation. In these cases, any value can be stored in the other
    non-applicable elements (e.g., np.nan). This is done so that masks are more
    easily applied.

    Parameters
    ----------
    values : np.ndarray of float64
        The measurement values (e.g., spectral flux [mJy], integrated
        flux [erg cm-2 s-1] and spectral index).

    errors : np.ndarray of float64
        The errors associated with the measurement values. They are measured in
        the same units as its value.

    times : np.ndarray of float64
        The times associated with the measurements [days].

    types : np.ndarray of ampy.core.structs.DataType
        The types associated with the measurements.

    bands : np.ndarray of U10
        The band names associated with the measurements.

    frequencies : np.ndarray of float64
        The frequencies associated with the spectral flux values [Hz].

    int_lower, int_upper : np.ndarray of float64
        The lower and upper frequencies associated with the integrated values
        [Hz].

    wave_numbers : np.ndarray of float64
        The wave numbers corresponding to the frequencies [micro-meters].
    """
    if_units = 'erg cm-2 s-1'
    sf_units = 'mJy'
    time_units = 'd'
    freq_units = 'Hz'

    def __init__(
        self,
        values,
        errors,
        times,
        types,
        bands,
        frequencies,
        int_lower,
        int_upper,
        wave_numbers,
        extinguishable
    ):
        self.values = values
        self.errors = errors
        self.times = times
        self.types = types
        self.bands = bands
        self.frequencies = frequencies
        self.wave_numbers = wave_numbers
        self.extinguishable = extinguishable
        self.int_lower = int_lower
        self.int_upper = int_upper

        # Static truth arrays of data
        self.flux_loc = self.types != DataType.SPECTRAL_INDEX
        self.sflux_loc = self.types == DataType.SPECTRAL_FLUX
        self.iflux_loc = self.types == DataType.INTEGRATED_FLUX
        self.sindex_loc = self.types == DataType.SPECTRAL_INDEX

    @classmethod
    def from_data(cls, data):
        """
        Instantiates an `ObsArray` from an array of data values.

        Parameters
        ----------
        data : np.ndarray of ampy.core.structs.<DataType>
            DataTypes `{SpectralIndex, SpectralFlux, IntegratedFlux}`

        Returns
        -------
        .ObsArray
        """
        # Initializes info for all data types
        values = np.full(len(data), np.nan, dtype=np.float64)
        errors = np.full(len(data), np.nan, dtype=np.float64)
        times  = np.full(len(data), np.nan, dtype=np.float64)
        types  = np.full(len(data), np.nan, dtype=DataType)
        extinguishable = np.full(len(data), fill_value=False)

        # Initializes info for flux data types [str]
        bands = np.full(len(data), np.nan, dtype='U10')

        # Initializes info for spectral flux data type [Hz, um]
        frequencies = np.full(len(data), np.nan, dtype=np.float64)
        wave_numbers = np.empty(len(data), dtype=np.float64)

        # Initializes info for integrated data types [Hz]
        int_lower = np.full(len(data), np.nan, dtype=np.float64)
        int_upper = np.full(len(data), np.nan, dtype=np.float64)

        for i, f in enumerate(data):
            times[i] = f.time.to_value(cls.time_units)
            types[i] = f.type

            if f.type != DataType.SPECTRAL_INDEX:
                bands[i] = f.band

                # TODO: Allow user to specify
                if 9e13 <= f.frequency.to_value('Hz') <= 2.99e15:
                    extinguishable[i] = True

            # Extract spectral flux info
            if f.type == DataType.SPECTRAL_FLUX:
                values[i] = f.value.to_value(cls.sf_units)
                errors[i] = f.avg_uncertainty.to_value(cls.sf_units)
                frequencies[i] = f.frequency.to_value(cls.freq_units)
                wave_numbers[i] = 1 / f.wavelength.to_value('um')

            # Extract integrated flux info
            elif f.type == DataType.INTEGRATED_FLUX:
                values[i] = f.value.to_value(cls.if_units)
                errors[i] = f.avg_uncertainty.to_value(cls.if_units)
                int_lower[i] = f.int_range.lower.to_value(cls.freq_units)
                int_upper[i] = f.int_range.upper.to_value(cls.freq_units)

            # Extract spectral index info
            elif f.type == DataType.SPECTRAL_INDEX:
                values[i] = f.value.value
                errors[i] = f.avg_uncertainty.value
                int_lower[i] = f.int_range.lower.to_value(cls.freq_units)
                int_upper[i] = f.int_range.upper.to_value(cls.freq_units)

        return cls(
            values, errors, times, types, bands, frequencies,
            int_lower, int_upper, wave_numbers, extinguishable
        )


def filter_dict(d, mask):
    """"""
    if d is not None:
        return {
            key: np.array(vals)[mask]
            for key, vals in d.items()
        }


class Observation:
    """
    Time series of flux measurements.

    Parameters
    ----------
    data : np.ndarray
        The SpectralFlux, IntegratedFlux, SpectralIndex values.

    offsets : dict, optional
        <offset names>: <np.ndarray of where to apply offset>.

    hosts : dict, optional
        <host names>: <np.ndarray of where to apply host correction>.

    slops : dict, optional
        <slop names>: <np.ndarray of where the slop is applies>.

    include : np.ndarray, optional
        Should the data be included when modeling?
    """
    def __init__(self, data, offsets=None, hosts=None, slops=None, include=None, df=None):
        self._all_data = data
        self._all_offsets = offsets

        self.include = include
        self._included_data = data[include == 1]

        # Create convenient, fast arrays
        self._as_arrays = ObsArray.from_data(self._included_data)

        # Excluded stuff
        self._excluded_data = data[self.include == 0]
        self._excluded_offsets = filter_dict(offsets, include == 0)

        # Included Groups
        self.offsets = filter_dict(offsets, include == 1)
        self.slops = filter_dict(slops, include == 1)
        self.hosts = filter_dict(hosts, include == 1)

        # Optional: Store the CSV data frame
        self.df = df

        self.length = len(self._included_data)

    @classmethod
    def load(cls, path):
        """"""
        if isinstance(path, pathlib.Path):
            path = str(path)

        if path.endswith('.csv'):
            return Observation.from_csv(path)
        raise NotImplementedError("Only CSVs are supported!")

    def get_data(self, subset='all'):
        """ Returns the data. """
        if subset not in (valid := ('included', 'excluded', 'all')):
            raise ValueError(f'subset must be one of: {valid}')

        if subset == 'included':
            return self._included_data

        if subset == 'excluded':
            return self._excluded_data

        return self._all_data

    def get_offsets(self, subset='all'):
        """ Returns the offsets. """
        if subset not in (valid := ('included', 'excluded', 'all')):
            raise ValueError(f'subset must be one of: {valid}')

        if subset == 'included':
            return self.offsets

        if subset == 'excluded':
            return self._excluded_offsets

        return self._all_offsets

    @classmethod
    def from_csv(cls, path: str | Path):
        """
        Instantiates an `Observation` from a CSV.

        Parameters
        ----------
        path : str | Path
            The CSV file path.

        Returns
        -------
        Observation
            Instantiated from a CSV file path.
        """
        return cls.from_df(CSVReader(path).df)

    @classmethod
    def from_df(cls, df):
        """

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the observation.

        Returns
        -------
        .Observation
        """

        def init_dict(group: str) -> dict:
            """ Initialize group dictionary. """
            return {
                cg: np.zeros(len(df), dtype=bool)
                for cg in df[group].unique() if isinstance(cg, str)
            }

        # Handle optional columns
        slops, offsets, hosts = None, None, None

        # Default to modeling all data
        include = np.ones(len(df), dtype=int)

        if 'CalGroup' in df.columns.values:
            offsets = init_dict('CalGroup')

        if 'SlopGroup' in df.columns.values:
            slops = init_dict('SlopGroup')

        if 'HostGroup' in df.columns.values:
            hosts = init_dict('HostGroup')

        data = []
        for row in df.itertuples(name='Observation'):

            if hasattr(row, 'Include') and isinstance(row.Include, (int, float)):
                include[row.Index] = int(row.Include)

            # Parse data
            data_type = row.ValueType.lower()

            if data_type == DataType.INTEGRATED_FLUX.value:
                data.append(IntegratedFlux.from_csv_row(row))

            elif data_type == DataType.SPECTRAL_FLUX.value:
                data.append(SpectralFlux.from_csv_row(row))

            elif data_type == DataType.SPECTRAL_INDEX.value:
                data.append(SpectralIndex.from_csv_row(row))

            else:
                raise IOError(
                    f'Row {row.Index} has an invalid data type: '
                    f'{row.ValueType}.'
                )

            # Parse groups
            if offsets and isinstance(row.CalGroup, str):
                offsets[row.CalGroup][row.Index] = True

            if hosts and isinstance(row.HostGroup, str):
                hosts[row.HostGroup][row.Index] = True

            if slops and isinstance(row.SlopGroup, str):
                slops[row.SlopGroup][row.Index] = True

        return cls(
            np.asarray(data, dtype=object), offsets, hosts, slops, include, df
        )

    @property
    def data(self) -> np.ndarray:
        """
        Returns the list of data. No setter is defined so that locations can
        remain static. Otherwise, the locations would need to be determined on
        call which would slow down MCMC.

        Returns
        -------
        np.ndarray
        """
        return self._included_data

    @property
    def as_arrays(self) -> ObsArray:
        """
        Returns the data as arrays. No setter is defined so that locations can
        remain static. Otherwise, the locations would need to be determined on
        call which would slow down MCMC.

        Returns
        -------
        .ObsArray
            The array representation of the observation.
        """
        return self._as_arrays

    def times(self, quant: bool = False):
        """
        Returns the measurement times [days].

        Parameters
        ----------
        quant : bool, optional, default=False
            Should the values be astropy Quantities?

        Returns
        -------
        np.ndarray of float
        """
        return (
            np.asarray([d.time for d in self.data])
            if quant else self.as_arrays.times
        )

    def freqs(self, quant: bool = False):
        """
        Returns the measurement frequencies [Hz].

        Parameters
        ----------
        quant : bool, optional, default=False
            Should the values be astropy Quantities?

        Returns
        -------
        np.ndarray of float
        """
        return (
            np.asarray([d.frequency for d in self.data])
            if quant else self.as_arrays.frequencies
        )

    def int_lowers(self, quant: bool = False):
        """
        Returns the lower integration bounds [Hz].

        Parameters
        ----------
        quant : bool, optional, default=False
            Should the values be astropy Quantities?

        Returns
        -------
        np.ndarray of float
        """
        if quant:
            return np.asarray([d.int_range.lower for d in self.data])
        return self.as_arrays.int_lower

    def int_uppers(self, quant: bool = False):
        """
        Returns the upper integration bounds [Hz].

        Parameters
        ----------
        quant : bool, optional, default=False
            Should the values be astropy Quantities?

        Returns
        -------
        np.ndarray of float
        """
        if quant:
            return np.asarray([d.int_range.upper for d in self.data])
        return self.as_arrays.int_upper

    def epoch(self, mask=None):
        """

        Parameters
        ----------
        mask : np.ndarray of bool, optional

        Returns
        -------
        np.ndarray
        """
        times = self.as_arrays.times

        if mask is not None:
            times = times[mask]

        return np.array([times.min(), times.max()])

    def bands(self, mask=None):
        """
        Organizes the photometry locations by their bands.

        Parameters
        ----------
        mask : np.ndarray of bool, optional
            The subset of values to return.

        Returns
        -------
        dict[str, np.ndarray of int]
            The band photometry locations.
        """
        bands = (
            self.as_arrays.bands if mask is None
            else self.as_arrays.bands[mask]
        )

        groups = {}
        for i, val in enumerate(bands):
            groups.setdefault(val, []).append(i)

        # Convert lists to numpy arrays
        return {k: np.asarray(v, dtype=int) for k, v in groups.items()}

    def mock(self, subset, ndata=100, exclude=None):
        """
        Creates a mock observation.

        The mock observation will take each unique data point (defined
        the `subset` columns, and then generate `ndata` points for each of the
        data points.

        Parameters
        ----------
        subset : tuple of str
            The column names to determine uniqueness.

        ndata : int, optional, default=100
            The number of data points to generate for each band.

        exclude : iterable, optional, default=`None`
            Which data types should be excluded? Valid values include
            `integrated flux`, `spectral flux`, and `spectral index`.

        Returns
        -------
        ampy.core.obs.Observation
        """
        exclude = exclude or ()

        # Get the starting position of each band
        unique = self.df.drop_duplicates(subset=subset, keep="first").copy()

        # Generate evenly spaced times from the earliest
        # to the latest observation time [days].
        obs_times = self.epoch()

        mock_times = np.geomspace(
            obs_times[0], obs_times[1], ndata, dtype=float
        )

        # Generate the new data frame object
        chunks = []

        for _, row in unique.iterrows():
            if row.ValueType.lower() in exclude:
                continue

            # Repeat the row ndata times
            block = pd.DataFrame([row.to_dict()] * ndata)

            # Replace time column with dense grid
            block["Time"] = mock_times
            block["TimeUnits"] = "d"
            block["Include"] = 1

            chunks.append(block)

        try:
            df = pd.concat(chunks, ignore_index=True)
        except ValueError as e:
            raise ValueError(
                f"There are no objects to concatenate. This is likely because "
                f"all of the data was excluded."
            ) from e

        # Sort by the time column
        df_sorted = CSVReader._sort(df)

        return self.__class__.from_df(df_sorted)

    @property
    def flux_loc(self) -> np.array:
        """ Returns bools indicating the flux locations. """
        return self.as_arrays.flux_loc

    @property
    def sflux_loc(self) -> np.array:
        """ Returns bools indicating the spectral flux locations. """
        return self.as_arrays.sflux_loc

    @property
    def iflux_loc(self) -> np.array:
        """ Returns bools indicating the integrated flux locations. """
        return self.as_arrays.iflux_loc

    @property
    def sindex_loc(self) -> np.array:
        """ Returns bools indicating the spectral index locations. """
        return self.as_arrays.sindex_loc

    @property
    def extinguishable(self) -> np.array:
        """ Returns bools indicating the flux affected by dust extinction locations. """
        return self.as_arrays.extinguishable
