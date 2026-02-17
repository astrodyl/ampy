import tomllib

import numpy as np

from ampy.core.obs import Observation


class ModelingEngine:
    """
    Container for MCMC models used during fitting.

    Parameters
    ----------
    observation : core.obs.Observation
        The observational data.
    """
    def __init__(self, observation):
        self.observation = observation

    def __call__(self, plugins, params):
        """ Behaves the same as calling `model(plugins, params)`. """
        return self.model(plugins, params)

    @classmethod
    def load(cls, path):
        """
        Load the modeling engine from a config file.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the config file.

        Returns
        -------
        ampy.modeling.engine.ModelingEngine
        """
        if path.endswith('.toml'):
            return cls.from_toml(path)

        raise NotImplementedError("Only TOMLs are supported!")

    @classmethod
    def from_toml(cls, path):
        """
        Load the modeling engine from a TOML file.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the TOML file.

        Returns
        -------
        ampy.modeling.engine.ModelingEngine
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(Observation.load(data['observation']['path']))

    def model(self, plugins, params):
        """
        Models the afterglow flux, using all provided plugins.

        Parameters
        ----------
        plugins : iterable

        params : dict

        Returns
        -------
        np.ndarray of float
            The modeled afterglow values.

        Raises
        ------
        ValueError
            If the modeled values contain a NaN or +/- inf.
        """
        # Model the base afterglow flux
        modeled = plugins[0](
            self.observation, params.get(f"{plugins[0].name}")
        )

        if np.isnan(modeled.min()):
            raise ValueError(f"The base flux model modeled a NaN.")

        if len(plugins) == 1:
            return modeled

        # Modify the flux in-place for each plugin
        for plugin in plugins[1:]:
            plugin(modeled, self.observation, params.get(f"{plugin.name}"))

            if np.isnan(modeled.min()):
                raise ValueError(f"The '{plugin.name}' plugin modeled a NaN.")

        return modeled
