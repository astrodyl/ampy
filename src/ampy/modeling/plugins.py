import importlib
import inspect
import os
import pathlib
import tomllib

import numpy as np

from ampy.defaults.modeling import CalibrationDefaults
from ampy.inference.params import factory
from ampy.modeling.models.builtin import import_from_string
from ampy.modeling.models.builtin import ImportFromStringError


def plugin_factory(d, obs=None):
    """
    Loads the appropriate plugin.

    Parameters
    ----------
    d : dict or str
        The dict representing the plugin.

    obs : ``ampy.core.obs.Observation``, optional
        If using defaults, the observation is required.

    Returns
    -------
    The loaded plugin.
    """
    try:
        plugin = getattr(
            importlib.import_module(f"{__package__}.plugins"), f'{d["plugin"]}'
        )

    except ImportError as e:
        # User requested a plugin that does not
        # exist in this file.
        raise ImportFromStringError(
            f"Could not import plugin '{f"{d["plugin"]}"}'."
        ) from e

    except TypeError as e:
        # Received a path to the file rather than the
        # loaded file itself. Load it here then recall.
        with open(d, "rb") as f:
            plugin_cfg = tomllib.load(f)
        return plugin_factory(plugin_cfg)

    # We found the requested plugin, load and return it
    if d.get('defaults'):
        return plugin.from_defaults(obs=obs)

    return plugin.from_dict(d)


def load_plugins(l, base_dir=None, obs=None):
    """

    Parameters
    ----------
    l : array_like
        List of plugins to load.

    base_dir : str or ``pathlib.Path``, optional
        The base directory of the relative paths.

    obs : ``ampy.core.obs.Observation``, optional
        If using defaults, the observation is required.

    Returns
    -------
    list
    """
    plugins = []

    for plugin in l:

        if plugin.get("enabled", True):
            path = pathlib.Path(plugin["include"])

            if not path.is_absolute():
                path = os.path.join(base_dir, path)

            with open(path, "rb") as f:
                plugin_cfg = tomllib.load(f)

            # Load the actual plugin
            plugins.append(plugin_factory(plugin_cfg, obs))

    return plugins


class PluginBase:
    """
    Builtin plugin adapter for AMPy.

    Parameters
    ----------
    name : str
        The user-defined name of the plugin for easy access.

    params : np.ndarray of `ampy.inference.params.MCMCParameter`
        The parameters used for this plugin's model.

    model : Callable, conditional
        The model to use. The `model` can be either a class, object, or
        function. Required if `pre_computed` is not provided.

    pre_computed : np.ndarray, conditional
        The pre-computed values. Instead of calculating the values each MCMC
        call, use pre-computed values. Required if `model` is not provided.

    args : dict, optional
        Any additional args to pass into the model.

    Raises
    ------
    ValueError
        If both `model` and `pre_computed` are None. At least one is required.
        If both are provided, then `model` is ignored when the plugin is
        called.
    """
    def __init__(self, name, params, model=None, pre_computed=None, args=None):
        if model is pre_computed is None:
            raise ValueError(
                "Either 'model' or 'pre_computed' must be provided"
            )

        self.name = name
        self.params = params
        self.model = model
        self.pre_computed = pre_computed
        self.args = args or {}

    def __repr__(self):
        """ Human-readable representation """
        return f"{self.__class__.__name__}(model={self.model!r})"

    @classmethod
    def load(cls, path):
        """
        Load the plugin.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the config file.

        Returns
        -------
        Child of `PluginBase`
        """
        if path.endswith('.toml'):
            return cls.from_toml(path)

        raise NotImplementedError("Only TOMLs are supported!")

    @classmethod
    def from_toml(cls, path):
        """
        Load the plugin from a TOML file.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the TOML file.

        Returns
        -------
        Child of `PluginBase`
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d):
        """
        Load the plugin from a dictionary.

        Parameters
        ----------
        d : dict
            The dictionary containing the plugin info.

        Returns
        -------
        Child of `PluginBase`
        """
        # Load in all the plugin params
        params = []

        for param in d.get("init", []):
            params.append(
                factory(param | {"stage": "init", "plugin": d["name"]})
            )

        for param in d.get("eval", []):
            params.append(
                factory(param | {"stage": "eval", "plugin": d["name"]})
            )

        # Import the desired model
        model = import_from_string(d["model"])

        # Return the plugin
        return cls(d["name"], np.asarray(params), model=model)

    def optimize(self, obs):
        """
        Performs one-time optimizations for the model.

        Parameters
        ----------
        obs : ampy.core.obs.Observation
            The observational data.
        """
        # Store the parameter values by stage
        params = {'init': {}, 'eval': {}}

        # Are all the stage params fixed?
        fixed = {'init': True, 'eval': True}

        for param in self.params:
            # Determine if all stage parameters are fixed
            if fixed[param.stage] and not param.fixed:
                fixed[param.stage] = False

            # Separate the values into stages
            if param.fixed:
                params[param.stage][param.name] = param.value

        # Determine if there are any optimizations and apply them
        if inspect.isclass(self.model):

            if fixed['init'] and fixed['eval']:
                # All the model parameters are fixed, so we calculate
                # them once here and store their values.
                self.pre_computed = self.calculate(obs, params)

            elif fixed['init']:
                # Only the __init__ parameters are fixed. Instantiate the
                # class here, but evaluate it for each MCMC iteration.
                self.model = self.model(**params['init'])

        elif inspect.isfunction(self.model) and fixed['eval']:
            # All function parameters are fixed, evaluate it here once.
            self.pre_computed = self.calculate(obs, params)

    def calculate(self, obs, params):
        """
        Calculates the

        Parameters
        ----------
        obs : core.obs.Observation
            The observation object.

        params : dict
            The parameters containing the 'init' and 'eval' keys.

        Returns
        -------
        np.ndarray of float
            The
        """
        init = params.get('init', {})
        call = params.get('eval', {}) | self.args

        if inspect.isclass(self.model):
            return self.model(**init)(obs, call)

        # Either a func or callable object
        return self.model(obs, call)


class AfterglowFluxPlugin(PluginBase):
    """"""
    module = 'modeling'

    def __init__(self, name, params, model=None, pre_computed=None, args=None):
        super().__init__(name, params, model, pre_computed, args)

    def __call__(self, obs, params):
        """
        Applies an in-place dust extinction correction to the flux values.

        Parameters
        ----------
        obs : core.obs.Observation
            The observation object containing the wave numbers and
            extinguishable positions.

        params : dict
            The extinction parameters containing the 'init' and 'eval' keys.
        """
        if self.pre_computed is not None:
            return self.pre_computed

        return self.calculate(obs, params)


class DustPlugin(PluginBase):
    """ Builtin dust model plugin adapter for AMPy. """
    module = 'modeling'

    def __init__(self, name, params, model=None, pre_computed=None, args=None):
        super().__init__(name, params, model, pre_computed, args)

    def __call__(self, flux, obs, params):
        """
        Applies an in-place dust extinction correction to the flux values.

        Parameters
        ----------
        flux : np.ndarray of float
            The afterglow flux.

        obs : core.obs.Observation
            The observation object containing the wave numbers and
            extinguishable positions.

        params : dict
            The extinction parameters containing the 'init' and 'eval' keys.
        """
        if self.pre_computed is not None:
            flux *= self.pre_computed

        else:
            flux *= self.calculate(obs, params)


class HostGalaxyPlugin(PluginBase):
    """ Builtin host galaxy plugin adapter for AMPy. """
    module = 'modeling'

    def __init__(self, name, params, model=None, pre_computed=None, args=None):
        super().__init__(name, params, model, pre_computed, args)

    def __call__(self, flux, obs, params):
        """
        Applies the host galaxy contributions to the flux values.

        Parameters
        ----------
        flux : np.ndarray of float
            The afterglow flux.

        obs : core.obs.Observation
            The observation object containing the host galaxy groups.

        params : dict
            The host galaxy parameters containing the 'init' and 'eval' keys.
        """
        if self.pre_computed is not None:
            flux += self.pre_computed

        else:
            flux += self.calculate(obs, params)


class CalibrationPlugin(PluginBase):
    """ Calibration Offset plugin adapter for AMPy. """
    module = 'modeling'

    def __init__(self, name, params, model=None, pre_computed=None, args=None):
        super().__init__(name, params, model, pre_computed, args)

    def __call__(self, flux, obs, params):
        """
        Applies the calibration offsets to the flux values.

        Parameters
        ----------
        flux : np.ndarray of float
            The afterglow flux.

        obs : core.obs.Observation
            The observation object.

        params : dict
            The calibration parameters containing the 'init' and 'eval' keys.
        """
        if self.pre_computed is not None:
            flux *= self.pre_computed
        else:
            flux *= self.calculate(obs, params)

    @classmethod
    def from_defaults(cls, obs, **kwargs):
        """

        Parameters
        ----------
        obs : ``ampy.core.obs.Observation``
            The observation object.

        Returns
        -------
        ``ampy.modeling.plugins.CalibrationPlugin``
        """
        defaults = CalibrationDefaults()

        params = []
        for offset in obs.offsets:
            param = factory(
                defaults.get(offset) | {
                    "stage": "eval", "plugin": "calibration"
                }
            )
            params.append(param)

        return cls(
            name="calibration",
            params=np.asarray(params),  # noqa
            model=import_from_string(
                "ampy.modeling.models.builtin.calibration_offset_model"
            )
        )


class ChiSquaredPlugin(PluginBase):
    """ Likelihood plugin adapter for AMPy. """
    module = 'inference'

    def __init__(self, name, params, model=None, pre_computed=None, args=None):
        super().__init__(name, params, model, pre_computed, args)

    def __call__(self, flux, obs, params):
        """
        Applies the calibration offsets to the flux values.

        Parameters
        ----------
        flux : np.ndarray of float
            The afterglow flux.

        obs : core.obs.Observation
            The observation object.

        params : dict
            The parameters containing the 'init' and 'eval' keys.
        """
        return self.model(flux, obs, params)
