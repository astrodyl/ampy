import numpy as np

from ampy.core import utils as math_utils
from ampy.inference.params import MCMCFittingParameter


class ParameterView:
    """
    Maps the sampled array of values to an organized dictionary.

    Most samplers simply return an array of sampled values, with no information
    regarding what the parameters are or what scale they're in. This class
    contains the information required to map that sampled array to a dict with
    the parameter name, scale, and plug-in information.

    Parameters
    ----------
    plugins : np.ndarray of ampy.core.plugins.<plugin>
        The plugins containing the params.

    params : array_like of `inference.params.MCMCParameter
        The fixed and fitting MCMC parameters.

    Attributes
    ----------
    positions : dict of np.ndarray of bool
        The positions of the fixed/fitting parameters and their plugins.

    plugin_names : set of str
        The plugins associated with the parameters.
    """
    def __init__(self, plugins, params):
        self.plugins = plugins
        self.params = params
        self.positions = {}
        self.plugin_names = set()

        if not isinstance(params, np.ndarray):
            params = np.asarray(params, dtype=object)

        # Initialize positions
        fixed_pos = np.full(params.size, fill_value=False)

        for i, param in enumerate(params):
            fixed_pos[i] = param.fixed  # noqa
            plugin_name = param.plugin  # noqa

            # Store the plugin names
            self.plugin_names.add(plugin_name)

            # Organize positions by plugin names
            if plugin_name not in self.positions:
                self.positions[plugin_name] = np.full(
                    params.size, fill_value=False
                )
            self.positions[plugin_name][i] = True

        # Store the actual fixed and fitting parameters separately
        self.fixed = params[fixed_pos]
        self.fitting = params[~fixed_pos]

        # Organize the plugins for later convenience
        plugin_dict = {'modeling': [], 'inference': []}

        for plugin in self.plugins:
            plugin_dict[plugin.module].append(plugin)

        self.modeling_plugins = np.asarray(plugin_dict['modeling'])
        self.inference_plugins = np.asarray(plugin_dict['inference'])

    @classmethod
    def from_plugins(cls, plugins, obs=None):
        """
        Creates a ParameterView from the list of plugins.

        Parameters
        ----------
        plugins : iterable of `ampy.modeling.plugins`
            The plugins to be used with MCMC fitting.

        obs : ampy.core.obs.Observation, optional
            The input observation. If provided, it calls plugin.optimize(obs)
            for rach plugin.

        Returns
        -------
        ParameterView
        """
        params = []

        for plugin in plugins:
            if obs is not None:
                plugin.optimize(obs)

            for param in plugin.params:
                params.append(param)

        return cls(plugins, np.asarray(params, dtype=object))

    def has_param(self, name):
        """
        Determines if `name` is in the parameter list.

        Parameters
        ----------
        name : str
            The name of the parameter to check.

        Returns
        -------
            True if `name` is in the parameter list.
        """
        return True if name in [p.name for p in self.params] else False

    def get_params(self, plugin=None, ptype=None):
        """
        Returns the requested parameters.

        Parameters
        ----------
        plugin : str, optional
            The plugin name. If none, return all parameters. Else, only the
            parameters associated with the given plugin are returned.

        ptype : str, optional
            The parameter type. If none, return all parameters. Else, it must
            be none of 'fixed' or 'fitting'.

        Returns
        -------
        dict
        """
        if plugin is not None and plugin not in self.plugin_names:
            raise ValueError(f"{plugin} is not a valid plugin.")

        if plugin is None:
            return (
                self.params if ptype is None
                else getattr(self, f"_{ptype}")
            )

        # Filter by plugin name
        params = self.params[self.positions[plugin]]

        # Filter by fixed or fitting type
        if ptype is not None:
            for param in params:
                if param not in getattr(self, f"_{ptype}"):
                    params.pop(param)

        return params

    def samples_to_dict(self, theta, plugin=None):
        """
        Maps MCMC samples to an organized plugin dictionary.

        Parameters
        ----------
        theta : np.array of float
            The sampled MCMC values.

        plugin : str, optional
            Return only the parameters for this plugin.

        Returns
        -------
        dict
            The parameters organized by plugin.
        """
        if theta.size != len(self.fitting):
            raise ValueError(
                f'Size mismatch: theta[{theta.size}] != params'
                f'[{len(self.fitting)}].'
            )

        # The requested plugins to be returned
        req_plugins = [plugin] if plugin else self.plugin_names

        # Initialize the result with plugins
        params = {plugin: {'init': {}, 'eval': {}} for plugin in req_plugins}

        # Handle the fitting parameters
        for i, param in enumerate(self.fitting):
            param: MCMCFittingParameter = param

            if param.plugin in req_plugins:
                val = math_utils.to_scale(
                    theta[i], param.infer_scale, param.model_scale
                )
                params[param.plugin][param.stage][param.name] = val

        # Handle the fixed parameters
        for param in self.fixed:
            if param.plugin in req_plugins:
                params[param.plugin][param.stage][param.name] = param.value

        return params
