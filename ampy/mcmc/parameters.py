from pathlib import Path

import numpy as np

from ampy.core.structs import ScaleType
from ampy.core import utils
from ampy.mcmc import priors


def factory(d: dict):
    """
    Instantiates a MCMCParameter from the dict ``d``.

    Parameters
    ----------
    d : dict
        The parameter values.

    Returns
    -------
    MCMCFixedParameter or MCMCFittingParameter
        Instantiated from the dict ``d``.
    """
    return MCMCFittingParameter.from_dict(d) if 'prior' in d \
        else MCMCFixedParameter.from_dict(d)


# noinspection PyUnresolvedReferences
class Parameters:
    """
    Container class for afterglow parameters.

    Parameters
    ----------
    params : array_like
        The fixed and fitting MCMC parameters.

    Attributes
    ----------
    all : np.ndarray
        All MCMC parameters.

    fixed : np.ndarray
        The fixed MCMC parameters.

    fitting : np.ndarray
        The fitting MCMC parameters.

    pos : dict
        The positions of the parameters and their
        categories.
    """
    # Nyaa :3
    _valid_cats = ('model', 'extinction', 'host', 'offsets', 'slop')

    def __init__(self, params):
        if not isinstance(params, np.ndarray):
            params = np.asarray(params)

        init_arr = np.full(params.size, False)

        # Initialize positions for each cat
        self.pos = {
            cat : np.array(init_arr, copy=True)
            for cat in self._valid_cats
        }

        # Initialize positions for additional useful locators
        self.pos['fixed'] = np.array(init_arr, copy=True)
        self.pos['fitting'] = np.array(init_arr, copy=True)

        # Determine positions of the categories
        # Stored once here instead of in a prop
        # to speed up MCMC as much as possible.
        for i, p in enumerate(params):
            self.pos['fixed'][i] = p.fixed
            self.pos['fitting'][i] = not p.fixed
            self.pos[p.category][i] = True

        self.all = params
        self.fixed = params[self.pos['fixed']]
        self.fitting = params[self.pos['fitting']]

    @classmethod
    def from_toml(cls, d):
        """
        Instantiate ``Parameters`` from a dict.

        Parameters
        ----------
        d : dict or str or Path
            Either a path to the toml file or a dict
            representing the TOML file.

        Returns
        -------
        Parameters
            Instantiated from ``d``.
        """
        if isinstance(d, (str, Path)):
            d = utils.TOMLReader(d).read()

        params = []
        for cat, vals in d.items():
            for val in vals:
                params.append(
                    factory(val | {'category': cat})
                )

        return cls(np.asarray(params, dtype=object))

    def has(self, name):
        """
        Determines if ``name`` is in the parameter list.

        Parameters
        ----------
        name : str
            The name of the parameter to check.

        Returns
        -------
            True if ``name`` is in the parameter list.
        """
        return True if name in [p.name for p in self.all] else False

    def samples_to_dict(self, theta, cat=None, scale='linear'):
        """
        Maps MCMC samples to a dictionary.

        Parameters
        ----------
        theta : np.array of float
            The MCMC samples.

        cat : str, optional
            Limit the dictionary to the ``cat`` categories.

        scale : str, optional, default='linear'
            The scale to return the parameters in.

        Returns
        -------
        dict
            The parameters in dict form.
        """
        if theta.size != len(self.fitting):
            raise ValueError(
                f'Size mismatch: theta[{theta.size}] != params'
                f'[{len(self.fitting)}].'
            )

        # Define the categories to return ~Nyaa :3
        cats = [cat] if cat else self._valid_cats

        # Initialize the result with cats
        params = {cat: {} for cat in cats}

        # Needs to be in order of fitting then fixed (to match theta)
        for i, p in enumerate(np.concatenate([self.fitting, self.fixed])):
            if p.category in cats:
                val = utils.to_scale(
                    theta[i] if i < theta.size else p.value, p.scale, scale
                )
                params[p.category][p.name] = val

        return params


class MCMCParameter:
    """
    A parameter to be used with MCMC.

    Attributes
    ----------
    name : str
        The name of the parameter.

    scale : `ampy.core.structs.ScaleType`
        The scale of the parameter.

    category : str
        One of: `model`, `extinction`, `host`,
        `offsets`, or `slop`

    group : str, optional
        The data group of the parameter.
    """
    def __init__(
            self,
            name: str,
            scale: ScaleType,
            category: str,
            group: str = None
    ):
        self.name = name
        self.scale = scale
        self.group = group
        self.category = category

    @classmethod
    def from_dict(cls, d: dict):
        """ Placeholder. """
        raise NotImplementedError(
            '`from_dict` method is not implemented.'
        )


class MCMCFixedParameter(MCMCParameter):
    """
    A fixed parameter to be used within a model for MCMC.

    Attributes
    ----------
    value : float
        The fixed value of the parameter.
    """
    def __init__(
            self,
            name: str,
            value: float,
            scale: ScaleType,
            category: str,
            group: str = None
    ):
        super().__init__(name, scale, category, group)
        self.value = value

    def __repr__(self) -> str:
        """ Human-readable string. """
        return (
            f'MCMCFixedParameter(name={self.name}, '
            f'cat={self.category}, '
            f'val={self.value})'
        )

    @classmethod
    def from_dict(cls, d: dict):
        """
        Instantiates the class from a dictionary.

        Parameters
        ----------
        d : dict
            The class attributes and values.

        Returns
        -------
        MCMCFixedParameter
            Instantiated from ``d``.
        """
        if not isinstance(d.get('value'), (int, float)):
            raise TypeError(
                f"Expected a number for value in `{d.get('name')}`. "
                f"Received `{type(d.get("value"))}` instead."
            )

        if not isinstance(d.get('scale'), str):
            raise TypeError(
                f'Expected type `str` for scale in `{d.get('name')}`. '
                f'Received `{type(d.get("scale"))}` instead.'
            )

        return cls(
            d.get('name'),
            d.get('value'),
            ScaleType(d.get('scale')),
            d.get('category'),
            d.get('group')
        )

    @property
    def fixed(self) -> bool:
        return True


class MCMCFittingParameter(MCMCParameter):
    """
    A free parameter to be sampled with the MCMC routine.

    Attributes
    ----------
    prior : `ampy.core.enums.Prior`
        The prior probability distribution.
    """
    def __init__(
            self,
            name: str,
            scale: ScaleType,
            prior,
            category: str,
            group: str = None
    ):
        MCMCParameter.__init__(self, name, scale, category, group)
        self.prior = prior

    def __repr__(self) -> str:
        """ Human-readable string. """
        return (
            f'MCMCFittingParameter('
            f'name={self.name}, '
            f'cat={self.category})'
        )

    @classmethod
    def from_dict(cls, d: dict):
        """
        Instantiates the class from a dictionary.

        Parameters
        ----------
        d : dict
            The class attributes and values.

        Returns
        -------
        MCMCFittingParameter
            Instantiated from `d`.
        """
        if not isinstance(d.get('prior'), dict):
            raise TypeError(
                f'Expected type `dict` for prior in {d.get('name')}. '
                f'Received `{type(d.get("prior"))}` instead.'
            )

        if not isinstance(d.get('scale'), str):
            raise TypeError(
                f'Expected type `str` for scale in {d.get('name')}. '
                f'Received `{type(d.get("scale"))}` instead.'
            )

        return cls(
            d.get('name'),
            ScaleType(d.get('scale')),
            priors.prior_factory(d.get('prior')),
            d.get('category'),
            d.get('group')
        )

    @property
    def fixed(self) -> bool:
        return False
