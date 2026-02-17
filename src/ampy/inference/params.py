import copy

from ampy.core.structs import ScaleType
from ampy.inference import priors


def factory(d: dict):
    """
    Instantiates a MCMCParameter from the dict `d`.

    Parameters
    ----------
    d : dict
        The parameter values.

    Returns
    -------
    MCMCFixedParameter or MCMCFittingParameter
        Instantiated from the dict `d`.
    """
    return MCMCFittingParameter.from_dict(d) if 'prior' in d \
        else MCMCFixedParameter.from_dict(d)


class MCMCParameter:
    """
    A parameter to be used with MCMC.

    Attributes
    ----------
    name : str
        The name of the parameter.

    stage : str
        The name of the usage stage. Must be one of 'init` or 'eval'.

    plugin : str
        The name of the associated plugin.
    """
    def __init__(
        self,
        name: str,
        stage: str,
        plugin: str,
    ):
        self.name = name
        self.stage = stage
        self.plugin = plugin

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
        stage: str,
        plugin: str,
    ):
        super().__init__(name, stage, plugin)
        self.value = value

    def __repr__(self) -> str:
        """ Human-readable representation. """
        return (
            f'MCMCFixedParameter(name={self.name}, value={self.value}, '
            f'plugin={self.plugin}, stage={self.stage})'
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
            Instantiated from `d`.
        """
        if not isinstance(d.get('value'), (int, float)):
            raise TypeError(
                f"Expected a number for value in `{d.get('name')}`. "
                f"Received `{type(d.get("value"))}` instead."
            )
        return cls(d['name'], d['value'], d['stage'], d['plugin'])

    @property
    def fixed(self) -> bool:
        return True

    def serialize(self):
        """"""
        return copy.deepcopy(vars(self))


class MCMCFittingParameter(MCMCParameter):
    """
    A free parameter to be sampled with the MCMC routine.

    Attributes
    ----------
    prior : `ampy.inference.priors.<PriorType>`
        The prior probability distribution.

    infer_scale : `ampy.core.structs.ScaleType`
        The scale of the parameter during inference.

    model_scale : `ampy.core.structs.ScaleType`
        The scale of the parameter during modeling.
    """
    def __init__(
        self,
        name: str,
        prior,
        stage: str,
        plugin: str,
        infer_scale: ScaleType,
        model_scale: ScaleType
    ):
        super().__init__(name, stage, plugin)

        self.prior = prior
        self.infer_scale = infer_scale
        self.model_scale = model_scale

    def __repr__(self) -> str:
        """ Human-readable string. """
        return (
            f'MCMCFixedParameter(name={self.name}, prior={self.prior}, '
            f'plugin={self.plugin}, stage={self.stage})'
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

        # If infer_scale is None, default to linear. If model_scale
        # is None, default to whatever infer_scale is.
        infer_scale = ScaleType(d.get('infer_scale', 'linear'))
        model_scale = ScaleType(d.get('model_scale', infer_scale))

        return cls(
            d['name'], priors.prior_factory(d['prior']), d['stage'],
            d['plugin'], infer_scale, model_scale
        )

    @property
    def fixed(self) -> bool:
        return False

    def serialize(self):
        """"""
        out = copy.deepcopy(vars(self))

        # Overwrite non-serializable attributes
        out['prior'] = self.prior.serialize()
        out['infer_scale'] = self.infer_scale.value
        out['model_scale'] = self.model_scale.value

        return out
