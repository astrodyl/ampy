import copy
from enum import Enum

import numpy as np
from scipy import stats
from typing_extensions import override

from ampy.core.structs import BoundedMixin


class Prior(Enum):
    """ Supported prior types. """
    GAUSSIAN   = 'gaussian'
    UNIFORM    = 'uniform'
    SINE       = 'sine'
    MILKYWAYRV = 'milkywayrv'


def prior_factory(d: dict):
    """
    Instantiates the appropriate prior class.

    Parameters
    ----------
    d : dict
        Contains the required key : value pairs for the prior specified using
        the `type` key.

    Returns
    -------
    ``ampy.inference.priors.<Prior>``
        The prior object.
    """
    ptype = Prior(d.get('type'))

    match ptype:
        case Prior.GAUSSIAN:
            return GaussianPrior.from_dict(d)

        case Prior.UNIFORM:
            return UniformPrior.from_dict(d)

        case Prior.SINE:
            return SinePrior.from_dict(d)

        case Prior.MILKYWAYRV:
            return MilkyWayRvPrior.from_dict(d)


class UniformPrior(BoundedMixin):
    """
    Uniform prior with an optional initialization region.

    This class represents a uniform prior over the closed interval
    ``[lower, upper]``. It also supports an optional *initialization region*
    (``initial_guess ± initial_sigma``) used when drawing initial samples
    for MCMC walkers. This can help concentrate starting positions without
    changing the actual prior used for inference.

    Attributes
    ----------
    lower : float
        Lower bound of the prior (inherited from :class:`~BoundedMixin`).

    upper : float
        Upper bound of the prior (inherited from :class:`~BoundedMixin`).

    initial_guess : float or None
        Center of the initialization region, if provided.

    initial_sigma : float or None
        Half-width of the initialization region, if provided.

    Notes
    -----
    * The initialization region affects :meth:`draw` only when ``initial=True``.
      It does **not** change the underlying prior used during inference.

    * :meth:`evaluate` returns the **log-prior** (0 inside bounds, ``-np.inf``
      outside), which is convenient for log-posterior calculations.
    """

    type = Prior.UNIFORM

    def __init__(
        self,
        lower: float,
        upper: float,
        initial_guess: float = None,
        initial_sigma: float = None
    ):
        BoundedMixin.__init__(self, lower, upper)
        self.initial_guess = initial_guess
        self.initial_sigma = initial_sigma

    def __repr__(self) -> str:
        """ Human-readable representation. """
        class_name = self.__class__.__name__
        return f"{class_name}(lower={self.lower}, upper={self.upper})"

    def __call__(self, *args, **kwargs):
        """
        Alias for :meth:`evaluate`.

        This allows the prior instance to be called like a function.

        Returns
        -------
        float or numpy.ndarray
            Log-prior evaluated at the provided value(s).
        """
        return self.evaluate(*args, **kwargs)

    @classmethod
    def from_dict(cls, d: dict):
        """
        Construct a :class:`UniformPrior` from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing keys:

            - ``"lower"`` : float
            - ``"upper"`` : float
            - ``"initial_guess"`` : float, optional
            - ``"initial_sigma"`` : float, optional

        Returns
        -------
        UniformPrior
            Instantiated prior.

        Raises
        ------
        TypeError
            If any provided values are not numeric.

        Examples
        --------
        >>> prior = UniformPrior.from_dict({"lower": 0.0, "upper": 1.0})
        >>> prior.lower, prior.upper
        (0.0, 1.0)

        >>> prior = UniformPrior.from_dict(
        ...     {"lower": 0.0, "upper": 10.0, "initial_guess": 5.0, "initial_sigma": 1.0}
        ... )
        """
        if not isinstance(lower := d.get('lower'), (int, float)):
            raise TypeError('Uniform lower must be a number.')

        if not isinstance(upper := d.get('upper'), (int, float)):
            raise TypeError('Uniform upper must be a number.')

        if (initial := d.get('initial_guess')) is not None:
            if not isinstance(initial, (int, float)):
                raise TypeError('Initial guess must be a number.')

        if (sigma := d.get('initial_sigma')) is not None:
            if not isinstance(sigma, (int, float)):
                raise TypeError('Initial sigma must be a number.')

        return cls(lower, upper, initial, sigma)

    def _bounds(self, initial: bool):
        """
        Determine sampling bounds for :meth:`draw`.

        Parameters
        ----------
        initial : bool
            If True and both ``initial_guess`` and ``initial_sigma`` are set,
            returns a truncated interval:

            ``[initial_guess - initial_sigma, initial_guess + initial_sigma]``

            clipped to the prior bounds ``[lower, upper]``.

            If False (or if initialization parameters are not set), returns the
            full prior bounds.

        Returns
        -------
        tuple[float, float]
            Sampling bounds ``(low, high)``.
        """
        if initial:
            if self.initial_guess is not None and self.initial_sigma is not None:
                return (
                    max(self.initial_guess - self.initial_sigma, self.lower),
                    min(self.initial_guess + self.initial_sigma, self.upper),
                )
        return self.lower, self.upper

    def serialize(self):
        """
        Serialize the prior parameters.

        Returns
        -------
        dict
            Dictionary representation containing ``lower``, ``upper``,
            ``initial_guess``, and ``initial_sigma``.
        """
        return {
            'lower': self.lower, 'upper': self.upper,
            'initial_guess': self.initial_guess,
            'initial_sigma': self.initial_sigma
        }

    def draw(self, n: int, initial: bool = True) -> float | np.ndarray:
        """
        Draw samples from the uniform prior.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        initial : bool, optional
            If True and initialization parameters are set, samples are drawn
            from the clipped initialization region. If False, samples are drawn
            from the full prior range.

        Returns
        -------
        float or numpy.ndarray
            If ``n == 1``, a scalar sample is returned.
            Otherwise, an array of shape ``(n,)`` containing samples.

        Examples
        --------
        >>> prior = UniformPrior(lower=0.0, upper=10.0, initial_guess=5.0, initial_sigma=1.0)
        >>> s0 = prior.draw(n=1000, initial=True)   # mostly in [4, 6]
        >>> s1 = prior.draw(n=1000, initial=False)  # in [0, 10]
        """
        return np.random.uniform(*self._bounds(initial), size=n)

    def evaluate(self, x: float) -> float:
        """
        Evaluate the log-prior at ``x``.

        Parameters
        ----------
        x : float
            Value at which to evaluate the prior.

        Returns
        -------
        float
            Log-prior value:

            * ``0.0`` if ``x`` is within bounds
            * ``-np.inf`` if ``x`` is outside bounds

        Notes
        -----
        This method returns the log-density for convenience in log-posterior
        calculations.
        """
        return 0.0 if self.encompasses(x) else -np.inf


class GaussianPrior:
    """
    Gaussian (normal) prior distribution.

    This class represents a univariate Gaussian prior with mean ``mu`` and
    standard deviation ``sigma``. It provides utilities for sampling,
    evaluating the probability density function (PDF), serialization,
    and convenient bounds for initialization or visualization.

    The prior is defined as:

    .. math::

        p(x) = \\frac{1}{\\sqrt{2\\pi}\\,\\sigma}
               \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)

    Attributes
    ----------
    mu : float
        Mean of the distribution.

    sigma : float
        Standard deviation of the distribution.

    Notes
    -----
    The ``lower`` and ``upper`` properties return the ±3σ bounds of the
    distribution. These are often useful for visualization or initialization
    ranges but are **not hard bounds** of the prior.
    """
    type = Prior.GAUSSIAN

    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")

        self.mu = mu
        self.sigma = sigma

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(mu={self.mu}, sigma={self.sigma})"

    @property
    def lower(self) -> float:
        """
        Lower 3σ bound of the prior.

        Returns
        -------
        float
            ``mu - 3*sigma``.

        Notes
        -----
        This is a convenience range estimate, not a truncation of the prior.
        """
        return self.mu - 3.0 * self.sigma

    @property
    def upper(self) -> float:
        """
        Upper 3σ bound of the prior.

        Returns
        -------
        float
            ``mu + 3*sigma``.

        Notes
        -----
        This is a convenience range estimate, not a truncation of the prior.
        """
        return self.mu + 3.0 * self.sigma

    @classmethod
    def from_dict(cls, d: dict):
        """
        Construct a :class:`GaussianPrior` from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing prior parameters with keys:

            - ``"mu"`` : float
            - ``"sigma"`` : float

        Returns
        -------
        GaussianPrior
            Instantiated prior.

        Raises
        ------
        TypeError
            If ``mu`` or ``sigma`` are not numeric.

        Examples
        --------
        >>> prior = GaussianPrior.from_dict({"mu": 0.0, "sigma": 1.0})
        >>> prior.mu, prior.sigma
        (0.0, 1.0)
        """
        if (mu := d.get('mu')) is not None:
            if not isinstance(mu, (int, float)):
                raise TypeError('Mu must be a number.')

        if (sigma := d.get('sigma')) is not None:
            if not isinstance(sigma, (int, float)):
                raise TypeError('Sigma must be a number.')

        return cls(mu, sigma)

    def serialize(self):
        """
        Serialize the prior parameters.

        Returns
        -------
        dict
            Dictionary representation containing ``mu`` and ``sigma``.

        Notes
        -----
        This method returns a deep copy so that external modification does
        not affect the prior instance.
        """
        return copy.deepcopy(vars(self))

    def draw(self, n: int) -> float | np.ndarray:
        """
        Draw samples from the Gaussian prior.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        float or numpy.ndarray
            If ``n == 1``, a scalar sample is returned.
            Otherwise, an array of shape ``(n,)`` containing samples.

        Examples
        --------
        >>> prior = GaussianPrior(mu=0.3, sigma=0.1)
        >>> s = prior.draw(n=1000)
        >>> abs(s.mean() - prior.mu) < 0.01
        True
        """
        return np.random.normal(self.mu, self.sigma, size=n)

    def evaluate(self, x) -> float | np.ndarray:
        """
        Evaluate the Gaussian probability density at ``x``.

        Parameters
        ----------
        x : float or array_like
            Value(s) at which to evaluate the prior PDF.

        Returns
        -------
        float or numpy.ndarray
            Probability density evaluated at ``x``. Shape matches input.

        Notes
        -----
        This returns the **probability density**, not the log-density.
        For MCMC inference, the log-PDF is typically used elsewhere.

        Examples
        --------
        >>> prior = GaussianPrior(mu=0.0, sigma=1.0)
        >>> prior.evaluate(0.0)
        0.3989...
        >>> prior.evaluate([0.0, 1.0]).shape
        (2,)
        """
        return stats.norm.logpdf(x, loc=self.mu, scale=self.sigma)


class MilkyWayRvPrior:
    """
    Milky Way Rv Prior.

    References
    ----------
    Adam Trotter (2011):
        The Gamma-Ray Burst Afterglow Modeling Project:
        Foundational Statistics and Absorption & Extinction Models.
        See: Page 108.
    """
    def __init__(self):
        self.mu = 0.4150
        self.sigma_low = 0.00779
        self.sigma_high = 0.09074

    def __repr__(self) -> str:
        """ Human-readable representation. """
        return (
            f"{self.__class__.__name__}(mu={self.mu}, "
            f"sigma={(self.sigma_low, self.sigma_high)})"
        )

    @property
    def lower(self) -> float:
        """ Returns the lower 3-sigma bound of the prior. """
        return 0.302  # self.mu - 3.0 * self.sigma_low

    @property
    def upper(self) -> float:
        """ Returns the upper 3-sigma bound of the prior. """
        return 0.778  # self.mu + 3.0 * self.sigma_high

    @classmethod
    def from_dict(cls, d: dict):
        """
        Creates instance from dict ensuring values are OK.

        Parameters
        ----------
        d : dict
            `mu`   : float
            `low`  : float
            `high` : float

        Returns
        -------
        GaussianPrior
            Instantiated from dictionary
        """
        return cls()

    def serialize(self):
        """"""
        return copy.deepcopy(vars(self))

    def draw(self, n: int) -> float | np.ndarray:
        """
        Draws `n` samples from the asymmetric distribution.

        Parameters
        ----------
        n : int
            The number of samples to draw.

        Returns
        -------
        float or np.ndarray, with shape (n, 1)
            The samples drawn.
        """
        # Probabilities for each side (proportional to sigmas)
        p_left = self.sigma_low / (self.sigma_low + self.sigma_high)

        # Randomly decide which branch for each sample
        branches = np.random.rand(n) < p_left

        samples_log = np.zeros(n)

        # Left branch: truncated normal, x < mu
        n_left = branches.sum()
        if n_left > 0:
            samples_log[branches] = stats.truncnorm.rvs(
                -np.inf, 0.0, loc=self.mu, scale=self.sigma_low, size=n_left
            )

        # Right branch: truncated normal, x >= mu
        n_right = n - n_left
        if n_right > 0:
            samples_log[~branches] = stats.truncnorm.rvs(
                0.0, np.inf, loc=self.mu, scale=self.sigma_high, size=n_right
            )

        return samples_log

    def evaluate(self, x) -> float | np.ndarray:
        """
        Evaluates the prior at the sampled value `x`.

        Parameters
        ----------
        x : float or array_like
            The sampled value.

        Returns
        -------
        float or np.ndarray of float
            The prior evaluated at `x`.
        """
        sig = self.sigma_low if x < self.mu else self.sigma_high
        norm = (self.sigma_low + self.sigma_high) * np.sqrt(np.pi / 2)

        return np.exp(-0.5 * ((x - self.mu) / sig) ** 2) / norm


class SinePrior(UniformPrior):
    r"""
    Sine prior for an orientation angle.

    This prior is useful for parameters representing an angle :math:`\theta`
    drawn from an isotropic distribution of directions. For an isotropic
    orientation, the probability of observing a particular polar angle
    scales as the solid-angle element:

    .. math::

        d\Omega = 2\pi \sin(\theta)\, d\theta

    Therefore, the (unnormalized) density is :math:`p(\theta) \propto \sin(\theta)`
    over a bounded interval :math:`\theta \in [\theta_\mathrm{min}, \theta_\mathrm{max}]`.

    On the interval ``[lower, upper]`` (in radians), the properly normalized PDF is:

    .. math::

        p(\theta) = \frac{\sin(\theta)}{\cos(\theta_\mathrm{min}) - \cos(\theta_\mathrm{max})}

    Parameters
    ----------
    lower : float
        Lower bound of the prior in radians.

    upper : float
        Upper bound of the prior in radians.

    initial_guess : float, optional
        Center of an optional initialization region used when sampling initial
        values (e.g., initial walker positions).

    initial_sigma : float, optional
        Half-width of the initialization region about ``initial_guess``. When used,
        the initialization region is clipped to ``[lower, upper]``.

    Notes
    -----
    Sampling is performed using inverse transform sampling by drawing uniformly
    in :math:`\cos(\theta)` over the bounds and converting back to :math:`\theta`
    via :math:`\arccos`.

    See Also
    --------
    UniformPrior : Provides bounded behavior and optional initialization
    bounds.
    """

    type = Prior.SINE

    def __init__(
        self,
        lower: float,
        upper: float,
        initial_guess: float = None,
        initial_sigma: float = None
    ):
        super().__init__(lower, upper, initial_guess, initial_sigma)

    @override
    def draw(self, n: int, initial: bool = True) -> float | np.ndarray:
        """
        Draw samples from the sine prior.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        initial : bool, optional
            If True and initialization parameters are set, draws from the
            clipped initialization region; otherwise draws from the full bounds.

        Returns
        -------
        float or numpy.ndarray
            If ``n == 1``, a scalar sample is returned.
            Otherwise, an array of shape ``(n,)`` containing samples in radians.
        """
        lower, upper = self._bounds(initial)

        cos_min = np.cos(lower)
        cos_max = np.cos(upper)
        cos_theta = cos_min - np.random.rand(n) * (cos_min - cos_max)
        return np.arccos(cos_theta)

    @override
    def evaluate(self, x: float) -> float:
        """
        Evaluate the log-prior at ``x``.

        Parameters
        ----------
        x : float
            Angle value in radians.

        Returns
        -------
        float
            Log-density:

            * ``log(p(x))`` if ``x`` is within bounds
            * ``-np.inf`` if ``x`` is outside bounds
        """
        if not self.encompasses(x):
            return -np.inf

        return np.sin(x) / (np.cos(self.lower) - np.cos(self.upper))

    def serialize(self):
        """
        Serialize the prior parameters.

        Returns
        -------
        dict
            Dictionary representation containing ``lower``, ``upper``,
            ``initial_guess``, and ``initial_sigma``.

        Notes
        -----
        This method returns a deep copy so that external modification does
        not affect the prior instance.
        """
        return copy.deepcopy(vars(self))
