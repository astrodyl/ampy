import copy
from contextlib import nullcontext

import emcee
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from ampy.core import utils

try:
    import ptemcee
except ImportError:
    pass


def get_pool_context(workers=None, executor='process'):
    """
    If ``executor==process`` and ``workers>1``:
        Returns ``ProcessPoolExecutor(max_workers=workers)``.

        Since processes need to load everything into memory, this
        should only be used if the likelihood calculation takes
        about one-second or more to calculate.

    If ``executor==thread`` and ``workers>1``:
        Returns ``ThreadPoolExecutor(max_workers=workers)``.

        Note that unless a free-threaded Python is installed,
        multithreading will not yield any benefits. Even if a
        no-GIL Python version is used, the performance increase
        depends  on the likelihood implementation.

        Pure Python implementations will see a large performance
        increase. If the likelihood uses Cython, then it depends
        on how the code is compiled and optimized.

    Parameters
    ----------
    workers : int, optional, default=None
        The max number of workers.

    executor : str, optional, default='process'
        See above docstring for details. Must be ``process``
        or ``thread``.

    Returns
    -------
    ``ProcessPoolExecutor`` or ``ThreadPoolExecutor`` or ``nullcontext``
        The pool context manager.
    """
    if workers and workers > 1:

        if executor == 'process':
            return ProcessPoolExecutor(max_workers=workers)

        if executor == 'thread':
            return ThreadPoolExecutor(max_workers=workers)

    return nullcontext()


class PTSampler:
    """
    Provides an API adapter that matches ``emcee``.

    Parallel tempering in ``emcee`` stopped receiving
    support and was removed from official releases.

    Using the latest version of ``emcee`` that supports
    the PTSampler forces users to use very old packages
    that the PTSampler depends on.

    There is a community developed version called ``ptemcee``.
    However, the authors stopped maintaining it years ago.
    Sadly, there's also zero documentation and/or tutorials,
    and the API is completely different.

    This class aims to provide an API that matches ``emcee``.
    Methods are only added on an as-needed basis and are by
    no means complete.

    Parameters
    ----------
    ntemps : int
        The number of temperatures.

    nwalkers : int
        The number of walkers.

    ndim : int
        The number of fitting dimensions.

    log_like, log_prior
        The log likelihood and log prior methods.

    log_l_args, log_p_args : array_like, optional
        The log likelihood and log prior arguments.

    log_l_kwargs, log_p_kwargs : array_like, optional
        The log likelihood and log prior kwargs.

    pool : optional
        An object with a ``map`` method that follows the
        same calling sequence as emcee's built-in ``map``
        function. This is generally used to compute the
        log-probabilities in parallel.

    kwargs
        Any kwargs to be passed to the sampler.
    """
    def __init__(
        self, ntemps, nwalkers, ndim, log_like, log_prior,
        log_l_args=(), log_p_args=(), log_l_kwargs=(), log_p_kwargs=(),
        pool=None, **kwargs
    ):
        mapper = pool.map if pool is not None else None

        # Initialize the sampler
        self._sampler = ptemcee.Sampler(
            nwalkers, ndim, log_like, log_prior,
            log_l_args, log_p_args, log_l_kwargs, log_p_kwargs,
            ptemcee.make_ladder(ndim, ntemps), mapper=mapper, **kwargs
        )
        self._chain = None
        self._iteration = 0

        self._ndim = ndim
        self._ntemps = ntemps
        self._nwalkers = nwalkers

    @property
    def sampler(self):
        return self._sampler

    @property
    def chain(self):
        return self._chain

    @property
    def iteration(self):
        return self._iteration

    @property
    def ntemps(self):
        return self._ntemps

    @property
    def nwalkers(self):
        return self._nwalkers

    @property
    def ndim(self):
        return self._ndim

    def run_mcmc(self, x0, iterations, **kwargs):
        """
        Perform MCMC sampling.

        Parameters
        ----------
        x0 : np.ndarray
            The initial position vector.

        iterations : int
            The number of steps to run.

        kwargs
            thin_by

            random

        Returns
        -------
        np.ndarray with shape [ntemps, nwalkers, ndim]
            The last samples.
        """
        self._chain = self.sampler.chain(x0, **kwargs)
        self._chain.run(iterations)
        self._iteration = self._chain.length

        return self.chain.x[-1]

    def reset(self):
        """
        Overwrites the sampler with a new one.

        There's no reset method in ``ptemcee`` that I'm aware
        of. Overwriting with a new sampler is safer than
        attempting to reset attributes individually.
        """
        self._sampler = ptemcee.Sampler(
            self.sampler.nwalkers, self.sampler.ndim,
            self.sampler.logl, self.sampler.logp,
            self.sampler.logl_args, self.sampler.logp_args,
            self.sampler.logl_kwargs, self.sampler.logp_kwargs,
            ptemcee.make_ladder(self.ndim, self.ntemps)
        )
        self._chain = None
        self._iteration = 0

    def get_last_sample(self):
        """ Returns last samples with shape [ntemps, nwalkers, ndim]. """
        if self.chain is None:
            raise AttributeError(
                'Tried to get the last sample, but '
                'there are no samples. Have you '
                'called `run_mcmc` yet?'
            )
        return self.chain.x[-1]

    def draw_positions(self, params, models) -> np.ndarray:
        """
        Draw the initial positions from the priors.

        PTSampler requires that the start positions be
        valid (i.e., log posterior is finite). To meet
        this, any invalid position is overwritten with
        the best position for that temperature.

        Parameters
        ----------
        params : Parameters
            The MCMC parameters.

        models : MCMCModel
            The MCMC models.

        Returns
        -------
        np.ndarray of float with shape [ntemps, nwalkers, ndim]
            The starting positions.
        """
        # Draw starting positions
        pos = np.zeros((self.ntemps, self.nwalkers, self.ndim))

        for i in range(self.ntemps):
            for j, p in enumerate(params.fitting):
                pos[i, :, j] = p.prior.draw(self.nwalkers)

        # Overwrite with each temperature's best position
        log_p = np.full((self.ntemps, self.nwalkers), -np.inf)

        for i in range(self.ntemps):
            for j in range(self.nwalkers):
                log_p[i, j] = log_posterior_fn(pos[i, j], params, models)  # type: ignore

            pos[i][np.isinf(log_p[i])] = np.array(
                pos[i][np.nanargmax(log_p[i])], copy=True
            )

        return pos

    def get_value(self, name, flat=False, thin=1, discard=0, temp=0):
        """
        Get the attribute ``name``.

        Parameters
        ----------
        name : str
            Name of the attribute to retrieve.

        flat : bool, optional, default=False
            Flatten the chain across the ensemble.

        thin : int, optional, default=1
            Take only every ``thin`` steps from the
            chain.

        discard : int, optional, default=0
            Discard the first ``discard`` steps in the
            chain as burn-in.

        temp : int, optional, default=0
            Take only the ``temp`` attribute. Defaults to
            the temp at index ``0`` which is the highest
            probability temperature.

        Returns
        -------
        np.ndarray
        """
        if self.chain is None:
            raise AttributeError(
                f'Tried to get {name}, but there '
                f'are no chains. Have you called '
                f'`run_mcmc` yet?'
            )

        try:
            v = getattr(self, name)
        except AttributeError:
            v = getattr(self.chain, name)

        if len(v.shape) == 4:
            # shape(iterations, ntemps, nwalkers, ndim)
            v = v[:, temp, :, :]
        else:
            # shape(iterations, ntemps, nwalkers)
            v = v[:, temp, :]

        # Discard and thin
        v = v[discard + thin - 1: self.iteration: thin]

        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v

    def get_chain(self, **kwargs):
        """
        Get the stored chain of MCMC samples.

        Parameters
        ----------
        kwargs
            flat : bool, optional, default=False
                Flatten the chain across the ensemble.

            thin : int, optional, default=1
                Take only every ``thin`` steps from the
                chain.

            discard : int, optional, default=0
                Discard the first ``discard`` steps in the
                chain as burn-in.

            temp : int, optional, default=0
                Take only the ``temp`` chain. Defaults to
                the temp at the ``0``th index which corresponds
                to the highest probability temperature.

        Returns
        -------
        np.ndarray with shape [..., nwalkers, ndim]
            The samples contained in ``ptemcee.Chain.x``.
        """
        return self.get_value('x', **kwargs)

    def get_log_prob(self, **kwargs):
        """
        Get the chain of log probabilities evaluated at
        the MCMC samples.

        Parameters
        ----------
        kwargs
            flat : bool, optional, default=False
                Flatten the chain across the ensemble.

            thin : int, optional, default=1
                Take only every ``thin`` steps from the
                chain.

            discard : int, optional, default=0
                Discard the first ``discard`` steps in the
                chain as burn-in.

            temp : int, optional, default=0
                Take only the ``temp`` log prob. Defaults to
                the temp at the ``0``th index which corresponds
                to the highest probability temperature.

        Returns
        -------
        np.ndarray with shape [..., nwalkers]
            The chain of log probabilities.
        """
        return self.get_value("logP", **kwargs)


class EnsembleSampler(emcee.EnsembleSampler):
    """
    Adapter for ``emcee.EnsembleSampler``.
    """
    def __init__(self, nwalkers, ndim, log_prob_fn, args, **kw):
        super().__init__(nwalkers, ndim, log_prob_fn, args=args, **kw)

    def draw_positions(self, params, **kwargs) -> np.ndarray:
        """
        Draw the initial positions from the priors.

        Parameters
        ----------
        params : Parameters
            The MCMC parameters.

        kwargs :
            For compatability with ``PTSampler.draw_positions``.

        Returns
        -------
        np.ndarray of float with shape [nwalkers, ndim]
            The starting positions.
        """
        pos = np.zeros((self.nwalkers, self.ndim))

        for i, p in enumerate(params.fitting):
            pos[:, i] = p.prior.draw(self.nwalkers)

        return pos


class MCMC:
    """
    Performs MCMC sampling.

    Parameters
    ----------
    model : MCMCModels
        The afterglow model.

    parameters : Parameters
        The model parameters.
    """
    def __init__(self, model, parameters):
        # Model
        self.models = model
        self.params = parameters
        self.observation = model.obs

        # Sampler
        self.sampler = None
        self.burn_chain = None

        self.start_burn_pos = None
        self.start_run_pos = None

    @property
    def ndim(self):
        """ The number of fitting dimensions. """
        return len(self.params.fitting)

    def get_best_params(self, as_dict=True, **kwargs):
        """
        Returns the sampled values from the chain with the
        highest likelihood.

        Parameters
        ----------
        as_dict : bool, optional, default=True
            If True, return the sampled values as a dict.

        kwargs : dict
            cat : str, optional
                Limit the params to the ``cat`` categories.

            scale : str, optional, default='linear'
                The scale to return the parameters in.

        Returns
        -------
        dict or np.ndarray
            The values from the highest likelihood chain.
        """
        max_index = np.nanargmax(self.sampler.get_log_prob(flat=True))
        params = self.sampler.get_chain(flat=True)[max_index]
        return self.params.samples_to_dict(params, **kwargs) if as_dict else params

    # <editor-fold desc="Sampling Routine">
    def set_sampler(self, sampler, nwalkers, pool, ntemps=None, **kwargs):
        """
        Sets the sampler. Duh.

        Parameters
        ----------
        sampler : str
            The sampler type. Must be ``ensemble`` or ``parallel_tempered``.

        nwalkers : int
            The number of walkers.

        pool : ``ProcessPoolExecutor`` or ``ThreadPoolExecutor`` or ``nullcontext``
            The pool to use for multithreading/processing.
            See ``get_pool_context()`` for details.

        ntemps : int, optional
            The number of temperatures for ``parallel_tempered``.

        kwargs
            Any kwargs to be passed to the sampler.
        """
        if sampler == 'ensemble':
            self.sampler = EnsembleSampler(
                nwalkers, self.ndim, log_posterior_fn,
                args=(self.params, self.models), pool=pool, **kwargs # type: ignore
            )

        elif sampler == 'parallel_tempered':
            self.sampler = PTSampler(
                ntemps, nwalkers, self.ndim, log_likelihood_fn, log_prior_fn,
                log_l_args=(self.params, self.models), log_p_args=(self.params,),
                pool=pool, **kwargs
            )

    def run(
        self, nwalkers, iterations, burn=0, sampler='ensemble',
        workers=None, ntemps=None, sampler_kw=None, run_kw=None
    ):
        """
        Runs the MCMC sampling routine.

        Parameters
        ----------
        nwalkers : int
            The number of walkers.

        iterations : int
            The number of iterations.

        burn : int, optional, default=0
            The number of iterations to burn. If ``burn>0``,
            stores the burn sampler to ``self.burn_sampler``
            before resetting it for the main run.

        sampler : str, optional, default='ensemble'
            Must be ``ensemble`` or ``parallel_tempered``.

        workers : int, optional, default=None
            The max number of workers to use.

        ntemps : int, optional, default=None
            The number of temperatures for ``PTSampler``.

        sampler_kw : dict, optional
            Any kwargs to pass to the sampler.

        run_kw : dict, optional
            Any kwargs to pass to the ``run_mcmc`` method.
        """
        with get_pool_context(workers) as pool:
            self.set_sampler(
                sampler, nwalkers, pool, ntemps, **(sampler_kw or {}))

            start_pos = self.sampler.draw_positions(
                params=self.params, models=self.models)

            if burn < 1:
                self.start_run_pos = start_pos

            else:
                self.start_burn_pos = start_pos

                # Run burn in and save the last position
                self.start_run_pos = (
                    self.sampler.run_mcmc(
                        self.start_burn_pos, burn, **(run_kw or {})
                    )
                )

                # Save the chain if desired for diagnostics. Cannot
                # save the entire sampler because deepcopy detaches
                # the pool which prevents multiprocessing/threading
                self.burn_chain = copy.deepcopy(self.sampler.get_chain())
                self.sampler.reset()

            # Run production
            self.sampler.run_mcmc(
                self.start_run_pos, iterations, **(run_kw or {})
            )


class MCMCModels:
    """
    Container for MCMC models used during fitting.

    Parameters
    ----------
    obs : Observation
        The observational data.

    afg_model :
        The afterglow model.

    afg_kw : dict, optional
        Any kwargs needed to instantiate the model.

    ext_model : optional
        The dust extinction model.

    ext_mw_pc : np.ndarray, optional
        The pre-computed Milky Way extinction values.

    ext_sf_pc : np.ndarray, optional
        The pre-computed source-frame extinction values.
    """
    def __init__(
        self, obs, afg_model,
        afg_kw=None, ext_model=None, ext_mw_pc=None, ext_sf_pc=None
    ):
        # Afterglow
        self.afg_model = afg_model
        self.afg_kw = afg_kw if afg_kw else {}

        # Extinction
        self.ext_model = ext_model
        self.ext_mw_pc = ext_mw_pc
        self.ext_sf_pc = ext_sf_pc

        # Observation
        self.obs = obs

    def model(self, params):
        """
        Models the observed GRB afterglow flux.

        Parameters
        ----------
        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        Returns
        -------
        np.ndarray of float
            The modeled observed GRB afterglow flux.
        """
        # Model the GRB afterglow flux
        modeled = self.model_afterglow(params)

        if np.isnan(modeled.min()):
            return np.array([np.nan])

        # Correct for dust and host then return
        return self.model_extinction(modeled, params)

    def model_afterglow(self, params):
        """
        Models the unextinguished GRB afterglow flux.

        Parameters
        ----------
        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        Returns
        -------
        np.ndarray of float
            The modeled GRB afterglow flux.
        """
        return self.afg_model(
            **params.get('model'), **self.afg_kw).model(self.obs)

    def model_extinction(self, modeled, params):
        """
        Corrects the afterglow flux, ``modeled``, for
        dust extinction and host galaxy contributions.

        Applies the corrections in the order:
            1. Source-frame dust extinction.
            2. Host galaxy contribution.
            3. Milky Way dust extinction.

        Parameters
        ----------
        modeled : np.array
            The modeled flux.

        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        Returns
        -------
        np.ndarray of float
            The extinguished and host galaxy corrected flux.
        """
        if self.ext_model is None:
            return modeled

        pos = self.obs.extinguishable
        wn = self.obs.as_arrays.wave_numbers[pos]

        # Extinction params TEMP!
        z = params.get('model').get('z')
        ext = params.get('extinction')
        ebv_sf = ext.get('ebv_source_frame')
        ebv_mw = ext.get('ebv_milky_way')

        # Apply source-frame extinction
        if ebv_sf is not None:
            p = {'init': {'Rv': ext.get('rv_source_frame') or 3.1}, 'eval': {'Ebv': ebv_sf}}
            modeled[pos] *= self._model_extinction(p, (1 + z) * wn, self.ext_sf_pc)

        # Apply host galaxy correction
        if params.get('host') is not None and self.obs.hosts is not None:
            for name, corr in params.get('host').items():
                modeled[self.obs.hosts[name]] += corr

        # Apply Milky Way extinction
        if ebv_mw is not None:
            p = {'init': {'Rv': ext.get('rv_milky_way') or 3.1}, 'eval': {'Ebv': ebv_mw}}
            modeled[pos] *= self._model_extinction(p, wn, self.ext_mw_pc)

        # return corrected flux.
        return modeled

    def _model_extinction(self, p, wn, pc=None):
        """ Internal use only. """
        # Return the pre-computed extinction
        if pc is not None: return pc

        # Calculate the extinction and return
        return self.ext_model(
            **p.get('init')).extinguish(wn, **p.get('eval'))


# https://emcee.readthedocs.io/en/stable/tutorials/parallel/
# For multiprocessing purposes, emcee requires that methods and
# arguments be pickle-able. As such, the methods below are made
# global to meet this requirement.


def log_prior_fn(theta, params) -> float:
    """
    Evaluates the natural log of the priors.

    Parameters
    ----------
    theta : np.ndarray of float, with length of `fitting_params`
        The sampled MCMC parameter values.

    params : Parameters
        MCMC parameter container.

    Returns
    -------
    float
        The log of the evaluated priors.
    """
    lp = 0

    for i, p in enumerate(params.fitting):
        if np.isinf(prior := p.prior.evaluate(theta[i])):
            return -np.inf

        if prior != 0:
            lp += np.log(prior)

    return lp


def log_likelihood_fn(theta, params, models) -> float:
    """
    Calculates the natural log of the likelihood.

    Parameters
    ----------
    theta : np.ndarray of float
        The MCMC sampled values.

    params : Parameters
        The MCMC parameter container.

    models : MCMCModels
        The MCMC models container.

    Returns
    -------
    float or -np.inf
        The log of the likelihood if the parameters were valid.
        Else, -np.inf.
    """
    p = params.samples_to_dict(theta)

    # Model the observed afterglow
    modeled = models.model(p)

    # A nan always results in -inf likelihood.
    if np.isnan(modeled.min()):
        return -np.inf

    # Apply calibration offsets
    modeled = calibration_offsets(
        modeled, p.get('offsets'), models.obs.offsets
    )

    # Format the slop (if using)
    s = slop(p.get('slop').get('slop'), models.obs)

    # return log likelihood
    return -0.5 * chi_squared(modeled, models.obs, s)  # type: ignore


def log_posterior_fn(theta, params, models) -> float:
    """
    Calculates the natural log of the posterior
    probability.

    The posterior probability is the probability of
    the parameters, ``theta``, given the evidence X
    denoted by p(theta | X).

    Parameters
    ----------
    theta : np.ndarray of float
        The MCMC sampled values.

    params : Parameters
        The MCMC parameter container.

    models : MCMCModels
        The MCMC models container.

    Returns
    -------
    float
        The natural log of the posterior.
    """
    if np.isfinite(lp := log_prior_fn(theta, params)):
        ll = log_likelihood_fn(theta, params, models)

        if np.isfinite(ll):
            return lp + ll

    return -np.inf


def calibration_offsets(modeled, offsets, pos) -> np.ndarray:
    """
    Applies calibration offsets to the modeled values.

    Parameters
    ----------
    modeled : np.ndarray of float
        The modeled values.

    offsets : dict
        Key value pairs of ``CalGroup`` and offset values [mag].

    pos : dict
        The calibration positions.

    Returns
    -------
    np.ndarray
        The modeled values with applied offsets.
    """
    if offsets is not None:
        for name, offset in offsets.items():
            modeled[pos[name]] *= 10.0 ** -(0.4 * offset)
    return modeled


def slop(s, obs) -> float | np.ndarray | None:
    """
    Formats for slop to the modeled data.

    Parameters
    ----------
    s : float or dict
        The slop value or grouped slop values.

    obs : Observation
        The observational data.

    Returns
    -------
    float or np.ndarray or None
        The slop value(s).
    """
    if isinstance(s, (int, float)):
        return s

    elif isinstance(s, dict):
        res = np.empty(obs.length)

        for name, val in s.items():
            res[obs.slops[name]] = val

        return res


def chi_squared(modeled, obs, slops=None) -> float:
    """
    Calculates the combined chi-squared between
    the modeled and observational data for both
    the flux and spectral indices.

    The flux chi-squared calculation uses a so-
    called chi-squared effective which utilizes
    a slop parameter. Spectral indices use the
    standard chi-squared formulation.

    Parameters
    ----------
    modeled : np.ndarray of float
        The modeled or predicted values.

    obs : Observation
        The observational data.

    slops : float or np.ndarray of float, optional
        The slop value(s).

    Returns
    -------
    float
        The combined chi-squared value.
    """

    # Handle flux and indices the same
    if slops is None:
        return utils.chi_squared(
            modeled,
            obs.as_arrays.values,
            obs.as_arrays.errors,
        )

    # Chi-squared for flux (uses slop)
    flux_mask = obs.flux_loc

    cs_flux = utils.chi_squared(
        modeled[flux_mask],
        obs.as_arrays.values[flux_mask],
        obs.as_arrays.errors[flux_mask],
        slops if isinstance(slops, float) else slops[flux_mask]
    )

    # Chi-squared for spectral indices (does not use slop)
    index_mask = obs.sindex_loc

    if not index_mask.any():
        return cs_flux

    cs_indices = utils.chi_squared(
        modeled[index_mask],
        obs.as_arrays.values[index_mask],
        obs.as_arrays.errors[index_mask],
    )

    # return combined chi-squared
    return cs_flux + cs_indices
