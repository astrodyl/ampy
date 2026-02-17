import numpy as np
import ptemcee


class PTSampler:
    """
    Provides an API adapter that matches `emcee`.

    Parallel tempering in `emcee` stopped receiving support and was removed
    from official release. Using the latest version of `emcee` that supports
    the PTSampler forces users to use very old packages that the PTSampler
    depends on. There is a community developed version called `ptemcee`.
    However, the authors stopped maintaining it years ago.

    This class aims to provide an API that matches `emcee`. Methods are only
    added on an as-needed basis and are by no means complete.

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
        An object with a ``map`` method that follows the same calling sequence
        as emcee's built-in ``map`` function. This is used to compute the
        log-probabilities in parallel.

    kwargs
        Any kwargs accepted by `ptemcee.Sampler`.
    """
    name = 'tempered'

    def __init__(
        self, ntemps, nwalkers, ndim, log_like, log_prior,
        log_l_args=(), log_p_args=(), log_l_kwargs=(), log_p_kwargs=(),
        pool=None, **kwargs
    ):
        mapper = pool.map if pool is not None else map

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

    @property
    def acor(self):
        return self.get_autocorr_time()

    @property
    def acceptance_fraction(self):
        return self.chain.jump_acceptance_ratio[0]

    @property
    def swap_acceptance_fraction(self):
        return self.chain.swap_acceptance_ratio[0]

    @property
    def lnprobability(self):
        return self.get_log_prob()

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
            Any kwargs accepted by `ptemcee.Chain`.

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

        There's no reset method in `ptemcee` that I'm aware of. Overwriting
        with a new sampler is safer than attempting to reset attributes
        individually.
        """
        self._sampler = ptemcee.Sampler(
            self.sampler.nwalkers, self.sampler.ndim,
            self.sampler.logl, self.sampler.logp,
            self.sampler.logl_args, self.sampler.logp_args,
            self.sampler.logl_kwargs, self.sampler.logp_kwargs,
            ptemcee.make_ladder(self.ndim, self.ntemps),
            mapper=self.sampler._mapper  # noqa
        )
        self._chain = None
        self._iteration = 0

    def save(self, path):
        """
        Saves the chain and log posterior to `path`.

        To load the data, do: data = np.load(path).
        To access the chain, do: data['chain'].

        Parameters
        ----------
        path : str or pathlib.Path
            The path to save the file.
        """
        np.savez(
            file=path, chain=self.get_chain(), lnprob=self.get_log_prob(),
            betas=self.sampler.betas
        )

    def draw_positions(self, params, engine, log_post_fn):
        """
        Draw the initial positions from the priors.

        PTSampler requires that the start positions be valid. To meet this,
        any invalid position is overwritten with the best position for that
        temperature.

        Parameters
        ----------
        params : ``ampy.core.params.ParameterView``
            The MCMC parameters.

        engine : ``ampy.modeling.engine.ModelingEngine``
            The MCMC models.

        log_post_fn : func
            The log-posterior function.

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
                log_p[i, j] = log_post_fn(pos[i, j], engine, params)

            pos[i][np.isinf(log_p[i])] = np.array(
                pos[i][np.nanargmax(log_p[i])], copy=True
            )

        return pos

    def get_autocorr_time(self):
        """ Returns the autocorrelation time for the 0th temperature. """
        return self.chain.get_acts()[0]

    def get_last_sample(self):
        """ Returns last samples with shape [ntemps, nwalkers, ndim]. """
        if self.chain is None:
            raise AttributeError(
                'Tried to get the last sample, but there are no '
                'samples. Have you called `run_mcmc` yet?'
            )
        return self.chain.x[-1]

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
            Takes only every ``thin`` steps from the chain.

        discard : int, optional, default=0
            Discard the first ``discard`` steps in the chain as burn-in.

        temp : int, optional, default=0
            Takes only the `temp` attribute. Defaults to the temp at index
            `0` which is the highest probability temperature.

        Returns
        -------
        np.ndarray
        """
        if self.chain is None:
            raise AttributeError(
                f'Tried to get {name}, but there are no chains. Have you'
                f'called `run_mcmc` yet?'
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
                Take only every `thin` steps from the chain.

            discard : int, optional, default=0
                Discard the first ``discard`` steps in the chain as burn-in.

            temp : int, optional, default=0
                Take only the `temp` chain. Defaults to the temp at the
                `0`\th index which corresponds to the highest probability
                temperature.

        Returns
        -------
        np.ndarray with shape [..., nwalkers, ndim]
            The samples contained in ``ptemcee.Chain.x``.
        """
        return self.get_value('x', **kwargs)

    def get_log_prob(self, **kwargs):
        """
        Get the chain of log probabilities evaluated at the MCMC samples.

        Parameters
        ----------
        kwargs
            flat : bool, optional, default=False
                Flatten the chain across the ensemble.

            thin : int, optional, default=1
                Take only every `thin` steps from the chain.

            discard : int, optional, default=0
                Discard the first ``discard`` steps in the chain as burn-in.

            temp : int, optional, default=0
                Take only the ``temp`` log prob. Defaults to the temp at the
                `0`\th index which corresponds to the highest probability
                temperature.

        Returns
        -------
        np.ndarray with shape [..., nwalkers]
            The chain of log probabilities.
        """
        return self.get_value("logP", **kwargs)
