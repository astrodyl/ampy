import emcee
import numpy as np


class EnsembleSampler(emcee.EnsembleSampler):
    """
    Adapter for `emcee.EnsembleSampler`.
    """
    name = 'ensemble'

    def __init__(self, nwalkers, ndim, log_prob_fn, args, **kw):
        super().__init__(nwalkers, ndim, log_prob_fn, args=args, **kw)

    @classmethod
    def load(cls, nwalkers, ndim, log_prob_fn, pool, args=None, **kw):
        """
        Instantiate the EnsembleSampler.

        Parameters
        ----------
        nwalkers : int
            The number of walkers in the ensemble.

        ndim : int
            Number of dimensions in the parameter space.

        log_prob_fn : callable
            A function that takes a vector in the parameter space as input and
            returns the natural logarithm of the posterior probability (up to
            an additive constant) for that position.

        pool : `ProcessPoolExecutor` or `ThreadPoolExecutor` or `nullcontext`
            The pool to use for multithreading/processing.

        args : iterable, optional
            A list of extra positional arguments for
            `log_prob_fn`. `log_prob_fn` will be called with the sequence
            `log_prob_fn(p, *args, **kwargs)`.

        kw : dict, optional
            Extra keyword arguments for `log_prob_fn`. `log_prob_fn` will be
            called with the sequence `log_prob_fn(p, *args, **kwargs)`.

        Returns
        -------
        ampy.inference.samplers.ensemble.EnsembleSampler
            The loaded ensemble sampler.
        """
        # Set the backend if not already set by the user
        if kw.get('backend') is None:
            if kw.get('path') is not None:
                kw['backend'] = emcee.backends.HDFBackend(str(kw['path']))

        # If the user is not resuming from a previous run, reset the backend
        if kw.get('backend') is not None and kw.get('resume', False):
            kw['backend'].reset(nwalkers, ndim)

        return cls(nwalkers, ndim, log_prob_fn, args, pool=pool, **kw)

    def save(self, path):
        """
        Saves the chain and log posterior to `path`.

        It is highly recommended to use `emcee.backends.HDFBackend` instead
        of this method.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to save the file.
        """
        np.savez(
            file=path, chain=self.get_chain(), lnprob=self.get_log_prob()
        )

    def draw_positions(self, params, **kwargs) -> np.ndarray:
        """
        Draw the initial positions from the priors.

        Parameters
        ----------
        params : Parameters
            The MCMC parameters.

        kwargs :
            For compatability with `PTSampler.draw_positions`.

        Returns
        -------
        np.ndarray of float with shape [nwalkers, ndim]
            The starting positions.
        """
        pos = np.zeros((self.nwalkers, self.ndim))

        for i, p in enumerate(params.fitting):
            pos[:, i] = p.prior.draw(self.nwalkers)

        return pos
