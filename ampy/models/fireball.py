import numpy as np

from ampy.core.input import Observation
from ampy.models.base import BlastWaveModel, ObservedSpectrumModel
from ampy.models.base import AbsorptionFrequencyModel, BaseFireballModel
from ampy.models.base import SynchrotronFrequencyModel
from ampy.models.base import CoolingFrequencyModel, PeakFluxModel


class StratifiedFireballModel(BaseFireballModel):
    """
    A fully analytic description of an ultra-relativistic
    shock moving into an external stratified medium with
    density rho = rho_0 * R^-k. Where k transitions from
    one asymptotic value to another.

    Parameters
    ----------
    E : float
        The explosion energy normalized to 1e52 ergs.

    p : float
        The electron energy index (dimensionless).

    eps_b : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons. Must be in the range [0, 1].

    z : float
        The redshift to the event.

    dL : float
        The luminosity distance to the event [1e28 cm]. Requiring
        the distance to be provided in addition to the redshift
        prevents the need to assume a cosmology here.

    sn, sni : float, optional
        The density smoothing factor. ``sni`` is the inverse of the
        smoothing factor. Useful for changing fitting basis in MCMC.

    k1 : float
        The density power-law index before the transition.

    k2 : float
        The density power-law index after the transition.

    X : float
        The hydrogen mass fraction.

    tj : float, optional, default=None
        The jet break observer-frame time [d].

    sj, sji : float, optional, default=None
        The jet break smoothing factor. `sji` is the inverse of the
        smoothing factor. Useful for changing fitting basis in MCMC.
        Required if ``tj != None``.

    use_sa : bool, optional, default=True
        Should self-absorption be modeled?
    """
    # noinspection PyPep8Naming
    def __init__(
            self, E, p, eps_b, eps_e, z, dL, nt, rt, X,
            k1=None, k2=None, tj=None, sj=None, sji=None, sn=None, sni=None, k1i=None, k2i=None, use_sa=True
    ):
        super().__init__(E, p, eps_b, eps_e, z, dL, X, tj, sj, sji, use_sa)

        if sn is None and sni is None:
            raise ValueError("Must specify either sn or sni.")

        # Medium
        self.k1 = k1 if k1 is not None else 1 / k1i
        self.k2 = k2 if k2 is not None else 1 / k2i

        self.rt = rt
        self.nt = nt
        self.sn = (sn or 1 / sni) if (sn or sni) else None

    @property
    def is_valid(self) -> bool:
        """ Whether the model is parameters are valid. """
        if self.sn < 0 and self.k1 < self.k2:
            return False

        if self.sn > 0 and self.k1 > self.k2:
            return False

        return super().is_valid and abs(self.sn) > 0.1

    def smooth(self, t):
        """
        Empirically smooths the number density normalizations
        and the power-law indices over the observer times ``t``.

        Parameters
        ----------
        t : np.ndarray
            The observer times [d].

        Returns
        -------
        tuple of np.ndarray of float
            The smoothed number density normalizations [cm-3] and
            the smoothed density power-law indices.
        """
        # Rename for convenience
        s, k1, k2 = self.sn, self.k1, self.k2

        r = self.radii(t)
        x = r / self.rt

        # Calculate the effective number densities
        n = self.nt * (2 ** (1 / s)) * (
            x ** (k1 * s) + x ** (k2 * s)
        ) ** -(1 / s)

        # Calculate the effective density power-law indices
        k_eff_num = k1 * x ** (k1 * s) + k2 * x ** (k2 * s)
        k_eff_den = x ** (k1 * s) + x ** (k2 * s)
        k_eff = k_eff_num / k_eff_den

        # Number density normalized to `rt`
        n0 = n * (r / self.rt) ** k_eff

        return n0, k_eff

    def radii(self, t):
        """
        Calculates the radius traversed by the blast wave
        during time ``t`` in a stratified medium defined by
        the power-law indices ``k1`` and ``k2``, and the radius
        and density at the transition, ``nt`` and ``rt``.

        Parameters
        ----------
        t : float or np.ndarray
            The observer times [days].

        Returns
        -------
        float or np.ndarray
            The radii traversed by the blast wave [cm].
        """
        bwm1 = BlastWaveModel(self.E, self.nt, self.k1, ref=self.rt)
        bwm2 = BlastWaveModel(self.E, self.nt, self.k2, ref=self.rt)
        t_decel = bwm1.decel_time() / 86_400

        r1 = bwm1.shock_radius(self.z, t, t_decel)
        r2 = bwm2.shock_radius(self.z, t, t_decel)

        # Rename for convenience
        s, k1, k2 = self.sn, self.k1, self.k2
        x1, x2 = r1 / self.rt, r2 / self.rt

        return (2 ** (1 / s)) * self.rt * (x1 ** -s + x2 ** -s) ** -(1 / s)

    def model(self, obs: Observation):
        """
        Models the observational data, ``obs``.

        Parameters
        ----------
        obs : Observation
            The observational data.

        Returns
        -------
        np.ndarray of float
            The modeled observational data.
        """
        if not self.is_valid:
            return np.array([np.nan])

        # return the modeled smoothed, unextinguished flux
        return ObservedSpectrumModel(**self.spectrum(obs.times()),
            arrays=obs.as_arrays, jet=self.jet_break(obs.times()),
        ).model()

    def spectrum(self, t, n=None, k=None):
        """
        Returns the characteristics that define a GRB spectrum.

        Parameters
        ----------
        t : np.ndarray of float or float
            The observer time [d].

        n : np.ndarray of float or float, optional
            The effective density normalization [cm-3].

        k : np.ndarray of float or float, optional
            The effective power law indices.

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        if n is None or k is None:
            n, k = self.smooth(t)

        return {
            'p': self.p, 'k': k,
            'f_peak': self.f_peak(t, n, k),
            'nu_m': (nu_m := self.nu_m(t, k)),
            'nu_c': (nu_c := self.nu_c(t, n, k)),
            'nu_a': self.nu_a(t, n, k, nu_m, nu_c) if self.use_sa else None,
        }

    def f_peak(self, t, n=None, k=None):
        """
        Calculates the peak flux in the case of an ultra-
        relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        n : float or np.ndarray of float, optional
            The smoothed density normalization [cm-3].

        k : float or np.ndarray of float, optional
            The density power-law indices.

        Returns
        -------
        float or np.ndarray of float
            The peak flux [mJy] at time `t` [d].
        """
        if k is None or n is None:
            n, k = self.smooth(t)

        return PeakFluxModel(
            self.E, n, self.eps_b, self.dL, self.z, k, self.X)(t, np.log10(self.rt))

    def nu_c(self, t, n=None, k=None):
        """
        Calculates the cooling frequency in the case of an ultra-
        relativistic shock moving into an external medium with
        density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        n : np.ndarray of float, optional
            The smoothed density normalization [cm-3].

        k : np.ndarray of float, optional
            The density power-law indices.

        Returns
        -------
        float or np.ndarray of float
            The cooling frequency in Hz at time `t`.
        """
        if n is None or k is None:
            n, k = self.smooth(t)

        return CoolingFrequencyModel(
            self.E, n, self.eps_b, k, self.z)(t, np.log10(self.rt))

    def nu_m(self, t, k=None):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        k : np.ndarray of float, optional
            The density power-law indices.

        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The synchrotron frequency [Hz] at time `t`.
        """
        if k is None:
            _, k = self.smooth(t)

        return SynchrotronFrequencyModel(
            self.E, self.eps_e, self.eps_b, k, self.z, self.X, self.p)(t)

    def nu_a(self, t, n=None, k=None, nu_m=None, nu_c=None):
        """
        Calculates the self-absorption frequency.

        The self-absorption frequency has a circular definition.
        For example, to calculate nu_a, you must first know how
        nu_a relates to the nu_m and nu_c, but nu_a isn't known
        because it needs to be known before it can be known >:)

        This solution is weak, but the self-absorption frequency
        is calculated for every case (except nu_a > both nu_c and
        nu_m, not supported). The result is a combined array where
        the self-absorptions are compared to the synchrotron and
        cooling frequencies.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        n : np.ndarray of float, optional
            The effective density normalization [cm-3].

        k : np.ndarray of float, optional
            The effective density power-law indices.

        nu_m : float or np.ndarray of float, optional
            The synchrotron frequencies [Hz] at time ``t``.

        nu_c : float or np.ndarray of float, optional
            The cooling frequencies [Hz] at time ``t``.

        Returns
        -------
        float or np.ndarray of float
            The self-absorption frequency [Hz] at time(s) ``t``.
        """
        if n is None or k is None:
            n, k = self.smooth(t)

        model = AbsorptionFrequencyModel(
            self.E, n, self.eps_e, self.eps_b, k, self.z, self.X, self.p
        )

        nu_m = self.nu_m(t, k) if nu_m is None else nu_m
        nu_c = self.nu_c(t, n, k) if nu_c is None else nu_c
        fast = nu_c < nu_m

        # Evaluate nu_a for all orderings
        ref = np.log10(self.rt)
        nu_amc = model.evaluate_amc(t, ref)
        nu_mac = model.evaluate_mac(t, ref)
        nu_cam = model.evaluate_cam(t, ref)
        nu_acm = model.evaluate_acm(t, ref)

        # Initialize with slow cooling values
        res = np.where(nu_amc < nu_m, nu_amc, nu_mac)

        if fast.any():  # Overwrite with fast cooling values
            res[fast] = np.where(nu_acm < nu_c, nu_acm, nu_cam)[fast]

        return res


class FireballModel(BaseFireballModel):
    """
    A fully analytic description of an ultra-relativistic
    shock moving into an external medium with density
    rho = rho_0 * R^-k.

    Parameters
    ----------
    E : float
        The explosion energy normalized to 1e52 ergs.

    p : float
        The electron energy index.

    eps_b : float
        The fraction of thermal energy in the magnetic field.
        Must be in the range [0, 1].

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons. Must be in the range [0, 1].

    z : float
        The redshift of the event.

    dL : float
        The luminosity distance to the event [1e28 cm]. Requiring
        the distance to be provided in addition to the redshift
        prevents the need to assume a cosmology here.

    rho0 : float
        The density normalization.

    k : float
        The density power-law index.

    X : float
        The hydrogen mass fraction.

    tj : float, optional, default=None
        The jet break observer-frame time [d].

    sj : float, optional, default=None
        The jet break smoothing factor. Required if ``tj != None``.

    use_sa : bool, optional, default=True
        Should self-absorption be modeled?
    """
    # noinspection PyPep8Naming
    def __init__(self, E, p, eps_b, eps_e, z, dL, rho0, k, X, tj=None, sj=None, sji=None, use_sa=True):
        super().__init__(E, p, eps_b, eps_e, z, dL, X, tj, sj, sji, use_sa)

        self.rho0 = rho0
        self.k = k

    def model(self, obs: Observation, subset: np.ndarray = None):
        """
        Models the observational data, ``obs``.

        Parameters
        ----------
        obs : Observation
            The observational data.

        subset : np.ndarray of bool, optional
            Models data where ``subset==True``.

        Returns
        -------
        np.ndarray of float
            The modeled observational data.
        """
        if not self.is_valid:
            return np.array([np.nan])

        # return the modeled observational data
        return ObservedSpectrumModel(**self.spectrum(obs.times()),
            jet=self.jet_break(obs.times()), arrays=obs.as_arrays
        ).model(subset)

    def spectrum(self, t):
        """
        Returns the characteristics that define the GRB spectrum.

        Parameters
        ----------
        t : np.ndarray of float or float
            The observer time [d].

        Returns
        -------
        dict
            keys: f_peak, nu_a, nu_m, nu_c, p, k.
        """
        return {
            'p': self.p, 'k': self.k,
            'f_peak': self.f_peak(t),
            'nu_m': (nu_m := self.nu_m(t)),
            'nu_c': (nu_c := self.nu_c(t)),
            'nu_a': self.nu_a(t, nu_m, nu_c) if self.use_sa else None,
        }

    def f_peak(self, t):
        """
        Calculates the peak flux in the case of an ultra-
        relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The peak flux value(s) [mJy] at time(s) ``t``.
        """
        return PeakFluxModel(
            self.E, self.rho0, self.eps_b, self.dL, self.z, self.k, self.X)(t)

    def nu_c(self, t):
        """
        Calculates the cooling frequency in the case of an ultra-
        relativistic shock moving into an external medium with
        density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The cooling frequency value(s) [Hz] at time(s) ``t``.
        """
        return CoolingFrequencyModel(
            self.E, self.rho0, self.eps_b, self.k, self.z)(t)

    def nu_m(self, t):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        Returns
        -------
        float or np.ndarray of float
            The synchrotron frequency value(s) [Hz] at time(s) ``t``.
        """
        return SynchrotronFrequencyModel(
            self.E, self.eps_e, self.eps_b, self.k, self.z, self.X, self.p)(t)

    def nu_a(self, t, nu_m=None, nu_c=None):
        """
        Calculates the self-absorption frequency.

        The self-absorption frequency has a circular definition.
        For example, to calculate nu_a, you must first know how
        nu_a relates to the nu_m and nu_c, but nu_a isn't known
        because it needs to be known before it can be known >:)

        This solution is weak, but the self-absorption frequency
        is calculated for every case (except nu_a > both nu_c and
        nu_m, not supported). The result is a combined array where
        the self-absorptions are compared to the synchrotron and
        cooling frequencies.

        Parameters
        ----------
        t : float or np.ndarray of float
            The observer time(s) [d].

        nu_m : float or np.ndarray of float, optional
            The synchrotron frequencies [Hz] at time `t`.

        nu_c : float or np.ndarray of float, optional
            The cooling frequencies [Hz] at time `t`.

        Returns
        -------
        float or np.ndarray of float
            The self-absorption frequency value(s) [Hz] at time(s) ``t``.
        """
        model = AbsorptionFrequencyModel(
            self.E, self.rho0, self.eps_e, self.eps_b,
            self.k, self.z, self.X, self.p
        )

        nu_m = self.nu_m(t) if nu_m is None else nu_m
        nu_c = self.nu_c(t) if nu_c is None else nu_c
        fast = nu_c < nu_m

        # Evaluate nu_a for all orderings
        nu_amc = model.evaluate_amc(t)
        nu_mac = model.evaluate_mac(t)
        nu_cam = model.evaluate_cam(t)
        nu_acm = model.evaluate_acm(t)

        # Initialize with slow cooling values
        res = np.where(nu_amc < nu_m, nu_amc, nu_mac)

        # Overwrite with fast cooling values
        res[fast] = np.where(nu_acm < nu_c, nu_acm, nu_cam)[fast]

        return res
