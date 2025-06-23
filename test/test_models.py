import unittest

import numpy as np
import astropy.units as u
import astropy.constants as const
from matplotlib import pyplot as plt

from ampy.models.base import PeakFluxModel, SynchrotronFrequencyModel
from ampy.models.base import CoolingFrequencyModel, AbsorptionFrequencyModel
from ampy.models.fireball import FireballModel, StratifiedFireballModel


# Constants in cgs units
m_p = const.m_p.cgs  # noqa
m_e = const.m_e.cgs  # noqa
q_e = u.Quantity(4.8032e-10 * u.g**0.5 * u.cm**1.5 / u.s)
c = const.c.cgs  # noqa


class TestSelfAbsorptionModel(unittest.TestCase):
    """"""

    def test_nu_a_acm(self):
        """
        Tests the self-absorption frequency in the fast-cooling
        regime with spectral ordering nu_a < nu_c < nu_m.

        True values are taken from Table 2.6 (ISM) and Table 2.7 (wind)
        in VDH (2007) [1]_.

        References
        ----------
        .. [1] Van Der Horst (2007): Broadband view of blast wave physics :
            a study of gamma-ray burst afterglows.
        """
        # ISM
        af_ism = AbsorptionFrequencyModel(
            E=1.0, rho0=1.0, eps_e=0.1, eps_b=0.1, k=0.0, z=0.0, X=0.7, p=2.2
        ).evaluate_acm(1.0) / (0.5 ** -0.5)

        # Wind
        af_wind = AbsorptionFrequencyModel(
            E=1.0, rho0=5e11 / m_p.value / 1e34, eps_e=0.1, eps_b=0.1, k=2.0, z=0.0, X=0.7, p=2.2
        ).evaluate_acm(1.0) / (0.5 ** 0.6)

        # True values (Tables 2.6, 2.7)
        af_ism_true = 1.25e9
        af_wind_true = 9.23e10

        # Assert equal within 1% (ISM) or 2% (WIND)
        self.assertAlmostEqual(af_ism / af_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(af_wind / af_wind_true, 1.0, delta=0.02)

    def test_nu_a_amc(self):
        """
        Tests the self-absorption frequency in the slow-cooling
        regime with spectral ordering nu_a < nu_m < nu_c.

        True values are taken from Table 2.6 (ISM) and Table 2.7 (wind)
        in VDH (2007) [1]_.

        References
        ----------
        .. [1] Van Der Horst (2007): Broadband view of blast wave physics :
            a study of gamma-ray burst afterglows.
        """
        # ISM
        af_ism = AbsorptionFrequencyModel(
            E=1.0, rho0=1.0, eps_e=0.1, eps_b=0.1, k=0.0, z=0.0, X=0.7, p=2.2
        ).evaluate_amc(1.0) / (0.5 ** -1)

        # Wind
        af_wind = AbsorptionFrequencyModel(
            E=1.0, rho0=5e11 / m_p.value / 1e34, eps_e=0.1, eps_b=0.1, k=2.0, z=0.0, X=0.7, p=2.2
        ).evaluate_amc(1.0) / (0.5 ** (-2 / 5))

        # True values (Tables 2.6, 2.7)
        af_ism_true = 7.75e10
        af_wind_true = 5.16e11

        # Assert equal within 1%
        self.assertAlmostEqual(af_ism / af_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(af_wind / af_wind_true, 1.0, delta=0.01)

    def test_nu_a_mac(self):
        """
        Tests the self-absorption frequency in the slow-cooling
        regime with spectral ordering nu_m < nu_a < nu_c.

        True values are taken from Table 2.6 (ISM) and Table 2.7 (wind)
        in VDH (2007) [1]_.

        References
        ----------
        .. [1] Van Der Horst (2007): Broadband view of blast wave physics :
            a study of gamma-ray burst afterglows.
        """
        # ISM
        af_ism = AbsorptionFrequencyModel(
            E=1.0, rho0=1.0, eps_e=0.1, eps_b=0.1, k=0.0, z=0.0, X=0.7, p=2.2
        ).evaluate_mac(1.0) / (0.5 ** -0.31)

        # Wind
        af_wind = AbsorptionFrequencyModel(
            E=1.0, rho0=5e11 / m_p.value / 1e34, eps_e=0.1, eps_b=0.1, k=2.0, z=0.0, X=0.7, p=2.2
        ).evaluate_mac(1.0) / (0.5 ** 0.016)

        # True values (Tables 2.6, 2.7)
        af_ism_true = 1.13e11
        af_wind_true = 4.38e11

        # Assert equal within 1%
        self.assertAlmostEqual(af_ism / af_ism_true, 1.0, delta=0.01)
        self.assertAlmostEqual(af_wind / af_wind_true, 1.0, delta=0.01)


class TestCharacteristicModels(unittest.TestCase):
    """"""

    @unittest.skip("Test=Density Smoothing, Reason=For visual inspection only")
    def test_density_smoothing(self):
        """ Plots the smoothed density and density power-laws. """

        model = StratifiedFireballModel(
            E=4.0, p=2.5, eps_b=0.001, eps_e=0.1, X=0.7,
            k1=2.0, k2=0.0, nt=30.0, rt=1e17, sn=3.0,
            dL=2.0, z=1.0
        )

        ts = np.geomspace(0.0012, 10, 500)  # 100s to 2 days
        radii = model.radii(ts)

        # Plot n effective
        for s in np.linspace(-3.0, 3.0, 5):
            model.sn = s
            n_eff, _ = model.smooth(ts)

            plt.loglog(radii, n_eff, label=f's = {s}')
            plt.axvline(model.rt, linestyle='--', color='black')
            plt.xlabel(r'Radius [cm]')
            plt.ylabel(r'$n_{eff}$')
            plt.legend(loc='best')
        plt.title('Number Density Normalization Smoothing (Normalized to nt)')
        plt.show()

        # Plot k effective
        for s in np.linspace(-3.0, 3.0, 5):
            model.sn = s
            _, k_eff = model.smooth(ts)

            plt.plot(radii, k_eff, label=f's = {s}')
            plt.axvline(model.rt, linestyle='--', color='black')
            plt.xlabel(r'Radius [cm]')
            plt.ylabel(r'$k_{eff}$')
            plt.legend(loc='best')
            plt.xscale('log')
        plt.title('Power-Law Index Smoothing')
        plt.show()

    def test_vdh_ism(self):
        """
        Test that the general k-model reduces to the ISM (k=0) case
        when k is set to 0.

        True values are taken from Table 2.6 in VDH (2007) [1]_.

        References
        ----------
        .. [1] Van Der Horst (2007): Broadband view of blast wave physics :
            a study of gamma-ray burst afterglows.
        """
        model = FireballModel(
            E=1.0, rho0=1.0, p=2.2, k=0.0, z=0.0, dL=1.0,
            eps_e=0.1, eps_b=0.1, X=0.7
        )

        # Modeled values
        f_peak = model.f_peak(1.0)
        nu_m = model.nu_m(1.0)
        nu_c = model.nu_c(1.0)
        nu_a = model.nu_a(1.0, nu_m, nu_c)

        # True values
        f_peak_true = 21.3 * 0.5
        nu_m_true = 8.98e11 * (0.5**0.5)
        nu_c_true = 5.98e13 * (0.5**-0.5)
        nu_a_slow_true = 7.75e10 * (0.5**-1)

        # Assert equal within 1%
        self.assertAlmostEqual(f_peak / f_peak_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_m / nu_m_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_c / nu_c_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_a / nu_a_slow_true, 1.0, delta=0.01)

    def test_vdh_wind(self):
        """
        Test that the general k-model reduces to the wind
        (k=2) case when k is set to 2.

        True values are taken from Table 2.7 in VDH (2007) [1]_.

        References
        ----------
        .. [1] Van Der Horst (2007): Broadband view of blast wave physics :
            a study of gamma-ray burst afterglows.
        """

        # Correct for the different normalizations. We use a
        # number density referenced to 1e17cm.
        n0 = 5e11 / 1.67e-24 / 1e34

        model = FireballModel(
            E=1.0, rho0=n0, p=2.2, k=2.0, z=0.0, dL=1.0,
            eps_e=0.1, eps_b=0.1, X=0.7
        )

        # Modeled values
        f_peak = model.f_peak(1.0)
        nu_m = model.nu_m(1.0)
        nu_c = model.nu_c(1.0)
        nu_a = model.nu_a(1.0, nu_m, nu_c)

        # True values
        f_peak_true = 60.8 * (0.5**1.5)
        nu_m_true = 1.85e12 * (0.5 ** 0.5)
        nu_c_true = 9.97e11 * (0.5 ** -1.5)
        nu_amc_true = 5.16e11 * (0.5 ** -0.4)

        # Assert equal within 1%
        self.assertAlmostEqual(f_peak / f_peak_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_m / nu_m_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_c / nu_c_true, 1.0, delta=0.01)
        self.assertAlmostEqual(nu_a / nu_amc_true, 1.0, delta=0.01)

    def test_spn_ism(self):
        """
        Test that the spectral values match Sari, Piran, & Narayan 1998.
        """
        model = FireballModel(
            E=1.0, rho0=1.0, p=2.5, k=0.0, z=0.0, dL=1.0,
            eps_e=1, eps_b=1, X=1.0
        )

        # Modeled values
        nu_c = model.nu_c(1.0)
        nu_m = model.nu_m(1.0)

        # Derivation of the peak flux differs by 8pi/9
        f_peak = model.f_peak(1.0) * 8.0 * np.pi / 9

        # True values
        nu_c_true = 2.7e12
        nu_m_true = 5.7e14
        f_peak_true = 1.1e2

        # Assert equal within 5%
        self.assertAlmostEqual(f_peak / f_peak_true, 1.0, delta=0.05)
        self.assertAlmostEqual(nu_m / nu_m_true, 1.0, delta=0.05)
        self.assertAlmostEqual(nu_c / nu_c_true, 1.0, delta=0.05)


if __name__ == '__main__':
    unittest.main()
