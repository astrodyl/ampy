try:
    from matplotlib import pyplot as plt
except ImportError:
    raise ImportError(
        '`matplotlib` is required to run these tests.'
    )

import unittest
import numpy as np

from ampy.models.base import SpectralFluxModel


class TestSpectralFlux(unittest.TestCase):
    """
    Visualize the different spectral flux orderings to
    verify that the smoothing is working well.
    """

    @unittest.skip("Test=CAM, Reason=For visual inspection only")
    def test_CAM(self):
        """
        Visual inspection of the CAM smoothing approximation.
        """
        # Define the frequencies
        nu_c, nu_a, nu_m = 1e9, 1e11, 1e13
        nu = np.geomspace(1e8, 1e18, 500)

        # Define other
        f_peak, p, k = 1, 2.5, 0.0

        # Get smoothed flux
        model = SpectralFluxModel(nu_m, nu_c, f_peak, p, k, nu_a=nu_a)
        smoothed_flux = model(nu)

        # Get SPN98 spectral indices
        b1, b2, b3 = model.spectral_indices()

        # Determine the segments
        seg1 = nu < nu_a
        seg2 = np.logical_and(nu_a < nu, nu <= nu_m)
        seg3 = nu > nu_m

        # Discontinuity ratio
        rat = (1 / 3) * np.sqrt(nu_c / nu_a)

        # Calculate the sharply-broken flux
        flux = np.empty(nu.size)
        flux[seg1] = f_peak * (nu[seg1] / nu_a) ** 2
        flux[seg2] = f_peak * rat * (nu[seg2] / nu_a) ** -0.5
        flux[seg3] = f_peak * rat * (nu_m / nu_a) ** -0.5 * (nu[seg3] / nu_m) ** -(p / 2)

        # Annotation for each break
        plt.annotate(r'$\nu_m$', xy=(1.2 * nu_m, 1e-9), xytext=(1.2 * nu_m, 1e-9), fontsize=12)
        plt.annotate(r'$\nu_a$', xy=(1.2 * nu_a, 1e-9), xytext=(1.2 * nu_a, 1e-9), fontsize=12)

        # Plot the two
        plt.vlines(nu_a, ymin=0.0, ymax=f_peak * rat, color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_m, ymin=0.0, ymax=f_peak * rat * (nu_m / nu_a) ** b2, color='black', linestyle='--', alpha=0.6)
        plt.title(r'$\nu_c < \nu_a < \nu_m$', fontsize=18)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Flux [mJy]')
        plt.grid(alpha=0.5)
        plt.loglog(nu, smoothed_flux, label='Smoothly Broken')
        plt.loglog(nu, flux, label='Sharply Broken')
        plt.legend()
        plt.show()

    @unittest.skip("Test=MAC, Reason=For visual inspection only")
    def test_MAC(self):
        """
        Visual inspection of the MAC smoothing approximation.
        """
        # Define the frequencies
        nu_m, nu_a, nu_c = np.full(500, 1e9), np.full(500, 1e11), np.full(500, 1e13)
        nu = np.geomspace(1e8, 1e18, 500)

        # Define other
        f_peak, p, k = 1, 2.5, 0.0

        # Get smoothed flux
        model = SpectralFluxModel(nu_m, nu_c, f_peak, p, k, nu_a=nu_a)
        smoothed_flux = model(nu)

        # Determine the segments
        seg0 = nu <= nu_m
        seg1 = np.logical_and(nu > nu_m, nu <= nu_a)
        seg2 = np.logical_and(nu_a < nu, nu <= nu_c)
        seg3 = nu > nu_c

        # Calculate the sharply-broken flux
        flux = np.empty(nu.size)
        flux[seg0] = f_peak * (nu_m[seg0] / nu_a[seg0]) ** ((p + 4) / 2) * (nu[seg0] / nu_m[seg0]) ** 2
        flux[seg1] = f_peak * (nu_a[seg1] / nu_m[seg1]) ** ((1 - p) / 2) * (nu[seg1] / nu_a[seg1]) ** 2.5
        flux[seg2] = f_peak * (nu[seg2]   / nu_m[seg2]) ** ((1 - p) / 2)
        flux[seg3] = f_peak * (nu_c[seg3] / nu_m[seg3]) ** ((1 - p) / 2) * (nu[seg3] / nu_c[seg3]) ** (-p / 2)

        # Annotation for each break
        b1, b2, b3 = model.spectral_indices()
        plt.annotate(r'$\nu_m$', xy=(1.2 * nu_m[0], 1e-9), xytext=(1.2 * nu_m[0], 1e-9), fontsize=12)  # noqa
        plt.annotate(r'$\nu_a$', xy=(1.2 * nu_a[0], 1e-9), xytext=(1.2 * nu_a[0], 1e-9), fontsize=12)  # noqa
        plt.annotate(r'$\nu_c$', xy=(1.2 * nu_c[0], 1e-9), xytext=(1.2 * nu_c[0], 1e-9), fontsize=12)  # noqa

        # Plot the two
        plt.vlines(nu_m, ymin=0.0, ymax=f_peak * (nu_m / nu_a) ** ((p + 4) / 2), color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_a, ymin=0.0, ymax=f_peak * (nu_a / nu_m) ** b2, color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_c, ymin=0.0, ymax=f_peak * (nu_c / nu_m) ** b2, color='black', linestyle='--', alpha=0.6)
        plt.title(r'$\nu_m < \nu_a < \nu_c$', fontsize=18)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Flux [mJy]')
        plt.grid(alpha=0.5)
        plt.loglog(nu, smoothed_flux, label='Smoothly Broken')
        plt.loglog(nu, flux, label='Sharply Broken')
        plt.legend()
        plt.show()

    @unittest.skip("Test=AMC, Reason=For visual inspection only")
    def test_AMC(self):
        """
        Visual inspection of the AMC smoothing approximation.
        """
        # Define the frequencies
        nu_a = np.full(500, 6e9)
        nu_m = np.full(500, 2e11)
        nu_c = np.full(500, 5e12)
        nu = np.geomspace(1e8, 1e18, 500)

        # Define other
        f_peak, p, k = 2e4, 2.5, 0.0

        # Get smoothed flux
        model = SpectralFluxModel(nu_m, nu_c, f_peak, p, k, nu_a=nu_a)
        smoothed_flux = model(nu)

        # Get SPN98 flux
        b1, b2, b3 = model.spectral_indices()

        # Determine the segments
        seg_f = nu < nu_m
        seg_g = np.logical_and(nu_c > nu, nu > nu_m)
        seg_h = nu > nu_c

        # Calculate the sharply-broken flux
        flux = np.empty(nu.size)
        flux[seg_f] = f_peak * (nu[seg_f] / nu_m[seg_f]) ** b1[seg_f]
        flux[seg_g] = f_peak * (nu[seg_g] / nu_m[seg_g]) ** b2[seg_g]
        flux[seg_h] = f_peak * (nu_c[seg_h] / nu_m[seg_h]) ** b2[seg_h] * (nu[seg_h] / nu_c[seg_h]) ** b3[seg_h]

        # Plot the two
        plt.vlines(nu_m, ymin=0.0, ymax=f_peak, color='black', linestyle='--', alpha=0.6)
        plt.vlines(nu_c, ymin=0.0, ymax=f_peak * (nu_c / nu_m) ** b2, color='black', linestyle='--', alpha=0.6)
        plt.title('Slow Cooling Spectral Flux')
        plt.loglog(nu, smoothed_flux, label='Smoothly Broken')
        plt.loglog(nu, flux, label='Sharply Broken')
        plt.legend()
        plt.show()

    @unittest.skip("Test=FTS, Reason=For visual inspection only")
    def test_fts_smoothing(self):
        """"""
        # Define the frequencies
        nu_a, nu_m, nu_c = 6e9, 2e11, 5e12
        nu_m = np.geomspace(1e16, 1e10, 500)
        nu_c = np.geomspace(1e10, 1e16, 500)
        nu = np.geomspace(1e8, 1e18, 500)

        # Define other
        f_peak, p, k = 2e4, 2.5, 0.0

        # Get smoothed flux
        model = SpectralFluxModel(nu_m, nu_c, f_peak, p, k, nu_a=nu_a)

        b1, b2, b3 = model.spectral_indices()
        s12, s23 = model.smoothing()

        sb = s23
        nu_b = 1e13
        b2 = b1

        s_slow = 1.15 - (0.125 * k) - (0.06 - 0.015 * k) * p
        s_fast = 3.34 + 0.17 * k - (0.82 + 0.035 * k) * p
        s_t = np.where(nu_m < nu_c, s_slow, s_fast)
        q = s_t * (b3 - b1)
        s_eff = s_slow + (s_fast - s_slow) / (1 + (nu / nu_b) ** q)

        x = f_peak * (((nu / nu_b) ** -(sb * (b1 - b2)) + 1) * (nu / nu_b) ** -(sb * (b2 - b3)) + 1) ** -(1 / sb) * (nu / nu_b) ** b3
        y = f_peak * (((nu / nu_b) ** -(s_eff * (b1 - b2)) + 1) * (nu / nu_b) ** -(s_eff * (b2 - b3)) + 1) ** -(1 / s_eff) * (nu / nu_b) ** b3

        plt.axvline(nu_b, color='black', linestyle='--', alpha=0.4)
        plt.loglog(nu, x, label=r'$s_{23}$')
        plt.loglog(nu, y, label=r'$s_{eff}$')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
