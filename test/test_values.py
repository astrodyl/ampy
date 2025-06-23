import unittest
from collections import namedtuple

import astropy.units as u
from astropy.units import UnitTypeError

from ampy.core.structs import Bound, FluxBase, SpectralFlux


class TestBound(unittest.TestCase):
    """
    Tests the Bound class.
    """
    def test_init(self):
        """"""
        low, upp = 1.0 * u.m, 2.0 * u.m
        b = Bound(lower=low, upper=upp)

        self.assertEqual(b.lower, low)
        self.assertEqual(b.upper, upp)
        self.assertEqual(b.value, (low, upp))

    def test_enforce_unit_type(self):
        """ Tests that the unit enforcements work. """
        low, upp = 1.0 * u.m, 200.0 * u.cm

        # lower, upper have diff but compatible
        # units. No exception should be raised.
        b = Bound(lower=low, upper=upp)

        # lower, upper have incompatible units
        with self.assertRaises(UnitTypeError):
            Bound(lower=1.0 * u.m, upper=2.0 * u.Hz)

        # set lower to non-length unit
        with self.assertRaises(UnitTypeError):
            b.lower = 1.01 * u.Hz

        # set upper to non-length unit
        with self.assertRaises(UnitTypeError):
            b.upper = 2.01 * u.mJy

        # ensure that values remain unchanged after
        # raising exceptions
        self.assertEqual(b.lower, low)
        self.assertEqual(b.upper, upp)

    def test_encompass(self):
        """ Tests the encompass method. """
        low, upp = 1.0 * u.m, 2.0 * u.m
        b = Bound(lower=low, upper=upp)

        # ensure inclusive works
        self.assertTrue(
            b.encompasses(1.0 * u.m, inclusive=True),
            msg="Failed comparison when inclusive=True"
        )
        self.assertFalse(
            b.encompasses(1.0 * u.m, inclusive=False),
            msg="Failed comparison when inclusive=False"
        )

        # ensure equivalent units work
        self.assertTrue(
            b.encompasses(150 * u.cm),
            msg="Failed equivalent unit comparison"
        )

        # ensure unit assumption works
        self.assertTrue(
            b.encompasses(1.5),
            msg="Failed unit assumption"
        )


class TestFluxBase(unittest.TestCase):
    """
    Tests the FluxBase class.
    """
    def test_time(self):
        """ Tests that the time methods work. """
        flux = 1.0 * u.mJy
        low = 0.10 * u.mJy
        upp = 0.15 * u.mJy
        time = 2.9 * u.s

        # time has proper units, should work
        fb = FluxBase(flux, low, upp, time)
        self.assertEqual(fb.time, time)

        # time has incompatible units, should raise exception
        with self.assertRaises(u.errors.UnitsError):
            fb.time = 1.0 * u.m

    def test_bounds(self):
        """ Tests that bounds setter enforces units. """
        flux = 1.0 * u.mJy
        low = 0.10 * u.mJy
        upp = 0.15 * u.mJy
        t = 1.0 * u.day

        fb = FluxBase(flux, low, upp, t)

        # units are compatible, should work
        fb.uncertainty = (0.11 * u.mJy, 0.22 * u.mJy)
        self.assertIsInstance(fb.uncertainty, Bound)

        # bound has incompatible units, should raise exception
        with self.assertRaises(u.UnitTypeError):
            fb.uncertainty = (1.0 * u.m, 2.0 * u.mJy)


class TestSpectralFlux(unittest.TestCase):
    """
    Tests the SpectralFlux class.
    """
    def test_conversions(self):
        """"""
        value = 1.0 * u.mJy
        lower = 0.10 * u.mJy
        upper = 0.15 * u.mJy
        frequency = 664 * u.THz
        time = 1.0 * u.day

        # Create with frequency
        sf = SpectralFlux(
            value=value,
            lower=lower,
            upper=upper,
            frequency=frequency,
            time=time
        )

        # Test that class calculated wavelength properly,
        # Defaults to m when calculating from frequency.
        wavelength = frequency.to(unit=u.m, equivalencies=u.spectral())

        self.assertEqual(sf.frequency, frequency)
        self.assertEqual(sf.wavelength, wavelength)

        # Test that class calculated frequency properly,
        # Defaults to Hz when calculating from wavelength.
        f = wavelength.to(unit=u.Hz, equivalencies=u.spectral())

        sf = SpectralFlux(
            value=value,
            lower=lower,
            upper=upper,
            wavelength=wavelength,
            time=time
        )

        self.assertEqual(sf.wavelength, wavelength)
        self.assertEqual(sf.frequency, f)

    def test_from_csv(self):
        """"""
        Row = namedtuple(
            typename='Row',
            field_names=[
                'Value', 'ValueUnits', 'ValueLower',
                'ValueUpper', 'Time', 'TimeUnits',
                'Wave', 'WaveUnits', 'Index'
            ]
        )

        row = Row(100.1, 'mJy', 7.3, 8.0, 34400.1, 's', 612, 'THz', 1)

        sf = SpectralFlux.from_csv_row(row)
        self.assertEqual(sf.value.value, row.Value)
        self.assertEqual(sf.value.unit, row.ValueUnits)
        self.assertEqual(sf.uncertainty.lower.value, row.ValueLower)
        self.assertEqual(sf.uncertainty.upper.value, row.ValueUpper)
        self.assertEqual(sf.time.value, row.Time)
        self.assertEqual(sf.time.unit, row.TimeUnits)
        self.assertEqual(sf.frequency.value, row.Wave)
        self.assertEqual(sf.frequency.unit, row.WaveUnits)


if __name__ == '__main__':
    unittest.main()
