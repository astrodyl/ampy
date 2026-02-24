import unittest
from pathlib import Path

from ampy.ampy import AMPy


class TestRun(unittest.TestCase):

    def setUp(self):
        current_file_path = Path(__file__).resolve()
        rsc_dir = current_file_path.parent / "resources"

        self.obs = rsc_dir / "synthetic.csv"
        self.registry = rsc_dir / "configs" / "registry.toml"

    def test_basic_ampy_ensemble_run(self):
        """ Runs a basic ampy ensemble inference. """
        # Create AMPy instance from run configuration
        ampy = AMPy.from_registry(self.obs, self.registry)

        # Run MCMC and get the most likely parameters
        best = ampy.run_mcmc(nwalkers=100, iterations=10, burn=10)

        self.assertIsNotNone(best)


if __name__ == '__main__':
    unittest.main()
