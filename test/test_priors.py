import unittest

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from ampy.core.structs import Prior
from ampy.mcmc.parameters import priors
from ampy.mcmc.priors import GaussianPrior, UniformPrior, TruncatedGaussianPrior, MilkyWayRvPrior


class TestPriorFactory(unittest.TestCase):
    """"""
    def test_prior_factory(self):
        """ Tests that the factory returns the correct prior object. """

        gaussian = priors.prior_factory({
            'type': 'gaussian',
            'mu': 5.0,
            'sigma': 1.0,
        })
        self.assertEqual(gaussian.type, Prior.GAUSSIAN)

        t_gaussian = priors.prior_factory({
            'type': 'tgaussian',
            'mu': 5.0,
            'sigma': 1.0,
            'lower': 4.5,
            'upper': 7.0,
        })
        self.assertEqual(t_gaussian.type, Prior.TGAUSSIAN)

        uniform = priors.prior_factory({
            'type': 'uniform',
            'lower': 4.5,
            'upper': 7.0,
        })
        self.assertEqual(uniform.type, Prior.UNIFORM)

        sine = priors.prior_factory({
            'type': 'sine',
            'lower': 4.5,
            'upper': 7.0,
        })
        self.assertEqual(sine.type, Prior.SINE)


class TestUniformPrior(unittest.TestCase):
    """
    Tests the `ampy.mcmc.priors.UniformPrior` class.
    """
    def test_instantiation(self):
        """"""
        lower, upper, guess, sigma = -5, 5, 3, 1
        uniform = priors.UniformPrior(lower, upper, guess, sigma)

        self.assertEqual(uniform.type, Prior.UNIFORM)
        self.assertEqual(uniform.lower, lower)
        self.assertEqual(uniform.upper, upper)
        self.assertEqual(uniform.initial_guess, guess)
        self.assertEqual(uniform.initial_sigma, sigma)

        uniform2 = UniformPrior.from_dict({
            'lower': lower,
            'upper': upper,
            'initial_guess': guess,
            'initial_sigma': sigma,
        })

        self.assertEqual(uniform2.type, uniform.type)
        self.assertEqual(uniform2.lower, uniform.lower)
        self.assertEqual(uniform2.upper, uniform.upper)
        self.assertEqual(uniform2.initial_guess, uniform.initial_guess)
        self.assertEqual(uniform2.initial_sigma, uniform.initial_sigma)

    def test_draw(self):
        """
        Tests that the draw method samples within the appropriate bounds.
        """
        lower, upper, guess, sigma = -5, 5, 3, 1

        # Create a distribution centered on zero, with an initial guess
        uniform = priors.UniformPrior(lower, upper, guess, sigma)

        samples_g = uniform.draw(1_000_000)
        self.assertEqual(len(samples_g), 1_000_000)
        self.assertAlmostEqual(guess, np.mean(samples_g), 1)

        # Assert that the samples are within the guess region
        self.assertTrue(samples_g.max() <= guess + sigma)
        self.assertTrue(samples_g.min() >= guess - sigma)

        # Sample from the full region, ignoring the guess
        center = (lower + upper) / 2

        samples_f = uniform.draw(1_000_000, False)
        self.assertEqual(len(samples_f), 1_000_000)
        self.assertAlmostEqual(center, np.mean(samples_f), 0)

        # Assert that the samples are within the full region
        self.assertTrue(samples_f.max() <= upper)
        self.assertTrue(samples_f.min() >= lower)

        # Assert that the samples are beyond the guess region
        self.assertTrue(samples_f.max() > guess + sigma)
        self.assertTrue(samples_f.min() < guess - sigma)

    def test_evaluate(self):
        """
        Tests that the evaluate method computes the correct value.
        """
        lower, upper, guess, sigma = -5, 5, 3, 1
        uniform = priors.UniformPrior(lower, upper, guess, sigma)

        for x in (-5.0, -5, 0, 5, 5.0):
            self.assertEqual(uniform.evaluate(x), 0.0)

        for x in (-10.0, -10, -5.1, 5.1, 10, 10.0):
            self.assertEqual(uniform.evaluate(x), -np.inf)

    @unittest.skip("Test=Plot Uniform, Reason=For visual inspection only")
    def test_plot_distribution(self):
        """
        Plots the results of `UniformPrior.draw` and `UniformPrior.evaluate()`.
        """
        lower, upper, guess, sigma = -5, 5, 3, 1

        # Create a distribution centered on zero, with an initial guess
        uniform = priors.UniformPrior(lower, upper, guess, sigma)

        n = 10_000
        samples_g = uniform.draw(n)
        samples_f = uniform.draw(n, initial=False)

        fig, ax = plt.subplots(1, 1)
        ax.hist(samples_f, density=True, bins='auto', facecolor='#2ab0ff', edgecolor='#169acf', alpha=0.5, label='Full Samples')
        _, _, rects = ax.hist(samples_g, density=True, bins='auto', alpha=0.5, label='Initial Samples')

        # Normalize the heights of the histograms
        h = (abs(lower) + abs(upper)) / (sigma * 2)

        for r in rects:
            r.set_height(r.get_height() / h)

        # Plot vertical lines to compare theoretical and sampled mu, sigma
        ax.vlines(guess + sigma, 0, 0.15, color='black', alpha=0.5, linestyles='dashed', label=r'True Initial Region')
        ax.vlines(guess - sigma, 0, 0.15, color='black', alpha=0.5, linestyles='dashed')

        ax.set_title(str(uniform) + f' with n={n} samples')
        ax.set_ylim(0, 0.15)
        ax.legend(loc='best')
        plt.show()


class TestGaussianPrior(unittest.TestCase):
    """
    Tests the `ampy.mcmc.priors.GaussianPrior` class.
    """
    def test_instantiation(self):
        """
        Tests that the instantiation methods all create an equivalent object.
        """
        mu, sigma = 3.123, 1.456
        gaussian = priors.GaussianPrior(mu, sigma)

        self.assertEqual(gaussian.type, Prior.GAUSSIAN)
        self.assertEqual(gaussian.mu, mu)
        self.assertEqual(gaussian.sigma, sigma)

        gaussian2 = GaussianPrior.from_dict({
            'mu': mu,
            'sigma': sigma,
        })

        self.assertEqual(gaussian2.type, gaussian.type)
        self.assertEqual(gaussian2.mu, gaussian.mu)
        self.assertEqual(gaussian2.sigma, gaussian.sigma)

    def test_draw(self):
        """
        Tests that the draw method draws samples from the correct
        distribution with the correct deviation.
        """
        mu, sigma, n = 3.123, 1.456, 1_000_000

        gaussian = GaussianPrior(mu, sigma)
        samples = gaussian.draw(n)

        self.assertEqual(len(samples), n)
        self.assertAlmostEqual(mu, np.mean(samples), 2)
        self.assertAlmostEqual(sigma, np.std(samples), 2)

    def test_evaluate(self):
        """ Tests that the evaluate method computes the correct value. """
        mu, sigma, x = 3.0, 1.0, 3.0

        self.assertEqual(
            GaussianPrior(mu, sigma).evaluate(x),
            stats.norm.pdf(x, loc=mu, scale=sigma)
        )

    @unittest.skip("Test=Plot Gaussian, Reason=For visual inspection only")
    def test_plot_distribution(self):
        """
        Plots the results of `GaussianPrior.draw` and `GaussianPrior.evaluate()`.
        """
        mu, sigma, n = 3.0, 1.0, 10_000
        fig, ax = plt.subplots(1, 1)

        # Create the x-axis, cutting off the gaussian tail
        x = np.linspace(
            stats.norm.ppf(0.01, loc=mu, scale=sigma),
            stats.norm.ppf(0.99, loc=mu, scale=sigma),
            1000
        )

        # Draw and evaluate the gaussian
        gaussian = GaussianPrior(mu, sigma)
        samples = gaussian.draw(n)
        pdf = gaussian.evaluate(x)

        # Plot the probability distribution function
        ax.plot(x, pdf, 'r-', lw=5, alpha=0.6, label='Sampled PDF')

        # Plot the histogram of samples for comparison
        ax.hist(samples, density=True, bins='auto', facecolor='#2ab0ff', edgecolor='#169acf', alpha=0.2)

        # Calculate the mu, sigma from the drawn samples
        sampled_mu = np.mean(samples)
        sampled_sigma = np.std(samples)

        # Plot vertical lines to compare theoretical and sampled mu, sigma
        ax.vlines(sampled_mu + sampled_sigma, 0, 0.5, color='black', alpha=0.5, linestyles='dashed', label=r'Sampled 1-$\sigma$')
        ax.vlines(sampled_mu - sampled_sigma, 0, 0.5, color='black', alpha=0.5, linestyles='dashed')

        ax.vlines(mu + sigma, 0, 0.5, color='green', alpha=0.5, linestyles='dotted', label=r'True 1-$\sigma$')
        ax.vlines(mu - sigma, 0, 0.5, color='green', alpha=0.5, linestyles='dotted')

        # Prettify the plot
        ax.set_title(str(gaussian) + f' with n={n}')
        ax.set_xlim([x[0], x[-1]])
        ax.legend(loc='best')
        plt.show()


class TestTGaussianPrior(unittest.TestCase):
    """
    Tests the `ampy.mcmc.priors.TruncatedGaussianPrior` class.
    """
    @unittest.skip("Test=Plot TGaussian, Reason=For visual inspection only")
    def test_plot_distribution(self):
        """
        Plots the results of `TruncatedGaussianPrior.draw` and
        `TruncatedGaussianPrior.evaluate()`.
        """
        n = 10_000
        mu, sigma, lower, upper = 5, 0.3, 4, 5.5
        gaussian = TruncatedGaussianPrior(mu, sigma, lower, upper)

        fig, ax = plt.subplots(1, 1)
        x = np.linspace(lower - 1, upper + 1, n)

        # Plot the probability distribution function
        pdf = np.asarray([gaussian.evaluate(xx) for xx in x])
        ax.plot(x, pdf, 'r-', lw=5, alpha=0.6, label='Sampled PDF')

        # Plot the histogram of samples for comparison
        samples = gaussian.draw(n)
        ax.hist(samples, density=True, bins='auto', histtype='stepfilled', alpha=0.2)

        ax.set_title(str(gaussian) + f' with n={n}')
        ax.legend(loc='best')
        plt.show()


class TestMilkyWayRvPrior(unittest.TestCase):
    """"""

    @unittest.skip("Test=Plot MilkyWayRv, Reason=For visual inspection only")
    def test_plot_distribution(self):
        """"""
        n = 100_000
        rv_prior = MilkyWayRvPrior()

        fig, ax = plt.subplots(1, 1)
        log_rv = np.linspace(0.35, 1.0, n)

        # Plot the probability distribution function
        pdf = np.asarray([rv_prior.evaluate(xx, True) for xx in log_rv])
        ax.plot(log_rv, pdf, 'r-', lw=5, alpha=0.6, label='Sampled PDF')

        # Plot the histogram of samples for comparison
        samples = rv_prior.draw(n)
        ax.hist(samples, density=True, bins='auto', facecolor='#2ab0ff', edgecolor='#169acf', alpha=0.4)

        ax.set_title(str(rv_prior) + f' with n={n}')
        ax.set_ylabel('$p(logR_{v}^{MW})$')
        ax.set_xlabel(r'$logR_{v}^{MW}$')
        ax.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    unittest.main()
