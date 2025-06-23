from ampy.mcmc.mcmc import MCMC

try:
    import corner
    import arviz as az
    from matplotlib import pyplot as plt
except ImportError:
    raise ImportError(
        "This example requires 'arviz', 'corner',"
        "and 'matplotlib' to be installed."
    )
from pathlib import Path


# Default plotting options for the corner plot. Modify as desired.
OPTIONS = {
    'label_size': 16, 'show_titles': True, 'color': 'mediumblue',
    'plot_datapoints': False, 'quantiles': [0.16, 0.5, 0.84],
    'label_kwargs': {'fontsize': 14}, 'title_kwargs': {"fontsize": 14},
    'fill_contours': True, 'smooth': 0.75, 'smooth1d': 0.75,
}


def latex(key: str) -> str:
    """
    Modify the dict to contain your model parameter names
    as keys if you want them to be converted to LaTeX.

    Parameters
    ----------
    key : str
        The parameter name

    Returns
    -------
    str
        The LaTeX formatted ``key`` or just ``key``
    """
    try:
        # Replace with your model parameters
        return {
            'E': r'$log_{10}E_{52}$',
            'eps_e': r'$log_{10}\epsilon_e$',
            'rv_milky_way': r'$log_{10}R_{v}^{MW}$',
            'rho0': r'$log_{10}\rho$',
        }[key]
    except KeyError:
        return key


def plot_trace(mcmc, out_dir) -> None:
    """
    Generate the trace plots using arviz.

    Parameters
    ----------
    mcmc : MCMC
        The finished MCMC object.

    out_dir : Path or str
        The directory to save the results.
    """
    # Use arviz style
    az.style.use("arviz-darkgrid")

    # Create the production inference data object
    var_names = [p.name for p in mcmc.params.fitting]
    inf_data = az.from_emcee(mcmc.sampler, var_names=var_names)

    # Save summary statistics to a csv
    az.summary(inf_data).to_csv(out_dir / "summary.csv")

    # Plot the trace plot
    az.plot_trace(inf_data)
    plt.savefig(out_dir / "trace.png")

    try:  # Optional stats
        print(f"Acceptance Fraction..{mcmc.sampler.acceptance_fraction}\n")
        print(f"Autocorrelation......{mcmc.sampler.acor}\n")
    except Exception as e:
        # ``emcee`` will throw an exception if they consider
        # the auto-correlation time to be too short, Don't
        # let your code crash after a long run because of this!
        print(e)


def plot_corner(chain, params, out_dir=None, kwargs=None):
    """
    Creates the corner plot of 1D and 2D posteriors.

    Parameters
    ----------
    chain :
        The flattened MCMC chain.

    params :
        The MCMC parameters.

    out_dir : Path or str
        The directory to save the results.

    kwargs : dict, optional, default=None
        Any kwargs to be passed to ``corner.corner``.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The ``matplotlib`` figure instance for the corner plot.
    """
    # The display ranges for each parameter. For this example,
    # we will use the lower upper prior bounds. For a Gaussian,
    # this is the 3-sigma limit.
    ranges = []

    # This will display the parameter names. Use the ``latex`
    # method to make them presentable.
    labels = []

    # Will use fixed number of bins, but there are algos to
    # determine binning from the data if desired.
    bins = [50 for _ in range(len(params))]

    # Store the positions of the parameters
    pos = []

    param_pos = {}
    for i, p in enumerate(params):
        param_pos[p.name] = i

    for p in params:
        ranges.append((p.prior.lower, p.prior.upper))
        labels.append(latex(p.name))
        pos.append(param_pos[p.name])

    # Overwrite any defaults with user provided values.
    options = OPTIONS | (kwargs or {})

    fig = corner.corner(
        data=chain[:, pos], bins=bins, labels=labels, range=ranges, **options
    )

    if out_dir:
        fig.savefig(r"output_path")

    # return the figure object
    return fig


if __name__ == "__main__":
    # Creates a corner plot with 2D and 1D posteriors.
    # https://emcee.readthedocs.io/en/stable/tutorials/line/

    # This is just a placeholder for the example.
    results = MCMC(**{}).run(**{})

    # Generate the corner plot
    figure = plot_corner(
        chain=results.get_chain.get(flat=True),
        params=results.params.fitting,
        out_dir=r"output_path"
    )

    # Generate the trace plot
    plot_trace(results, r"output_path")
