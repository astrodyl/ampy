from pathlib import Path

from ampy.core import utils
from ampy.core.input import Observation
from ampy.mcmc.mcmc import MCMC, MCMCModels
from ampy.mcmc.parameters import Parameters
from ampy.models.fireball import FireballModel


def main(mcmc_path, model_path, data_path) -> MCMC:
    """
    Bare minimum starting routine.

    Note that this main file will eventually be
    contained in an `Ampy` class. For now, it's
    an example run script.

    Parameters
    ----------
    mcmc_path : Path
        The path to the MCMC settings file.

    model_path : Path
        The path to the model parameter file.

    data_path : Path
        The path to the input file.

    Returns
    -------
    `ampy.mcmc.mcmc.MCMC`
        The completed MCMC object.
    """
    # Parse the inputs
    parameters   = Parameters.from_toml(model_path)
    observation  = Observation.from_csv(data_path)
    mcmc_params  = utils.MCMCSettingsReader(mcmc_path)
    sampler_name = mcmc_params.data['sampler']['name']

    # Choose your model
    model = FireballModel

    # Create the MCMC object
    model_wrapper = MCMCModels(observation, model)
    mcmc = MCMC(model_wrapper, parameters)

    # Run MCMC. This may take a while..
    mcmc.run(
        nwalkers=mcmc_params.num_walkers,
        iterations=mcmc_params.run_length,
        burn=mcmc_params.burn_length,
        sampler=sampler_name,
        workers=mcmc_params.workers,
        ntemps=mcmc_params.ntemps,
    )

    # Return the mcmc object
    return mcmc


if __name__ == "__main__":
    # See `example` directory for things you can do. This run
    # script is only a bare minimum getting started script.

    # To run, specify the IO paths below
    results = main(
        **{
            'mcmc_path':
                Path(r""),

            'model_path':
                Path(r""),

            'data_path':
                Path(r""),
        }
    )

    # The 'best' solution
    best = results.get_best_params()

    # The MCMC sampler
    sampler = results.sampler

    # The observational data
    obs = results.observation
