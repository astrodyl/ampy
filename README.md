# WARNING
I am actively developing this project. It is not considered to be 
ready yet. If you are reading this message, use at your own risk.

# Ampy
___
Afterglow modeling in Python (AMPy) provides a framework for modeling
gamma-ray burst (GRB) afterglows. Documentation can be found on my
GitBook: https://astrodyl.gitbook.io/astrodyl-docs/software-docs/ampy.
It is currently limited since AMPy is under active development.

# Installation
___
As with all projects, I highly recommend using a virtual environment.
After creating your `venv`, activate it, `cd` into the `ampy` directory,
and run:

```
pip install .
```

If you want to run the example scripts in `ampy/example/`, you will need
to install the optional dependencies by running:

```
pip install .[plot]
```
This will install `matplotlib`, `arviz`, and `corner`.

# Quick Start
___

See `run.py` in `ampy/example` for more details.

```Python
# Specify IO objects
parameters   = Parameters.from_toml(model_path)
observation  = Observation.from_csv(data_path)
mcmc_params  = utils.MCMCSettingsReader(mcmc_path)
sampler_name = mcmc_params.data['sampler']['name']

# Choose your model (FireballModel, JetSimpy, etc.)
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
```

There are some unit tests in the `ampy/test/` directory. These can be useful for
figuring out how to use certain objects.

# Optional Dependencies
___

## Parallel Tempering
AMPy supports parallel tempering. Since the `emcee` stopped supporting PT 
years ago, and since `ptemcee` also stopped receiving support, PT is
not required during installation. To use parallel tempering, install
[ptemcee]().
Note that Windows users may need to clone the repo, empty the `README` file,
and then `pip install` from source. There is a non-ASCII character in the
`README` file that causes issues on Windows machines.

## Models
AMPy has built-in support for the numerical model `jetsimpy` 
[[1]](https://ui.adsabs.harvard.edu/abs/2024ApJS..273...17W/abstract)
and the boosted fireball model
[[2]](https://ui.adsabs.harvard.edu/abs/2013ApJ...776L...9D/abstract)
used in `JetFit`
[[3]](https://ui.adsabs.harvard.edu/abs/2019PhDT........29W/abstract)
. Use of these models require that the user install additional packages.

### Using jetsimpy
First, you need to install `jetsimpy` by cloning the
[repo](https://github.com/haowang-astro/jetsimpy).
Follow their installation instructions. Then, using jetsimpy is as simple
as using the `JetSimpy` adapter in `ampy.mcmc.mcmc.JetSimpy`. If you would
like to model self-absorption (crudely), you can install my forked version:
[astrodyl/jetsimpy](https://github.com/astrodyl/jetsimpy/tree/develop)

### Using Boosted Fireball
I've currently written all the code required to use the boosted fireball model,
including numerous optimizations compared to `JetFit`. However, I currently do not
understand the tabulated hydrodynamic simulation results. Until I do, I'm' not
feel comfortable provided it as a built-in option.

### Afterglowpy
I plan on adding an 
`afterglowpy` [[4]](https://github.com/geoffryan/afterglowpy)
adapter soon.