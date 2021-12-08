## Approximating a Rosenbrock distribution

Here is a collection of scripts and notebooks for sampling from and approximating a Rosenbrock-like probability distribution. The files are:

1. `numpyro_nb.ipynb`: notebook for obtaining samples from the Rosenbrock distribution using MCMC (NUTS).
2. `covariance_training.py`: script for fitting a multivariate normal variational distribution using `jax`+`flax`+`optax`.
3. `nflow_training.py`: script for fitting a Real-NVP normalizing flow using `jax`+`flax`+`optax`.
4. `optimization.py`: script for performing a non-linear optimization and approximating a covariance matrix using the inverse Hessian of objective function.
5. `stats_summary.ipynb`: notebook for generating pair plots comparing samples from the different methods.
