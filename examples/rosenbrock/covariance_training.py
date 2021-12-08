#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# defaults
import argparse
from typing import Sequence
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

# jax, flax, optax
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.random as jrandom
import optax


def parse_cmd_line():
    parser = argparse.ArgumentParser(description='Run training.')
    parser.add_argument('-lr', type=float, action='store', default=0.0001,
                        help='Learning rate. Default: 0.0001.')
    parser.add_argument('-n_epochs', type=int, action='store', default=100000,
                        help='Number of training epochs. Default: 100000.')
    parser.add_argument('-restore', action='store_true', help='Restore previous parameters.')
    parser.add_argument('-seed', action='store', type=int, default=15,
                        help='Starting integer seed for PRNG. Default: 15.')
    parser.add_argument('-o', action='store', type=str, dest='outfile', default='params_cov.pkl',
                        help='Output pickle. Default: params_cov.pkl.')
    return parser.parse_args()


def seconds_to_mmss(seconds):
    mm, ss = divmod(seconds, 60)
    return '%4dm%5.3fs' % (mm, ss)


def eval_rosenbrock(x):
    """
    Use Rosenbrock as log-likelihood.
    """
    # Constants
    N_dim = x.size
    a = 1.0
    b = 5.0
    
    # Compute multi-dimensional Rosenbrock function
    f = 0.0
    for i in range(1, N_dim):
        df = b * (x[i] - x[i-1]**2)**2 + (a - x[i-1])**2
        f += df
    return -f

def loss_function_kl(params: optax.Params,
                     rng_key: jnp.ndarray):
    """
    Compute reverse-KL divergence between target and surrogate distributions. Here, both
    distributions are parameterized with mean and covariance matrices, which are used to
    sample and compute log probabilities.
    """
    # Construct covariance matrix and mean vector from parameters
    mu = jnp.array([params['mu1'], params['mu2'], params['mu3'], params['mu4']])
    cov = jnp.array([[params['s11'], params['s12'], params['s13'], params['s14']],
                     [params['s12'], params['s22'], params['s23'], params['s24']],
                     [params['s13'], params['s23'], params['s33'], params['s34']],
                     [params['s14'], params['s24'], params['s34'], params['s44']]])

    # Split the random key in order to get an over-writable subkey
    _, subkey = jrandom.split(rng_key)

    # Pre-generate samples from surrogate
    n_mc_samples = 5
    z_batched = jrandom.multivariate_normal(subkey, mu, cov, shape=(128, n_mc_samples))
   
    # Define Monte Carlo reverse-KL for a single example 
    def _compute_KL(z_mc):

        # Loop over MC samples
        reverse_KL = 0.0
        for i in range(n_mc_samples):

            # Get sample
            z = z_mc[i, :]

            # Evaluate target probabilities
            f_target = eval_rosenbrock(z)

            # Evaluate surrogate probabilitiy
            f_surrogate = jstats.multivariate_normal.logpdf(z, mu, cov)
    
        # Accumulate (negative) reverse-KL
        reverse_KL += f_surrogate - f_target

        return reverse_KL / n_mc_samples

    # Vectorize function over entire batch of samples
    reverse_KL_batched = jax.vmap(_compute_KL, in_axes=0, out_axes=0)(z_batched)
    return jnp.mean(reverse_KL_batched)
    

def fit(params: optax.Params,
        optimizer: optax.GradientTransformation,
        ref_key: jnp.ndarray,
        n_epochs=1000) -> optax.Params:

    # Initialize optimizer state
    opt_state = optimizer.init(params)

    # Split the reference random key (we'll overwrite keys during training)
    key, subkey = jrandom.split(ref_key)

    # Define function for computing gradient and updating parameters
    @jax.jit
    def step(params, opt_state, key):

        # Compute loss value and its gradient
        loss, grads = jax.value_and_grad(loss_function_kl)(params, key)
        
        # Get parameter updates
        updates, opt_state = optimizer.update(grads, opt_state, params)

        # Apply them
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    # Loop over epochs and batches in training set
    t0 = time.time()
    losses = np.zeros(n_epochs)
    s_losses = np.zeros(n_epochs)
    alpha = 0.1 # smoothing parameter for exponential smoothing of loss
    for i in range(n_epochs):

        # Update
        params, opt_state, losses[i] = step(params, opt_state, key)
        key, subkey = jrandom.split(key)

        # Compute exponentially-smoothed loss (purely for diagnostic purposes)
        if i == 0:
            smooth_loss = losses[i]
        else:
            smooth_loss = alpha * losses[i] + (1 - alpha) * smooth_loss
        s_losses[i] = smooth_loss

        # Periodically print diagnostics and training time
        if i % 1000 == 0:
            mmss = seconds_to_mmss(time.time() - t0)
            print('Epoch %04d / %04d, loss %10.4f, %s' % (i, n_epochs, smooth_loss, mmss))

    return params, losses, s_losses

def main(args):

    # Create initial random key
    ref_key = jrandom.PRNGKey(args.seed)

    # Optional restore of parameters
    if args.restore:
        with open('opt_params.pkl', 'rb') as fid:
            params = pickle.load(fid)

    # Otherwise, initialize pdf parameters (mean and covariance parameters)
    else:
        params = {'mu1': 0.0, 'mu2': 0.0, 'mu3': 0.0, 'mu4': 0.0,
                  's11': 1.0, 's22': 1.0, 's33': 1.0, 's44': 1.0,
                  's12': 0.0, 's13': 0.0, 's14': 0.0,
                  's23': 0.0, 's24': 0.0, 's34': 0.0}
    
    # Create optimizer
    optimizer = optax.adam(learning_rate=args.lr)

    # Run optimization
    params, losses, s_losses = fit(params, optimizer, ref_key, n_epochs=args.n_epochs)

    plt.plot(losses)
    plt.plot(s_losses)
    plt.show()

    # Go ahead and construct covariance matrix and mean vector from parameters
    mu = np.array([params['mu1'], params['mu2'], params['mu3'], params['mu4']])
    cov = np.array([[params['s11'], params['s12'], params['s13'], params['s14']],
                    [params['s12'], params['s22'], params['s23'], params['s24']],
                    [params['s13'], params['s23'], params['s33'], params['s34']],
                    [params['s14'], params['s24'], params['s34'], params['s44']]])

    # Save parameters
    with open(args.outfile, 'wb') as fid:
        pickle.dump(params, fid)
        pickle.dump(losses, fid)
        pickle.dump(mu, fid)
        pickle.dump(cov, fid)

    print(mu)
    print(cov)


if __name__ == '__main__':
    args = parse_cmd_line()
    main(args)

# end of file
