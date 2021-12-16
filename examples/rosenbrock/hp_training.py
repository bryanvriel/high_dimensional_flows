#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# defaults
import argparse
from typing import Sequence
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import time
from functools import partial

# jax, flax, optax
import jax
import jax.numpy as jnp
from jax import lax
import jax.scipy.stats as jstats
import jax.random as jrandom
import flax.linen as nn
import optax

# Custom functions
from models import Stacked_RNVP
from covariance_training import seconds_to_mmss, eval_rosenbrock

# Number of dimensions for Rosenbrock function
N_DIM = 4

def parse_cmd_line():
    parser = argparse.ArgumentParser(description='Run training.')
    parser.add_argument('-lr', type=float, action='store', default=0.0002,
                        help='Learning rate. Default: 0.0002.')
    parser.add_argument('-n_epochs', type=int, action='store', default=2000,
                        help='Number of training epochs. Default: 2000.')
    parser.add_argument('-restore', action='store_true', help='Restore previous parameters.')
    parser.add_argument('-seed', action='store', type=int, default=10,
                        help='Starting integer seed for PRNG. Default: 10.')
    parser.add_argument('-o', action='store', type=str, dest='outfile', default='params_nflow.pkl',
                        help='Output pickle. Default: params_nflow.pkl.')
    return parser.parse_args()


def gradient(f):
    """Creates a function to grad of a function f.
    Args:
        f: Callable with signature f(params, data).
    Returns:
        Callable with signature grad(params, data).
    """
    return jax.grad(f, argnums=1)


def divergence(f):
    """Creates a function to divergence of a vector-valued function f.
    Args:
        f: Callable with signature f(params, data).

    Returns:
        Callable with signature div(params, data).
    """
    def _div_over_f(params, x):
        n = x.shape[0]
        eye = jnp.eye(n)
        f_closure = lambda y: f(params, y)

        def _body_fun(i, val):
            primal, tangent = jax.jvp(f_closure, (x,), (eye[i],))
            return val + tangent[i]

        return lax.fori_loop(0, n, _body_fun, 0.0)

    return _div_over_f


def laplacian(f):
    """Creates a function to Laplacian of f.
    Args:
        f: Callable with signature f(params, data).

    Returns:
        Callable with signature lapl(params, data).
    """
    return divergence(gradient(f))


def loss_function_kl(params: optax.Params,
                     rng_key: jnp.ndarray):
    """
    Compute reverse-KL divergence between target and surrogate distributions.
    """
    # Instantiate model
    model = Stacked_RNVP(40, 4)

    # Split the random key in order to get an over-writable subkey
    _, subkey = jrandom.split(rng_key)

    # Pre-generate a batch of samples from base distribution
    n_mc_samples = 1
    base_mu = jnp.zeros(N_DIM)
    base_cov = jnp.eye(N_DIM)
    z_batched = jrandom.multivariate_normal(subkey, base_mu, base_cov, shape=(32, n_mc_samples))

    # Define Monte Carlo reverse-KL for a single example
    def _compute_loss(z_mc):            
        # Inverse transform
        z = z_mc[1, :]
        x = model.apply(params, z, method=Stacked_RNVP.inverse)

        # Fokker-Planck operator assembly
        # Drift and diffusion functions specific to problem under
        # consideration
        sigma = 1.0
        def potential(params, x_in):
            return -eval_rosenbrock(x_in)
        def drift(params, x_in):
            return -jax.grad(potential, argnums=1)(params, x_in)

        # Generic components for assembly of Hp
        def log_pdf(params, x_in):
            "evaluate log pdf"
            return model.log_pdf(params, x_in)
        # def grad_log_pdf(params, x_in):
        #     "evaluate gradient of pdf"
        #     return jax.grad(log_pdf, argnums=1)(params, x_in)
        # def jvp_drift(vec, x_in):
        #     "evaluate jacobian-vector product of drift vector field"
        #     return jax.grad(lambda w: jnp.vdot(drift(w), vec))(x_in)
        # def jvp_diff_flux(vec, x_in):
        #     "evaluate jacobian-vector product of diffusive flux vector field"
        #     return jax.grad(lambda w: jnp.vdot(jnp.dot(diff(w),grad_log_pdf(w)), vec))(x_in)
        # def div_drift(x_in):
        #     "evaluate divergence of drift vector field"
        #     vecs = jnp.eye(N_DIM)
        #     div = 0.0
        #     for i in range(N_DIM):
        #         v = vecs[:,i]
        #         div += jnp.vdot(jvp_drift(v, x_in), v)
        #     return div
        # def div_diff_flux(x_in):
        #     "evaluate divergence of diffusive flux vector field"
        #     vecs = jnp.eye(N_DIM)
        #     div = 0.0
        #     for i in range(N_DIM):
        #         v = vecs[:,i]
        #         div += jnp.vdot(jvp_diff_flux(v, x_in), v)
        #     return div

        # compute Hp loss function
        grad_log_pdf = gradient(log_pdf)
        hp = -divergence(drift)(params, x) - jnp.vdot(drift(params, x) - sigma*grad_log_pdf(params, x), grad_log_pdf(params, x)) + sigma*laplacian(log_pdf)(params, x)
        hp_loss = hp**2

        return hp_loss



    # Vectorize function over entire batch of samples
    loss_batched = jax.vmap(_compute_loss, in_axes=0, out_axes=0)(z_batched)
    return jnp.mean(loss_batched)
 

def fit(params: optax.Params,
        optimizer: optax.GradientTransformation,
        ref_key: jnp.ndarray,
        n_epochs=1000) -> optax.Params:
    """
    Main entrypoint for fitting a variational distribution.
    """
    # Initialize optimizer state
    opt_state = optimizer.init(params)

    # Split the reference random key (we'll overwrite keys during training)
    key, subkey = jrandom.split(ref_key)

    # Define function for computing gradient and updating parameters
    @jax.jit
    def step(params, opt_state, key):

        # Compute loss value and its gradient
        loss, grads = jax.value_and_grad(loss_function_kl)(
            params, key
        )
        
        # Get parameter updates
        updates, opt_state = optimizer.update(grads, opt_state, params)

        # Apply them
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    # Loop over epochs and batches in training set
    t0 = time.time()
    losses = np.zeros(n_epochs)
    s_losses = np.zeros(n_epochs)
    alpha = 0.2 # smoothing parameter for exponential smoothing of loss
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
        if i % 100 == 0:
            mmss = seconds_to_mmss(time.time() - t0)
            print('Epoch %04d / %04d, loss %10.4f, %s' % (i, n_epochs, smooth_loss, mmss))
        
    return params, losses, s_losses

def main(args):

    # Create initial random key
    ref_key = jrandom.PRNGKey(args.seed)

    # Optional restore of parameters
    if args.restore:
        with open('params_nflow.pkl', 'rb') as fid:
            params = pickle.load(fid)
    # Otherwise, initialize parameters with test example with the right shape
    else:
        init_key, xkey = jrandom.split(ref_key)
        x = jrandom.normal(xkey, (N_DIM,))
        params = Stacked_RNVP(40, 4).init(init_key, x)

    # Create optimizer
    optimizer = optax.adam(learning_rate=args.lr)

    # Run optimization
    params, losses, s_losses = fit(params, optimizer, ref_key, n_epochs=args.n_epochs)

    # Save parameters
    with open(args.outfile, 'wb') as fid:
        pickle.dump(params, fid)
        pickle.dump(losses, fid)
   
    plt.plot(losses)
    plt.plot(s_losses)
    plt.show() 

    # Draw samples from base distribution
    base_mu = np.zeros(N_DIM)
    base_cov = np.eye(N_DIM)
    z = stats.multivariate_normal.rvs(base_mu, base_cov, size=2000)

    # Pass through flow
    @jax.jit
    def generate_samples(z_batched):
        def _sample(z):
            x = Stacked_RNVP(40, 4).apply(params, z, method=Stacked_RNVP.inverse)
            return x
        x_batched = jax.vmap(_sample, in_axes=0, out_axes=0)(z_batched)
        return x_batched

    x_post = generate_samples(z)

    # Save samples
    with h5py.File('samples_hp_nflow.h5', 'w') as fid:
        fid['samples'] = x_post


if __name__ == '__main__':
    args = parse_cmd_line()
    main(args)

# end of file
