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

# jax, flax, optax
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.random as jrandom
import flax.linen as nn
import optax

# Custom functions
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


class MLP(nn.Module):
    """
    Standard feedforward neural network.
    """
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):

        # Iterate over features and apply activation after each layer
        for feat in self.features[:-1]:
            x = nn.tanh(nn.Dense(feat)(x))

        # Final layer without activation
        x = nn.Dense(self.features[-1])(x)
        return x


class R_NVP(nn.Module):
    """
    Single Real-NVP block.
    """

    hidden: int

    def setup(self):
        self.d = N_DIM
        self.k = self.d // 2
        self.sig_net = MLP([self.hidden, self.hidden, self.d - self.k])
        self.mu_net = MLP([self.hidden, self.hidden, self.d - self.k])

    def __call__(self, x, flip):
        """
        The forward transform x -> z, i.e. target -> base.
        """
        # Split the input
        x1 = x[:self.k]
        x2 = x[self.k:]

        # Optional flip
        if flip:
            x2, x1 = x1, x2

        # Forward
        sig = self.sig_net(x1)
        z1 = x1
        z2 = x2 * jnp.exp(sig) + self.mu_net(x1)
        z_hat = jnp.hstack((z1.squeeze(), z2.squeeze()))

        # Log-determinant of Jacobian
        jac_logdet = jnp.sum(sig)

        return z_hat, jac_logdet

    def inverse(self, Z, flip):
        """
        Inverse transform z -> x, i.e. base -> target.
        """
        # Split input
        z1 = Z[:self.k]
        z2 = Z[self.k:]

        # Inverse (including flip)
        x1 = z1
        x2 = (z2 - self.mu_net(z1)) * jnp.exp(-self.sig_net(z1))
        if flip:
            x2, x1 = x1, x2

        return jnp.hstack((x1.squeeze(), x2.squeeze()))


class Stacked_RNVP(nn.Module):
    """
    Flow of stacked Real-NVP blocks.
    """
    hidden: int
    n_flows: int

    def setup(self):
        # Must use list comprehension here for creating lists if self.setup()
        self.bijectors = [R_NVP(self.hidden) for _ in range(self.n_flows)]
        self.flips = [True if i % 2 else False for i in range(self.n_flows)]
            
    def __call__(self, X):
        """
        The forward transform x -> z, i.e. target -> base.
        """
        # The base distribution MVN arrays
        base_mu = jnp.zeros(N_DIM)
        base_cov = jnp.eye(N_DIM)

        # Loop over bijectors
        z = X
        log_jacobs = 0
        for bijector, flip in zip(self.bijectors, self.flips):

            # Forward pass through bijector
            z, logdet = bijector(z, flip)

            # Compute log-likelihood
            log_pz = jstats.multivariate_normal.logpdf(z, base_mu, base_cov)

            # Accumulate Jacobian log-determinant
            log_jacobs += logdet

        return z, log_pz, log_jacobs

    def inverse(self, Z):
        """
        Inverse transform z -> x, i.e. base -> target.
        """
        # Loop over bijectors and flips in reverse directions
        x = Z
        for bijector, flip in zip(reversed(self.bijectors), reversed(self.flips)):
            x = bijector.inverse(x, flip)
        return x


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
    n_mc_samples = 5
    base_mu = jnp.zeros(N_DIM)
    base_cov = jnp.eye(N_DIM)
    z_batched = jrandom.multivariate_normal(subkey, base_mu, base_cov, shape=(128, n_mc_samples))

    # Define Monte Carlo reverse-KL for a single example
    def _compute_KL(z_mc):

        # Loop over MC samples
        reverse_KL = 0.0
        for i in range(n_mc_samples):
            
            # Inverse transform
            z = z_mc[i, :]
            x = model.apply(params, z, method=Stacked_RNVP.inverse)

            # Evaluate target probabilities
            f_target = eval_rosenbrock(x)

            # Evaluate surrogate log prob
            _, log_pz, log_jacobs = model.apply(params, x)
            f_surrogate = log_pz + log_jacobs

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
    base_mu = np.zeros(3)
    base_cov = np.eye(3)
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
    with h5py.File('samples_nflow.h5', 'w') as fid:
        fid['samples'] = x_post


if __name__ == '__main__':
    args = parse_cmd_line()
    main(args)

# end of file
