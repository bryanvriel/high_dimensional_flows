
# defaults
from typing import Sequence
import scipy.stats as stats
import numpy as np
from functools import partial

# jax
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import flax.linen as nn

# Global parameters
N_DIM = 4

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
        Inverse pass through chain of bijectors to transform z -> x, i.e. base -> target.
        """
        # Loop over bijectors and flips in reverse directions
        x = Z
        for bijector, flip in zip(reversed(self.bijectors), reversed(self.flips)):
            x = bijector.inverse(x, flip)
        return x

    def log_pdf(self, params, X):
        """
        Probability distribution function.
        """
        _, log_pz, log_jacobs = self.apply(params, X)
        return log_pz + log_jacobs


# end of file