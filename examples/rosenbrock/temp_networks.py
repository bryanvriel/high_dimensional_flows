#-*- coding: utf-8 -*-

from hdflow import *

"""
For normalizing flows, the convention is:

    - Forward pass f(x)     : x -> z (i.e., the 'encoder')
    - Inverse pass f^{-1}(z): z -> x (i.e., the 'decoder')

"""

class R_NVP(tf.keras.layers.Layer):

    def __init__(self, N, hidden, N_hidden, name='rnvp'):
        """
        Build single R-NVP block.

        Parameters
        ----------
        N: int
            Dimensionality of inputs.
        hidden: int
            Size of hidden layer.
        N_hidden: int
            Number of hidden layers.
        name: str, optional
            Model name. Default: 'rnvp'

        Returns
        -------
        None
        """
        # Initialize parent class
        super().__init__(name=name)

        # Layer parameters
        self.N = N
        self.M = N // 2

        # The activation functions and initialization
        init = 'lecun_normal'
        self.sig_actfun = tf.tanh
        self.mu_actfun = tf.tanh

        # Create dense layers
        layers = [hidden] * N_hidden + [self.N - self.M,]
        self.sigma = DenseNetwork(layers, initializer=init, name='%s_sigma' % name)
        self.mu = DenseNetwork(layers, initializer=init, name='%s_mu' % name)

        return

    def call(self, x, c, flip=False, training=False):
        """
        Forward pass f(x, c): x -> z

        where c is a conditioning input.

        Parameters
        ----------
        x: (Nb, N) tensor
            Nb-batch of input tensors to be transformed.
        c: (Nb, Nc) tensor
            Nb-batch of conditioning inputs of dimension Nc.
        flip: bool, optional
            Flip the split inputs. Default: False.
        training: bool, optional
            Training pass. Default: False.

        Returns
        -------
        z_hat: (Nb, N) tensor
            Nb-batch of transformed tensors.
        jac_logdet: (Nb,) tensor
            Nb-batch of Jacobian log-determinants.
        """
        # Split the input
        x1 = x[:, :self.M]
        x2 = x[:, self.M:]

        # Optional flip
        if flip:
            x2, x1 = x1, x2

        # Concatenate inputs with conditioning variables
        A1 = tf.concat(values=[x1, c], axis=1)

        # Predict scaling and translation
        sigma = self.sigma(A1, activation=self.sig_actfun, training=training,
                           activate_outputs=False)
        shift = self.mu(A1, activation=self.mu_actfun, training=training,
                        activate_outputs=False)

        # Forward pass
        z1 = x1
        z2 = x2 * tf.exp(sigma) + shift
        z_hat = tf.concat(values=[z1, z2], axis=1)

        # Log-determinant of Jacobian
        jac_logdet = tf.reduce_sum(sigma, axis=1)

        return z_hat, jac_logdet

    def inverse(self, z, c, flip=False, training=False):
        """
        Inverse pass f^{-1}(z): z -> x

        where c is a conditioning input.

        Parameters
        ----------
        z: (Nb, N) tensor
            Nb-batch of input tensors to be transformed.
        c: (Nb, Nc) tensor
            Nb-batch of conditioning inputs of dimension Nc.
        flip: bool, optional
            Flip the split inputs. Default: False.
        training: bool, optional
            Training pass. Default: False.

        Returns
        -------
        x: (Nb, N) tensor
            Nb-batch of transformed tensors.
        """
        # Split input
        z1 = z[:, :self.M]
        z2 = z[:, self.M:]

        # Concatenate inputs with conditioning variables
        A1 = tf.concat(values=[z1, c], axis=1)

        # Predict scaling and translation
        sigma = self.sigma(A1, activation=self.sig_actfun, training=training,
                           activate_outputs=False)
        shift = self.mu(A1, activation=self.mu_actfun, training=training,
                        activate_outputs=False)

        # Inverse
        x1 = z1
        x2 = (z2 - shift) * tf.exp(-sigma)

        # Optional flip
        if flip:
            x2, x1 = x1, x2

        return tf.concat(values=[x1, x2], axis=1)


class Stacked_RNVP(tf.keras.Model):

    def __init__(self, N, hidden, N_hidden, n_step, base_dist, name='stacked'):
        """
        Stack multiple R-NVP blocks.

        Parameters
        ----------
        N: int
            Dimensionality of inputs.
        hidden: int
            Size of hidden layer.
        N_hidden: int
            Number of hidden layers.
        n_step: int
            Number of R-NVP blocks.
        base_dist: tfd.Distribution
            Base distribution.
        name: str, optional
            Model name. Default: 'stacked'

        Returns
        -------
        None
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create chain of bijectors
        self.n_step = n_step
        self.bijectors = []
        for i in range(n_step):
            name = 'nvp_%03d' % i
            self.bijectors.append(R_NVP(N, hidden, N_hidden, name=name))

        # Save base distribution
        self._base_dist = base_dist

        # Save distribution dimensionality
        self.N = N

        # Create list of flips for alternating coupling layers
        self.flips = [True if i % 2 else False for i in range(n_step)]

    def call(self, x, c, training=False):
        """
        Forward pass through chain of bijectors, along with conditioning
        inputs c.

        f(x; c): x -> z

        Also returns the quantities necessary to compute the total
        log-likelihood, p(X|c)

        Parameters
        ----------
        x: (Nb, N) tensor
            Nb-batch of input tensors to be transformed.
        c: (Nb, Nc) tensor
            Nb-batch of conditioning inputs of dimension Nc.
        training: bool, optional
            Training pass. Default: False.

        Returns
        -------
        z: (Nb, N) tensor
            Nb-batch of transformed tensors.
        log_pz: (Nb,) tensor
            Nb-batch of log-likelihoods.
        jac_logdet: (Nb,) tensor
            Nb-batch of Jacobian log-determinants.
        """
        # Loop over bijectors
        z = x
        log_jacobs = 0
        for bijector, flip in zip(self.bijectors, self.flips):

            # Forward pass to get estimated latent variable
            z, logdet = bijector(z, c, flip=flip, training=training)

            # Compute prior log-likelihood
            log_pz = self._base_dist.log_prob(z)

            # Accumulate Jacobian log-determinant
            log_jacobs += logdet

        return z, log_pz, log_jacobs

    def inverse(self, z, c, training=False):
        """
        Inverse pass through chain of bijectors, along with conditioning
        inputs c.

        f^{-1}(z; c): z -> x

        Parameters
        ----------
        z: (Nb, N) tensor
            Nb-batch of input tensors to be transformed.
        c: (Nb, Nc) tensor
            Nb-batch of conditioning inputs of dimension Nc.
        flip: bool, optional
            Flip the split inputs. Default: False.
        training: bool, optional
            Training pass. Default: False.

        Returns
        -------
        x: (Nb, N) tensor
            Nb-batch of transformed tensors.
        """
        # Loop over bijectors and flips in reverse direction
        x = z
        for bijector, flip in zip(reversed(self.bijectors), reversed(self.flips)):
            x = bijector.inverse(x, c, flip=flip, training=training)
        return x

    def sample(self, c):
        """
        Generate sample z from a base distribution and perform inverse pass
        through flows in order to sample x ~ f^{-1}(z; c).

        Parameters
        ----------
        c: (Nb, Nc) tensor
            Nb-batch of conditioning inputs of dimension Nc.

        Returns
        -------
        x: (Nb, N) tensor
            Nb-batch of sampled tensors.
        """
        n_samples = tf.shape(c)[0]
        z = self._base_dist.sample(n_samples)
        z = tf.reshape(z, (-1, self.N))
        x = self.inverse(z, c)
        return x

    def log_prob(self, x, c):
        """
        Compute log probability of data samples by flowing through forward passes
        of bijectors. Convenience call around self.call().

        Parameters
        ----------
        x: (Nb, N) tensor
            Nb-batch of input tensors to be transformed.
        c: (Nb, Nc) tensor
            Nb-batch of conditioning inputs of dimension Nc.

        Returns
        -------
        ll: scalar
            Total log-likelihood of inputs.
        """
        _, log_pz, log_jacobs = self.call(x, c)
        return log_pz + log_jacobs


# end of file
