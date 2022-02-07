#-*- coding: utf-8 -*-

# Defaults
import numpy as np
import six

# Tensorflow
import tensorflow as tf
# Use float64 for now for precision stability
DTYPE = tf.float64
tf.keras.backend.set_floatx('float64')

# Tensorflow probability
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class ConditionalNormal(tfd.Distribution):
    """
    An independent normal distribution where loc and scale are conditioned on
    supplied input variables. These input variables are provided to a neural network
    which outputs loc and scale.
    """

    def __init__(self, model_generator, batch_size, validate_args=False, allow_nan_stats=True,
                 dtype=DTYPE, name='ConditionalNormal'):

        self._batch_size = batch_size

        # Initialize parent class and encompassing name scope
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(ConditionalNormal, self).__init__(
                dtype=dtype,
                reparameterization_type=tfd.ReparameterizationType("FULLY_REPARAMETERIZED"),
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name
            )

            # Create neural network parameter model within name scope
            self._parameter_model = model_generator()

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            loc=tfp.util.ParameterProperties(),
            scale=tfp.util.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: tfb.Softplus(low=eps(dtype))
                )
            )
        )

    def parameter_model(self, model_args, squeeze=False):
        """
        Directly evaluate underlying parameter model to get mean field loc and scale.
        """
        loc, scale = self._parameter_model(*model_args, squeeze=squeeze)
        return loc, scale

    def mvn(self, **kwargs):
        """
        Generate a multivariate normal distribution with diagonal covariance. Serves as
        a multivariate version of independent normals for our specified batch size.
        loc and scale for mvn parameters are provided by neural network model.
        """
        model_args = kwargs['model_args']
        loc, scale = self._parameter_model(*model_args, squeeze=True)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
    
    def _sample_n(self, n, seed=None, **kwargs):
        """
        Generate n samples by passing *args to _parameter_model to get loc and scale.
        """
        # Get parameter model args and predict loc and scale
        model_args = kwargs['model_args']
        loc, scale = self._parameter_model(*model_args, squeeze=True)
        # Sample from normal distribution
        shape = tf.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)], axis=0)
        sampled = sample_normal(
            shape=shape, mean=0.0, stddev=1.0, dtype=self.dtype, seed=seed
        )
        # Scale and shift
        return sampled * scale + loc

    def _log_prob(self, x, **kwargs):
        """
        Compute log probability at *args.
        """
        # Get parameter model args and predict loc and scale
        model_args = kwargs['model_args']
        loc, scale = self._parameter_model(*model_args, squeeze=True)
        # Compute log probability
        log_unnormalized = -0.5 * tf.math.squared_difference(
            x / scale, loc / scale
        )
        log_normalization = tf.constant(0.5 * np.log(2.0 * np.pi), dtype=self.dtype) + \
                            tf.math.log(scale)
        return tf.reduce_sum(log_unnormalized - log_normalization, axis=1)

    def _entropy(self, **kwargs):
        """
        Compute entropy of independent normal distribution, parameterized by
        loc and scale generated by _parameter_model.
        """
        # Get parameter model args and predict loc and scale
        model_args = kwargs['model_args']
        loc, scale = self._parameter_model(*model_args, squeeze=True)
        # Compute entropy
        log_normalization = tf.constant(
            0.5 * np.log(2.0 * np.pi), dtype=self.dtype) + tf.math.log(scale)
        entropy = 0.5 + log_normalization
        return entropy * tf.ones_like(loc)

    # Samples are independent, so event shapes are scalar and set to batch size
    def _event_shape_tensor(self):
        return tf.constant([self._batch_size], dtype=tf.int32)
    def _event_shape(self):
        return tf.TensorShape([self._batch_size])


class DenseNetwork(tf.keras.Model):
    """
    Simple feedforward, multi-layer neural network.
    """

    def __init__(self, layer_sizes, initializer='lecun_normal', dtype=DTYPE, name='net'):
        # Initialize parent class
        super().__init__(name=name)

        # Create and store layers
        self.net_layers = []
        for count, size in enumerate(layer_sizes):
            # Layer names by depth count
            name = 'dense_%d' % count
            self.net_layers.append(
                tf.keras.layers.Dense(
                    size,
                    activation=None,
                    kernel_initializer=initializer,
                    dtype=dtype,
                    name=name
                )
            )
        self.n_layers = len(self.net_layers)

    def call(self, inputs, activation=tf.tanh, training=False, activate_outputs=False):
        """
        Pass inputs through network and generate an output. All layers except the last
        layer will pass through an activation function.
        """
        # Pass through all layers
        out = inputs
        for count, layer in enumerate(self.net_layers):
            out = layer(out, training=training)
            if count != (self.n_layers - 1):
                out = activation(out)

        # Output activation
        if activate_outputs:
            out = activation(out)

        return out

# --------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------

def assemble_scale_tensors(norms, keys, dtype=np.float64):
    """
    Assembles linear transformation tensors for applying/inverting normalization
    operations.
    """
    # Scale tensor (2D)
    values = [norms[key].denom for key in keys]
    W = np.diag(np.array(values, dtype=dtype))

    # Bias tensor (1D)
    values = [norms[key].xmin for key in keys]
    b = np.array(values, dtype=dtype).reshape(-1, len(keys))

    return tf.convert_to_tensor(W, dtype=DTYPE), \
           tf.convert_to_tensor(b, dtype=DTYPE)

def normalize_tensor(x, W, b):
    """
    Apply normalization linear transformation.
    """
    return 2 * tf.matmul(x - b, 1.0/W) - 1

def inverse_normalize_tensor(xn, W, b):
    """
    Apply inverse normalization linear transformation. 
    """
    return tf.matmul(0.5 * (xn + 1), W) + b

# --------------------------------------------------------------------------------
# Some necessary machinery pulled out from tfp.python.internal
# --------------------------------------------------------------------------------

def eps(dtype):
  """Returns the distance between 1 and the next largest representable value."""
  return np.finfo(as_numpy_dtype(dtype)).eps

def as_numpy_dtype(dtype):
  """Returns a `np.dtype` based on this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'as_numpy_dtype'):
    return dtype.as_numpy_dtype
  return dtype

def is_stateful_seed(seed):
  return seed is None or isinstance(seed, six.integer_types)

def sample_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None):
  """As `tf.random.normal`, but handling stateful/stateless `seed`s."""
  with tf.name_scope(name or 'normal'):
    # TODO(b/147874898): Remove workaround for seed-sensitive tests.
    if is_stateful_seed(seed):
      return tf.random.normal(
          shape=shape, seed=seed, mean=mean, stddev=stddev, dtype=dtype)

    seed = tfp.random.sanitize_seed(seed)
    return tf.random.stateless_normal(
        shape=shape, seed=seed, mean=mean, stddev=stddev, dtype=dtype)

# end of file
