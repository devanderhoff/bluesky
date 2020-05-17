import numpy as np
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.utils import try_import_tf, try_import_tfp
from ray.rllib.utils.annotations import override, DeveloperAPI
# from tensorflow_probability import distributions
tf = try_import_tf()
tfp = try_import_tfp()

class BetaDistributionAction(TFActionDistribution):

    def __init__(self, inputs, model):
        # print('input :', inputs)
        alpha, beta = tf.split(inputs, 2, axis=1)
        self.epsilon = tf.constant(1e-7)
        # self.alpha = tf.clip_by_value(alpha, 1.0, tf.float32.max)
        # self.beta = tf.clip_by_value(beta, 1.0, tf.float32.max)
        # print('Alpha :', self.alpha, 'Beta :', self.beta)
        # print('input:', alpha)
        self.alpha = tf.math.maximum(1.0, alpha)
        self.beta = tf.math.maximum(1.0, beta)
        # print('limited :', self.alpha)

        self.dist = tfp.distributions.Beta(concentration1=self.alpha, concentration0=self.beta, validate_args=True, allow_nan_stats=False)
        super().__init__(inputs, model)

    def deterministic_sample(self):
        # print('mean :', self.dist.mean())
        return self.dist.mean()

    def logp(self, x):
        x = tf.clip_by_value(x, self.epsilon, 1-self.epsilon)
        test = -tf.reduce_sum(self.dist.log_prob(x), axis=1)
        # print('logp :', test)
        return test

    def kl(self, other):
        # print('kl :',self.dist.kl_divergence(other.dist))
        return self.dist.kl_divergence(other.dist)

    def entropy(self):
        # print('entropy :', self.dist.entropy())
        return self.dist.entropy()

    def _build_sample_op(self):
        # print('sample :', self.dist.sample())
        return self.dist.sample()

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape) * 2

class CategoricalOrdinal(TFActionDistribution):
    """Categorical distribution for discrete action spaces."""

    def __init__(self, inputs, model=None, temperature=1.0):
        assert temperature > 0.0, "Categorical `temperature` must be > 0.0!"
        # Allow softmax formula w/ temperature != 1.0:
        # Divide inputs by temperature.

        inputs = tf.nn.sigmoid(inputs)  # of size [batchsize, num-actions*bins], initialized to be about uniform
        # output1 = tf.nn.softmax(inputs)
        #
        # inputs = tf.nn.sigmoid(inputs)
        #
        am_numpy = self.construct_mask(inputs.shape[-1])
        am_tf = tf.constant(am_numpy, dtype=tf.float32)
        inputs = tf.tile(tf.expand_dims(inputs, axis=-1), [1, 1, inputs.shape[-1]])
        inputs = tf.reduce_sum(tf.math.log(inputs + 1e-8) * am_tf + tf.math.log(1 - inputs + 1e-8) * (1 - am_tf),
                                axis=-1)
        # print('inputs :', inputs)

        super().__init__(inputs / temperature, model)

    def deterministic_sample(self):
        return tf.math.argmax(self.inputs, axis=1)

    def logp(self, x):
        y = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.inputs, labels=tf.cast(x, tf.int32))
        # print("logp :", y)
        return y

    def entropy(self):
        a0 = self.inputs - tf.reduce_max(self.inputs, axis=1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=1)

    def kl(self, other):
        a0 = self.inputs - tf.reduce_max(self.inputs, axis=1, keep_dims=True)
        a1 = other.inputs - tf.reduce_max(other.inputs, axis=1, keep_dims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=1, keep_dims=True)
        z1 = tf.reduce_sum(ea1, axis=1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=1)

    def _build_sample_op(self):
        y = tfp.distributions.Categorical(logits=self.inputs)
        sample = y.sample()
        return sample

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return action_space.n

    @staticmethod
    def construct_mask(bins):
        a = np.zeros([bins, bins])
        for i in range(bins):
            for j in range(bins):
                if i + j <= bins - 1:
                    a[i, j] = 1.0
        return a

class CategoricalOrdinalTFP(TFActionDistribution):
    """Categorical distribution for discrete action spaces."""

    def __init__(self, inputs, model=None, temperature=1.0):
        assert temperature > 0.0, "Categorical `temperature` must be > 0.0!"
        # Allow softmax formula w/ temperature != 1.0:
        # Divide inputs by temperature.
        inputs = inputs / temperature
        inputs = tf.nn.sigmoid(inputs)  # of size [batchsize, num-actions*bins], initialized to be about uniform
        # output1 = tf.nn.softmax(inputs)
        #
        # inputs = tf.nn.sigmoid(inputs)
        #
        am_numpy = self.construct_mask(inputs.shape[-1])
        am_tf = tf.constant(am_numpy, dtype=tf.float32)
        inputs = tf.tile(tf.expand_dims(inputs, axis=-1), [1, 1, inputs.shape[-1]])
        inputs = tf.reduce_sum(tf.math.log(inputs + 1e-8) * am_tf + tf.math.log(1 - inputs + 1e-8) * (1 - am_tf),
                                axis=-1)

        self.dist = tfp.distributions.Categorical(logits=inputs, dtype=tf.int64)
        super().__init__(inputs, model)

    def deterministic_sample(self):
        return self.dist.mode()

    def logp(self, x):
        return self.dist.log_prob(x)

    def entropy(self):
         return self.dist.entropy()

    def kl(self, other):
        return self.dist.kl_divergence(other.dist)

    def _build_sample_op(self):
        return self.dist.sample()

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return action_space.n

    @staticmethod
    def construct_mask(bins):
        a = np.zeros([bins, bins])
        for i in range(bins):
            for j in range(bins):
                if i + j <= bins - 1:
                    a[i, j] = 1.0
        return a


