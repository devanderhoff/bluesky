import numpy as np
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.utils import try_import_tf, try_import_tfp
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
