import numpy as np
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.utils import try_import_tf, try_import_tfp
# from tensorflow_probability import distributions
tf = try_import_tf()
tfp = try_import_tfp()

class BetaDistributionAction(TFActionDistribution):

    def __init__(self, inputs, model):
        alpha, beta = tf.split(inputs, 2, axis=1)
        self.epsilon = tf.constant(1e-7)
        self.alpha = tf.clip_by_value(alpha, 1.0, tf.float32.max)
        self.beta = tf.clip_by_value(alpha, 1.0, tf.float32.max)
        self.dist = tfp.distributions.Beta(concentration1=self.alpha, concentration0=self.beta, validate_args=True, allow_nan_stats=False)
        super().__init__(inputs, model)

    def deterministic_sample(self):
        # print('mean :', self.dist.mean())
        return self.dist.mean()

    def logp(self, x):
        # print(x)
        x = tf.clip_by_value(x, self.epsilon, 1-self.epsilon)
        test = -tf.reduce_sum(self.dist.log_prob(x), axis=1)
        # print('logp :', test)
        return test

    def kl(self, other):
        return tf.reduce_sum(self.dist.kl_divergence(other.dist), axis=1)

    def entropy(self):
        return tf.reduce_sum(self.dist.entropy(), axis=1)

    def _build_sample_op(self):
        return self.dist.sample()

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape) * 2
