import numpy as np
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.utils import try_import_tf, try_import_tfp
from tensorflow_probability import distributions
tf = try_import_tf()
tfp = try_import_tfp()

class BetaDistribution(TFActionDistribution):

    def __init__(self, inputs, model):
        alpha, beta = tf.split(inputs, 2, axis=1)
        self.alpha = alpha + tf.constant(1.0) + tf.constant(np.finfo(np.float32).tiny)
        self.beta = beta + tf.constant(1.0) + tf.constant(np.finfo(np.float32).tiny)
        # self.alpha = tf.constant(1.0)
        # self.beta = tf.constant(1.0)
        self.dist = distributions.Beta(concentration1=self.alpha, concentration0=self.beta, validate_args=True, allow_nan_stats=False)
        super().__init__(inputs, model)

    def deterministic_sample(self):
        return self.dist.mean()

    def logp(self, x):
        print(x)
        x += tf.constant(np.finfo(np.float32).tiny)
        test = self.dist.log_prob(x)
        print(test)
        return test

    def kl(self, other):
        return self.dist.kl_divergence(other)

    def entropy(self):
        return self.dist.entropy()

    def _build_sample_op(self):
        return self.dist.sample()

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape) * 2
