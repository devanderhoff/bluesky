import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf

# Add these to the model_config dict:
# model_config.get("centralized_input_size")
# model_config.get("policy_input_size")


tf = try_import_tf()
hiddensize = 256

class MyModelCentralized(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyModelCentralized, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # activation = 'tanh'
        activation = 'softplus'
        activation_value = 'tanh'

        # activation = get_activation_fn(model_config.get("fcnet_activation"))

        # Create policy network
        # input_shape_policy = model_config.get("policy_input_size")
        input_policy = tf.keras.layers.Input(shape=obs_space.shape, name="input_policy")
        policy_layer_1 = tf.keras.layers.Dense(hiddensize,
                                               name="policy_layer_1",
                                               activation=activation,
                                               kernel_initializer=normc_initializer(1.0))(input_policy)
        policy_layer_2 = tf.keras.layers.Dense(hiddensize,
                                               name='policy_layer_2',
                                               activation=activation,
                                               kernel_initializer=normc_initializer(1.0))(policy_layer_1)
        policy_layer_out = tf.keras.layers.Dense(num_outputs,
                                                 name="policy_layer_out",
                                                 activation=activation,
                                                 kernel_initializer=normc_initializer(1.0))(policy_layer_2)

        self.policy_model = tf.keras.Model(inputs=input_policy, outputs=policy_layer_out)
        self.register_variables(self.policy_model.variables)

        # Create centralized critic
        # input_shape_centralized = model_config.get("centralized_input_size")
        obs_centralized = tf.keras.layers.Input(shape=obs_space.shape, name="obs_centralized")
        central_vf_1 = tf.keras.layers.Dense(hiddensize,
                                            name="central_vf_1",
                                            activation=activation_value,
                                            kernel_initializer=normc_initializer(1.0))(obs_centralized)
        central_vf_2 = tf.keras.layers.Dense(hiddensize,
                                            name="central_vf_2",
                                            activation=activation_value,
                                            kernel_initializer=normc_initializer(1.0))(central_vf_1)
        central_vf_out = tf.keras.layers.Dense(1,
                                               name="central_vf_out",
                                               activation=tf.keras.activations.linear,
                                               kernel_initializer=normc_initializer(0.01))(central_vf_2)
        self.central_vf = tf.keras.Model(
            inputs=obs_centralized, outputs=central_vf_out)
        self.register_variables(self.central_vf.variables)

        # #
        # self.complete_model = tf.keras.Model(inputs=[input_shape_policy, input_shape_centralized], outputs=[policy_layer_out, central_vf_out])
        # self.register_variables(self.complete_model.variables)

    def forward(self, input_dict, state, seq_lens):
        policy_out = self.policy_model(input_dict['obs'])
        # print(policy_out)
        self._value_out = tf.reshape(self.central_vf(input_dict['obs']), [-1])
        return policy_out, state

    def value_function(self):
        return self._value_out

    def central_value_function(self, obs_centralized):
        return tf.reshape(
            self.central_vf(obs_centralized), [-1])
