import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from config_ml import Config
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
import numpy as np
from ray.rllib.policy.rnn_sequencing import add_time_dimension

# Add these to the model_config dict:
# model_config.get("centralized_input_size")
# model_config.get("policy_input_size")

settings = Config()
settings = settings.load_conf('config_file')

tf = try_import_tf()
hiddensize = 256
bias_initializer = tf.keras.initializers.Zeros()
centralized = True

class MyModelCentralized(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyModelCentralized, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        activation = 'tanh'
        # activation = 'relu'
        activation_value = 'tanh'

        # activation = get_activation_fn(model_config.get("fcnet_activation"))

        # Create policy network
        # input_shape_policy = model_config.get("policy_input_size")
        input_policy = tf.keras.layers.Input(shape=obs_space.shape, name="input_policy")
        policy_layer_1 = tf.keras.layers.Dense(hiddensize,
                                               name="policy_layer_1",
                                               bias_initializer=bias_initializer,
                                               kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
                                               activation=activation)(input_policy)
        policy_layer_2 = tf.keras.layers.Dense(hiddensize,
                                               name='policy_layer_2',
                                               bias_initializer=bias_initializer,
                                               kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
                                               activation=activation)(policy_layer_1)
        policy_layer_out = tf.keras.layers.Dense(num_outputs,
                                                 name="policy_layer_out",
                                                 activation=activation,
                                                 bias_initializer=bias_initializer,
                                                 kernel_initializer=normc_initializer(0.01))(policy_layer_2)

        self.policy_model = tf.keras.Model(inputs=input_policy, outputs=policy_layer_out)
        self.register_variables(self.policy_model.variables)

        # Create normal value network
        obs_normal = tf.keras.layers.Input(shape=obs_space.shape, name="obs_normal")
        vf_1 = tf.keras.layers.Dense(hiddensize,
                                             name="vf_1",
                                             activation=activation_value,
                                             kernel_initializer=normc_initializer(1.0))(obs_normal)
        vf_2 = tf.keras.layers.Dense(hiddensize,
                                             name="vf_2",
                                             activation=activation_value,
                                             kernel_initializer=normc_initializer(1.0))(vf_1)
        vf_out = tf.keras.layers.Dense(1,
                                        name="vf_out",
                                        activation=tf.keras.activations.linear,
                                        kernel_initializer=normc_initializer(0.01))(vf_2)
        self.vf = tf.keras.Model(
            inputs=obs_normal, outputs=vf_out)
        self.register_variables(self.vf.variables)

        # Create centralized critic
        # input_shape_centralized = model_config.get("centralized_input_size")

        input_size = obs_space.shape[-1] + settings.n_neighbours

        obs_centralized = tf.keras.layers.Input(shape=(input_size,), name="obs_centralized")
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
        # print('INSIDE MODEL WTF WORK :PEPEBLANKET2:')
        self._value_out = tf.reshape(self.vf(input_dict['obs']), [-1])
        return policy_out, state

    def value_function(self):
        return self._value_out

    def central_value_function(self, obs):
        # print('inside value fucntion')
        # idx_insert = np.arange(7, 7 + settings.n_neighbours * 3, 3)
        # for idx, actionlist in enumerate(opponent_action):
        #     action_list_deg = list(map(transform_action, actionlist))
        #     obs[idx] = np.insert(obs[idx], idx_insert[:-1], np.array(action_list_deg[:-1]))
        #     obs[idx].append(action_list_deg[-1])
        #
        #
        # print(obs)
        # combined_obs = tf.reshape(self.central_vf(obs), [-1])
        # self.central_vf(combined_obs)
        # print(obs)
        return tf.reshape(self.central_vf(obs), [-1])

class MyModelCentralized2(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, hidden_size=256, cell_size=64):
        super(MyModelCentralized2, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.cell_size = cell_size
        # hiddensize = hidden_size
        activation = 'tanh'
        # activation = 'relu'
        activation_value = 'tanh'

        # activation = get_activation_fn(model_config.get("fcnet_activation"))

        # Create policy network
        # input_shape_policy = model_config.get("policy_input_size")
        input_policy = tf.keras.layers.Input(shape=(None, obs_space.shape[0]), name="input_policy")
        state_in_h_policy = tf.keras.layers.Input(shape=(cell_size, ), name='h')
        state_in_c_policy = tf.keras.layers.Input(shape=(cell_size, ), name='c')
        seq_in = tf.keras.layers.Input(shape=(), name='seq_in', dtype=tf.int32)

        dense1 = tf.keras.layers.Dense(
            hidden_size, activation=tf.nn.relu, name='dense1')(input_policy)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name='LSTM')(inputs=dense1,
                                                                              mask=tf.sequence_mask(seq_in),
                                                                              initial_state=[state_in_h_policy, state_in_c_policy])
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name='logits')(lstm_out)

        self.policy_model = tf.keras.Model(
            inputs=[input_policy, seq_in, state_in_h_policy, state_in_c_policy],
            outputs=[logits, state_h, state_c])
        self.register_variables(self.policy_model.variables)
        #########3
        obs_normal = tf.keras.layers.Input(shape=obs_space.shape, name="obs_normal")
        vf_1 = tf.keras.layers.Dense(hiddensize,
                                     name="vf_1",
                                     activation=activation_value,
                                     kernel_initializer=normc_initializer(1.0))(obs_normal)
        # vf_2 = tf.keras.layers.Dense(hiddensize,
        #                              name="vf_2",
        #                              activation=activation_value,
        #                              kernel_initializer=normc_initializer(1.0))(vf_1)
        vf_out = tf.keras.layers.Dense(1,
                                       name="vf_out",
                                       activation=tf.keras.activations.linear,
                                       kernel_initializer=normc_initializer(0.01))(vf_1)
        self.vf = tf.keras.Model(
            inputs=obs_normal, outputs=vf_out)
        self.register_variables(self.vf.variables)
        # Create centralized critic
        # input_shape_centralized = model_config.get("centralized_input_size")

        input_size = obs_space.shape[-1] + settings.n_neighbours

        obs_centralized = tf.keras.layers.Input(shape=(input_size,), name="obs_centralized")
        central_vf_1 = tf.keras.layers.Dense(hidden_size,
                                             name="central_vf_1",
                                             activation=activation_value,
                                             kernel_initializer=normc_initializer(1.0))(obs_centralized)
        central_vf_2 = tf.keras.layers.Dense(hidden_size,
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
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        self._value_out = tf.reshape(self.vf(input_dict['obs']), [-1])
        output, new_state = self.forward_rnn(
            add_time_dimension(
                input_dict["obs_flat"], seq_lens, framework="tf"), state,
            seq_lens)
        return tf.reshape(output, [-1, self.num_outputs]), new_state


    def forward_rnn(self, inputs, state, seq_lens):
        model_out, h, c = self.policy_model([inputs, seq_lens] +
                                                          state)
        return model_out, [h, c]

    def value_function(self):
        return self._value_out

    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    def central_value_function(self, obs, states=None, seq_lens=None):
        # output, new_state = self.forward_rnn(
        #     add_time_dimension(
        #         input_dict["obs_flat"], seq_lens, framework="tf"), state,
        #     seq_lens)
        # model_out, h, c = self.policy_model([inputs, seq_lens] +
        #                                     state)
        return tf.reshape(self.central_vf(obs), [-1])
