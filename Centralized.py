import numpy as np

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin, \
    PPOLoss as TFLoss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils import try_import_tf

from config_ml import Config

settings = Config()
settings = settings.load_conf('config_file')

tf = try_import_tf()
NEW_OBS_ACTION = "cur_obs_with_action"

class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = make_tf_callable(self.get_session(), dynamic_shape=True)(
            self.model.central_value_function)

def centralized_critic_postprocessing(policy,sample_batch,other_agent_batches=None, episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""
    if policy.loss_initialized():
        assert other_agent_batches is not None
        # new_obs_array = np.array([])

        # Initiate frequently used values that stat consistent.
        idx_insert = np.arange(9, 9 + settings.n_neighbours * 3, 3)

        for idx_sample, value in enumerate(sample_batch[SampleBatch.INFOS]):
            if "sequence" in value:

                # Make new array containing all neighbouring AC for this current observation. This is passed along with
                # the info dict.
                # Delete the first element, as it contains the agent ID of the currenct agent considered.
                neighbours_ac = value['sequence'][1:]
                # idx_insert = np.arange(8, 8 + settings.n_neighbours * 3, 3)
                # print('AgentIDinbatch:', sample_batch[SampleBatch.AGENT_INDEX][idx_sample])
                # print('AgentIDinInfo', value['sequence'][0])
                # temp = sample_batch[SampleBatch.CUR_OBS][idx]

                # Create temporary array copies the current observations (without the opponent actions), and append
                # a 0 at the end to be able to use np.insert.
                temp = np.append(sample_batch[SampleBatch.CUR_OBS][idx_sample], np.float32(0))

                # Now retrieve opponent actions, by looping over all the neighbour_ac agent ID's in the opponent_batch
                # The sequence of neighbours_ac follows the same orderning used in the observation space.
                # So this sequence is used to correctly place the opponent action behind the observation.

                for idx_insert_idx, agentid in enumerate(neighbours_ac):
                    temp = np.insert(temp, idx_insert[idx_insert_idx], transform_action(other_agent_batches[agentid][1][SampleBatch.ACTIONS][idx_sample]))

                # New array contains as many actions as given by the amount of neighbours_ac, which is given by the
                # amount of current active AC in the simulation. So if the amount of neighbours would drop below the
                # value in the settings, the state space would not be the same. So padding is required.
                # Padding is done by comparing the required state space size with the current created.
                # Difference is padded with "fill"

                if len(temp[:-1]) < max(idx_insert+1):
                    fill = max(idx_insert+1) - len(temp[:-1])
                    fill_zero = np.full((1, fill), -1, dtype=np.float32)
                    temp = np.append(temp, fill_zero)

                # New_obs_list = np.append(New_obs_list, temp[:-1])

                # Delete the last element, which was the padded 0 for the np.insert.
                # New_obs_list.append(temp[:-1])

                # First sample should create a new array, rest should be appended. (remember, this is the full sample
                # batch). temp[:-1] is done to delete the additional 0 used to enable NP.INSERT to work.

                if idx_sample == 0:
                    new_obs_array = np.array([temp[:-1]])
                else:
                    new_obs_array = np.concatenate((new_obs_array, [temp[:-1]]), axis=0)
            else:
                # If sequence is not present in the info dict, this means that there is only 1 plane left.
                # create required padding and send as observation.
                fill = (7 + settings.n_neighbours * 3) - len(sample_batch[SampleBatch.CUR_OBS][idx_sample])
                fill_zero = np.full((1, fill), -1, dtype=np.float32)
                temp_2 = np.append(sample_batch[SampleBatch.CUR_OBS][idx_sample], fill_zero)

                if idx_sample == 0:
                    new_obs_array = temp_2
                else:
                    new_obs_array = np.concatenate((new_obs_array, [temp_2]), axis=0)

        # Add new observations including actions to sample batch.
        sample_batch[NEW_OBS_ACTION] = new_obs_array

        # Calculated the predicted value function, and include in the batch.
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(sample_batch[NEW_OBS_ACTION])

    else:
        # If policy is not initialized, create dummy batch.

        fake_size = 7 + settings.n_neighbours*3
        sample_batch[NEW_OBS_ACTION] = np.array([])
        sample_batch[NEW_OBS_ACTION] = np.zeros((1, fake_size), dtype=np.float32)
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS], dtype=np.float32)

    # Check if sample_batch is done to tidy up stuff.
    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    # Compute advantages using the new observations.
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])

    return batch

def transform_action(action):
    action_hdg = 0
    if action == 0:
        action_hdg = -15
    elif action == 1:
        action_hdg = -10
    elif action == 2:
        action_hdg = -5
    elif action == 3:
        action_hdg = 0
    elif action == 4:
        action_hdg = 5
    elif action == 5:
        action_hdg = 10
    elif action == 6:
        action_hdg = 15
    return np.float32(action_hdg)

def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)

    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    policy.central_value_out = policy.model.central_value_function(
        train_batch[NEW_OBS_ACTION])
    adv = tf.ones_like(train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)
    policy.loss_obj = TFLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        policy.central_value_out,
        policy.kl_coeff,
        adv,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"])

    return policy.loss_obj.loss

def setup_mixins(policy, obs_space, action_space, config):
    # copied from PPO
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])

def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.central_value_out),
    }

def choose_optimizer(policy, config):
    if True:
        return tf.train.AdamOptimizer(policy.cur_lr)
    elif False:
        return tf.train.RMSPropOptimizer(policy.cur_lr)


CCPPO = PPOTFPolicy.with_updates(
    name="CCPPO",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_loss_init=setup_mixins,
    grad_stats_fn=central_vf_stats,
    optimizer_fn=choose_optimizer,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])

def get_policy_class(config):
    return CCPPO

CCTrainer = PPOTrainer.with_updates(name="CCPPOTrainer", default_policy=CCPPO, get_policy_class=get_policy_class)
