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
import logging


# logging.basicConfig(filename='logoutpost.log', level=logging.DEBUG)

settings = Config()
settings = settings.load_conf('config_file')

tf = try_import_tf()
NEW_OBS_ACTION = "cur_obs_with_action"

class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = make_tf_callable(self.get_session(),dynamic_shape=True)(
            self.model.central_value_function)

def centralized_critic_postprocessing(policy,sample_batch,other_agent_batches=None, episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""
    # print('I AM HERE')
    if policy.loss_initialized():
        assert other_agent_batches is not None
        # own_agent_id = sample_batch[SaAh yeahmpleBatch.INFOS][-1]['sequence'][0]

        OppActionsArr = []
        New_obs_list = []
        New_obs_array = np.array([])
        # if isinstance(sample_batch[SampleBatch.INFOS], np.ndarray):
        for idx, value in enumerate(sample_batch[SampleBatch.INFOS]):
            if "sequence" in value:
                neighbours_ac = value['sequence'][1:]
                # print(neighbours_ac)
                OpponentActions = []
                idx_insert = np.arange(8, 8 + settings.n_neighbours * 3, 3)
                # print(idx_insert)
                # idx_insert = idx_insert[:-1]
                temp = sample_batch[SampleBatch.CUR_OBS][idx]
                temp = np.append(temp, np.float32(0))

                for idx_insert_idx, agentid in enumerate(neighbours_ac):
                    # print('Print idx', idx_insert[idx_insert_idx])
                    # print('Print sample batch', temp) #sample_batch[SampleBatch.CUR_OBS][idx])
                    # print('Print transform', transform_action(other_agent_batches[agentid][1][SampleBatch.ACTIONS][idx]))
                    # temp_action = transform_action(other_agent_batches[agentid][1][SampleBatch.ACTIONS][idx])
                    temp = np.insert(temp, idx_insert[idx_insert_idx], transform_action(other_agent_batches[agentid][1][SampleBatch.ACTIONS][idx]))
                    # New_obs_list.append(temp)
                    OpponentActions.append(other_agent_batches[agentid][1][SampleBatch.ACTIONS][idx])

                if len(temp[:-1]) < max(idx_insert+1):
                    fill = max(idx_insert+1) - len(temp[:-1])
                    fill_zero = np.full((1,fill), -1, dtype=np.float32)
                    temp = np.append(temp, fill_zero)
                # New_obs_list = np.append(New_obs_list, temp[:-1])
                OppActionsArr.append(OpponentActions)
                New_obs_list.append(temp[:-1])
                # print('idxinsert', max(idx_insert))
                # print(np.shape(temp[:-1]))
                if idx == 0:
                    New_obs_array = np.array([temp[:-1]])
                else:
                    New_obs_array = np.concatenate((New_obs_array, [temp[:-1]]), axis=0)
            else:
                # New_obs_list.append(sample_batch[SampleBatch.CUR_OBS][idx])
                # if len(sample_batch[SampleBatch.CUR_OBS][idx]) < (5 + settings.n_neighbours * 3):
                fill = (6 + settings.n_neighbours * 3) - len(sample_batch[SampleBatch.CUR_OBS][idx])
                fill_zero = np.full((1, fill), -1, dtype=np.float32)
                temp_2 = np.append(sample_batch[SampleBatch.CUR_OBS][idx], fill_zero)
                if idx==0:
                    New_obs_array = temp_2
                else:
                    New_obs_array = np.concatenate((New_obs_array, [temp_2]), axis=0)
        # logger.debug(sample_batch["CUR_OBS_WITH_ACTION"])
        # sample_batch[NEW_OBS_ACTION] = np.array([])
        # print(sample_batch[NEW_OBS_ACTION])

        # logging.debug("Sample batch")
        # logging.debug(sample_batch)
        # logging.debug(np.array(New_obs_list, dtype=np.float32))
        # sample_batch[NEW_OBS_ACTION] = np.array(New_obs_list, dtype=np.float32)
        # print(New_obs_array)
        sample_batch[NEW_OBS_ACTION] = New_obs_array
        # sample_batch[NEW_OBS_ACTION] = np.array(New_obs_list, dtype=np.float32)
        # logging.debug("New obs action")
        # logging.debug(sample_batch[NEW_OBS_ACTION])
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(sample_batch[NEW_OBS_ACTION])
    else:
        # print('past here')
        fake_size = 6 + settings.n_neighbours*3
        sample_batch[NEW_OBS_ACTION] = np.array([])
        sample_batch[NEW_OBS_ACTION] = np.zeros((1, fake_size), dtype=np.float32)
        # print(sample_batch[NEW_OBS_ACTION])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS], dtype=np.float32)
        # print(sample_batch[SampleBatch.VF_PREDS])

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

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

def include_action_in_obs(obs, opponent_actions):
    idx_insert = np.arange(7, 7 + settings.n_neighbours * 3, 3)
    # for idx, actionlist in enumerate(opponent_action):
    #     action_list_deg = list(map(transform_action, actionlist))
    #     obs[idx] = np.insert(obs[idx], idx_insert[:-1], np.array(action_list_deg[:-1]))
    #     obs[idx].append(action_list_deg[-1])
    return


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

# print('Create CCPPO policy')
CCPPO = PPOTFPolicy.with_updates(
    name="CCPPO",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_loss_init=setup_mixins,
    grad_stats_fn=central_vf_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])
# print(CCPPO)

def get_policy_class(config):
    return CCPPO
CCTrainer = PPOTrainer.with_updates(name="CCPPOTrainer", default_policy=CCPPO, get_policy_class=get_policy_class)
# print(CCTrainer)