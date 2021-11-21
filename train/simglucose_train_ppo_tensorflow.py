from simglucose.simulation.env import risk_diff
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines.common.policies import MlpLnLstmPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize

from train.env.simglucose_gym_env import T1DSimEnv, T1DDiscreteSimEnv, T1DAdultSimEnv
from train.reward.custom_rewards import shaped_reward_around_normal_bg, shaped_negative_reward_around_normal_bg, \
    smooth_reward, no_negativity
from train.save_on_best_result_callback_v2 import SaveOnBestTrainingRewardCallback


def main():
    save_folder = 'training_ws/'
    checkpoint_callback = CheckpointCallback(save_freq=128, save_path=save_folder,
                                             name_prefix="rl_model")
    env_class = T1DAdultSimEnv
    reward_func = no_negativity
    vec_env_kwargs = {'start_method': 'fork'}
    env_kwargs = {'reward_fun': reward_func}
    n_envs = 16
    env = make_vec_env(env_class, n_envs=n_envs, monitor_dir='./training_ws', vec_env_cls=SubprocVecEnv,
                       vec_env_kwargs=vec_env_kwargs, env_kwargs=env_kwargs)
    #env = VecNormalize(env, clip_obs=350, clip_reward=1001, gamma=0.9999)

    policy_kwargs = {'net_arch': [8, 'lstm', dict(vf=[16, 16], pi=[16, 16])], 'n_lstm': 16}
    model = PPO2(MlpLnLstmPolicy, env, verbose=1, tensorboard_log="./simglucose_ppo_tensorboard/",
                 n_steps=32, learning_rate=3e-5, ent_coef=0.01, gamma=0.999, nminibatches=n_envs,
                 policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=50000000, callback=[checkpoint_callback])


if __name__ == "__main__":
    main()
#  , dict(vf=[32, 32], pi=[32, 32])
