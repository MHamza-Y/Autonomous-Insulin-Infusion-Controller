from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy

from save_on_best_result_callback_v3 import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from train.env.simglucose_gym_env import T1DSimEnv, T1DSimDiffEnv
from train.reward.custom_rewards import custom_reward, shaped_reward_around_normal_bg, \
    shaped_negative_reward_around_normal_bg
import torch as th


def main():
    save_folder = 'training_ws/'
    checkpoint_callback = CheckpointCallback(save_freq=256, save_path=save_folder,
                                             name_prefix="rl_model")
    best_model_save_callback = SaveOnBestTrainingRewardCallback(check_freq=256, log_dir=save_folder)

    vec_env_kwargs = {'start_method': 'fork'}
    env_kwargs = {'reward_fun': shaped_negative_reward_around_normal_bg}
    env = make_vec_env(T1DSimDiffEnv, n_envs=48, monitor_dir='./training_ws', vec_env_cls=SubprocVecEnv,
                       vec_env_kwargs=vec_env_kwargs, env_kwargs=env_kwargs)
    policy_kwargs = {'activation_fn': th.nn.LeakyReLU, 'net_arch': [16, 16, 16, 16]}
    model = PPO(MlpPolicy, env, verbose=1, tensorboard_log="./simglucose_ppo_tensorboard/",
                n_steps=256, gae_lambda=0.98, learning_rate=1e-5, ent_coef=0.0006, n_epochs=10, gamma=0.99,
                policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=50000000, callback=[best_model_save_callback, checkpoint_callback])


if __name__ == "__main__":
    main()
