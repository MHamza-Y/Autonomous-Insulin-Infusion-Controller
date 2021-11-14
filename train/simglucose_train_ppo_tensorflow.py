from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize

from train.env.simglucose_gym_env import T1DSimEnv
from train.reward.custom_rewards import shaped_reward_around_normal_bg
from train.save_on_best_result_callback_v2 import SaveOnBestTrainingRewardCallback


def main():
    save_folder = 'training_ws/'
    checkpoint_callback = CheckpointCallback(save_freq=256, save_path=save_folder,
                                             name_prefix="rl_model")
    best_model_save_callback = SaveOnBestTrainingRewardCallback(check_freq=256, log_dir=save_folder)

    vec_env_kwargs = {'start_method': 'fork'}
    env_kwargs = {'reward_fun': shaped_reward_around_normal_bg}
    n_envs = 32
    env = make_vec_env(T1DSimEnv, n_envs=n_envs, monitor_dir='./training_ws', vec_env_cls=SubprocVecEnv,
                       vec_env_kwargs=vec_env_kwargs, env_kwargs=env_kwargs)
    #env = VecNormalize(env)
    policy_kwargs = {'net_arch': [16, 'lstm', dict(vf=[32, 32], pi=[32, 32])]}
    model = PPO2(MlpLnLstmPolicy, env, verbose=1, tensorboard_log="./simglucose_ppo_tensorboard/",
                 n_steps=256, learning_rate=1e-5, ent_coef=0.001, gamma=1, nminibatches=n_envs,
                 policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=50000000, callback=[best_model_save_callback, checkpoint_callback])


if __name__ == "__main__":
    main()
