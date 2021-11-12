from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy

from save_on_best_result_callback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from train.env.simglucose_gym_env import T1DSimEnv


def main():
    save_folder = 'training_ws/'
    checkpoint_callback = CheckpointCallback(save_freq=256, save_path=save_folder,
                                             name_prefix="rl_model")
    best_model_save_callback = SaveOnBestTrainingRewardCallback(check_freq=256, log_dir=save_folder)

    vec_env_kwargs = {'start_method': 'fork'}
    env = make_vec_env(T1DSimEnv, n_envs=64, monitor_dir='./training_ws', vec_env_cls=SubprocVecEnv,
                       vec_env_kwargs=vec_env_kwargs)

    model = PPO(MlpPolicy, env, verbose=1, tensorboard_log="./simglucose_ppo_tensorboard/",
                n_steps=1024, gae_lambda=0.98,learning_rate=1e-3, ent_coef=0.01, n_epochs=4, gamma=0.999)
    #policy_kwargs={'net_arch': [dict(pi=[32, 32], vf=[32, 32])]}

    model.learn(total_timesteps=50000000, callback=[best_model_save_callback, checkpoint_callback])


if __name__ == "__main__":
    main()
