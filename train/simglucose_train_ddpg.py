import gym
from gym.envs.registration import register
from stable_baselines import PPO2, DDPG
from stable_baselines.common import make_vec_env
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.bench.monitor import Monitor

from save_on_best_result_callback import SaveOnBestTrainingRewardCallback
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from train.env.simglucose_gym_env import T1DSimEnv


def main():
    save_folder = 'training_ws/'
    checkpoint_callback = CheckpointCallback(save_freq=256, save_path=save_folder,
                                             name_prefix="rl_ddpg_model")
    best_model_save_callback = SaveOnBestTrainingRewardCallback(check_freq=256, log_dir=save_folder)

    vec_env_kwargs = {'start_method': 'spawn'}
    env = make_vec_env(T1DSimEnv, n_envs=40, monitor_dir='./training_ws', vec_env_cls=SubprocVecEnv,
                       vec_env_kwargs=vec_env_kwargs)
    # env = Monitor(T1DSimEnv(),filename='./training_ws')
    model = DDPG(MlpPolicy, reward_scale=0.01, gamma=0.99, buffer_size=10000)

    #model = PPO2.load(load_path='training_ws/1rl_model_19886080_steps.zip')
    #model.load_parameters({'learning_rate': 3e-4, 'noptepochs': 2, 'ent_coef': 0.0006})
    model.learn(total_timesteps=50000000, callback=[best_model_save_callback, checkpoint_callback])


if __name__ == "__main__":
    main()
