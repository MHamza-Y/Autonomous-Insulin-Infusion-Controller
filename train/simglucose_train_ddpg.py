
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy

from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.bench.monitor import Monitor

from save_on_best_result_callback import SaveOnBestTrainingRewardCallback

from train.env.simglucose_gym_env import T1DSimEnv


def main():
    save_folder = 'training_ws/'
    checkpoint_callback = CheckpointCallback(save_freq=256, save_path=save_folder,
                                             name_prefix="rl_ddpg_model")
    best_model_save_callback = SaveOnBestTrainingRewardCallback(check_freq=256, log_dir=save_folder)


    env = Monitor(T1DSimEnv(),filename='./training_ws')
    model = DDPG(MlpPolicy,env, reward_scale=0.01, gamma=0.99, buffer_size=10000,tensorboard_log="./simglucose_ddpg_tensorboard/")

    model.learn(total_timesteps=50000000, callback=[best_model_save_callback, checkpoint_callback])


if __name__ == "__main__":
    main()
