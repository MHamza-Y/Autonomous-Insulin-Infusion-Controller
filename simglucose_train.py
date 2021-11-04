import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from save_on_best_result_callback import SaveOnBestTrainingRewardCallback

from simglucose.envs.simglucose_gym_env import T1DSimEnv


register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002'}
)


save_folder = 'training_ws/'
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_folder,
                                         name_prefix="rl_model")
best_model_save_callback = SaveOnBestTrainingRewardCallback(check_freq=2110, log_dir=save_folder)

env = gym.make('simglucose-adolescent2-v0')

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./simglucose_ppo_tensorboard/")
model.learn(total_timesteps=1000000, callback=[best_model_save_callback, checkpoint_callback])

observation = env.reset()
for t in range(1000):
    env.render(mode='human')
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation)
    print("Reward = {}".format(reward))
    print("Action = {}".format(action))
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break

