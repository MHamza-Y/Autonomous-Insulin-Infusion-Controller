from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpLnLstmPolicy

from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.bench.monitor import Monitor

from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from train.env.simglucose_gym_env import T1DSimEnv


model = PPO2.load('training_ws/rl_model_4775936_steps.zip')
env  = T1DSimEnv()
env.reset()
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