from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv

from train.env.simglucose_gym_env import T1DSimEnv
import glob
import os

from train.reward.custom_rewards import shaped_reward_around_normal_bg

list_of_files = glob.glob('./training_ws/*.zip')  # * means all if need specific format then *.csv
latest_saved_model = max(list_of_files, key=os.path.getctime)
print(latest_saved_model)


def main():
    vec_env_kwargs = {'start_method': 'spawn'}
    # env = make_vec_env(T1DSimEnv, n_envs=40, monitor_dir='./training_ws', vec_env_cls=SubprocVecEnv,
    #                   vec_env_kwargs=vec_env_kwargs)
    vec_env_kwargs = {'start_method': 'fork'}
    env_kwargs = {'reward_fun': shaped_reward_around_normal_bg}
    env = make_vec_env(T1DSimEnv, n_envs=32, vec_env_cls=SubprocVecEnv,
                       vec_env_kwargs=vec_env_kwargs, env_kwargs=env_kwargs)
    model = PPO2.load(latest_saved_model, env=env)

    observation = env.reset()

    env.training = False

    for t in range(1000):

        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        print(observation[0])
        print("Reward = {}".format(reward[0]))
        print("Action = {}".format(action[0]))
        # print('Info = {}'.format(info))


        if done[0]:
            print("Episode finished after {} timesteps".format(t + 1))
            break


if __name__ == '__main__':
    main()
