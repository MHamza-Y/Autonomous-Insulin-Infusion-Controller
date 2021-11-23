from simglucose.simulation.env import risk_diff as orig_risk_diff
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv

from train.env.simglucose_gym_env import T1DSimEnv, T1DDiscreteSimEnv, T1DAdultSimEnv, T1DAdultSimV2Env, T1DDiscreteEnv
import glob
import os

from train.reward.custom_rewards import shaped_reward_around_normal_bg, shaped_negative_reward_around_normal_bg, \
    smooth_reward, no_negativity, risk_diff, no_negativityV2

list_of_files = glob.glob('./training_ws/*.zip')  # * means all if need specific format then *.csv
latest_saved_model = max(list_of_files, key=os.path.getctime)
#latest_saved_model = 'training_ws/rl_model_1150976_steps.zip'

# latest_saved_model = 'training_ws/best_model/best_model.zip'

def main():
    vec_env_kwargs = {'start_method': 'spawn'}
    # env = make_vec_env(T1DSimEnv, n_envs=40, monitor_dir='./training_ws', vec_env_cls=SubprocVecEnv,
    #                   vec_env_kwargs=vec_env_kwargs)
    vec_env_kwargs = {'start_method': 'fork'}
    env_kwargs = {'reward_fun': no_negativityV2}
    env = make_vec_env(T1DAdultSimEnv, n_envs=1, vec_env_cls=SubprocVecEnv,
                       vec_env_kwargs=vec_env_kwargs, env_kwargs=env_kwargs)
    model = PPO2.load(latest_saved_model, env=env)

    observation = env.reset()

    env.training = False

    for t in range(10000):

        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)

        print('Meal = {},'.format(info[0]['meal']) + ' Obs = {},'.format(observation[0]) + ' Action = {}'.format(
            action[0]))
        print("Reward = {}".format(reward[0]))

        if done[0]:
            print("Episode finished after {} timesteps".format(t + 1))
            break


if __name__ == '__main__':
    main()
