from stable_baselines3 import PPO

from train.env.simglucose_gym_env import T1DSimEnv, T1DSimDiffEnv
import glob
import os

from train.reward.custom_rewards import custom_reward, shaped_reward_around_normal_bg, \
    shaped_negative_reward_around_normal_bg

list_of_files = glob.glob('./training_ws/*.zip')  # * means all if need specific format then *.csv
latest_saved_model = max(list_of_files, key=os.path.getctime)
print(latest_saved_model)


def main():
    vec_env_kwargs = {'start_method': 'spawn'}
    # env = make_vec_env(T1DSimEnv, n_envs=40, monitor_dir='./training_ws', vec_env_cls=SubprocVecEnv,
    #                   vec_env_kwargs=vec_env_kwargs)
    env = T1DSimDiffEnv(patient_name='adult#004',reward_fun=shaped_negative_reward_around_normal_bg)
    model = PPO.load(latest_saved_model, env=env)
    env = env
    observation = env.reset()


    for t in range(1000):

        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        print('Obs = {}'.format(observation) + ' Action = {}'.format(action))
        print("Reward = {}".format(reward))

        env.render(mode='human')
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break


if __name__ == '__main__':
    main()
