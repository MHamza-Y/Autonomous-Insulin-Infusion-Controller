import gym, torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts

from train.env.simglucose_gym_env import T1DSimEnv

task = T1DSimEnv
lr = 1e-4
gamma = 0.9
n_step = 128
eps_train, eps_test = 0.1, 0.05
epoch = 10
step_per_epoch = 1000
collect_per_step = 10
target_freq = 320
batch_size = 64
train_num, test_num = 8, 100
buffer_size = 20000
writer = SummaryWriter('log/dqn')

train_envs = ts.env.VectorEnv([lambda: gym.make(task) for _ in range(train_num)])
test_envs = ts.env.VectorEnv([lambda: gym.make(task) for _ in range(test_num)])

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Linear(np.prod(state_shape), 32), nn.ReLU(inplace=True),
            nn.Linear(32, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 32), nn.ReLU(inplace=True),
            nn.Linear(32, np.prod(action_shape))
        ])
    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float)
        batch = s.shape[0]
        logits = self.model(s.view(batch, -1))
        return logits, state

env = gym.make(task)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=lr)


policy = ts.policy.DQNPolicy(net, optim, gamma, n_step,
    use_target_network=True, target_update_freq=target_freq)
train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(buffer_size))
test_collector = ts.data.Collector(policy, test_envs)