import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import robosuite as suite
from secant.envs.robosuite import make_robosuite
from robosuite.controllers import load_controller_config
from utils import *

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
number_episode = int(1e4)
multi = 2
train_freq = 4
warm_start = 100
# env = gym.make('CartPole-v0')
# env = env.unwrapped
env = make_robosuite(
    task="NutAssemblyRound",
    mode="train",
    scene_id=2,
    image_width=600,
    image_height=600,
)

# # create an environment for policy learning from pixels
# env = suite.make(
#     env_name="Lift",
#     robots="UR5e",                          # load a Sawyer robot and a Panda robot
#     gripper_types="default",                # use default grippers per robot arm
#     controller_configs=controller_config,   # each arm is controlled using OSC
#     has_renderer=False,                     # no on-screen rendering
#     has_offscreen_renderer=True,            # off-screen rendering needed for image obs
#     control_freq=20,                        # 20 hz control for applied actions
#     horizon=200,                            # each episode terminates after 200 steps
#     use_object_obs=False,                   # don't provide object observations to agent
#     use_camera_obs=True,                   # provide image observations to agent
#     camera_names="agentview",               # use "agentview" camera for observations
#     camera_heights=84,                      # image height
#     camera_widths=84,                       # image width
#     reward_shaping=True,                    # use a dense reward signal for learning
# )
# N_ACTIONS = env.action_dim
# N_STATES = 84*84*3
# ENV_A_SHAPE = 0 #if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0] * 600 * 600
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(N_STATES, 50)
        # self.fc1.weight.data.normal_(0, 0.1)   # initialization
        # self.out = nn.Linear(50, N_ACTIONS)
        # self.out.weight.data.normal_(0, 0.1)   # initialization
        self.conv1 = nn.Conv2d(in_channels=N_STATES, out_channels=N_STATES*multi, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=N_STATES*multi, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.out = nn.Linear(16*600*600, N_ACTIONS)

    def forward(self, x):
        # x = self.fc1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x),0)
        # x = x.permute(0,3,1,2)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            # action = torch.max(actions_value, 1)[1].data.numpy()
            action = torch.softmax(actions_value, 1)[0].data.numpy()
            # action = list(action[0]) if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            # action = np.random.randint(0, N_ACTIONS)
            # action = [int(x) for x in str(action)]
            # action = np.array(list(action))
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            action = env.action_space.sample()
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    dqn = DQN()
    # buffer = ReplayBuffer(MEMORY_CAPACITY)
    i = 0
    print('\nCollecting experience...')
    for i_episode in range(1, number_episode + 1):
        s = env.reset()
        ep_r = 0
        while True:
            # env.render()
            a = dqn.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)
            i += 1
            # buffer.add(s, a, r, s_, done)

            # # modify the reward
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2

            dqn.store_transition(s.flatten(), a, r, s_.flatten())
            # if i >= warm_start and i % train_freq == 0:
            #     transitions = buffer.sample(BATCH_SIZE)
            #     dqn.learn()


            ep_r += r

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            if done:
                # reward, length = info['episode']['r'], info['episode']['l']
                # print(
                #     'Time steps so far: {}, episode so far: {}, '
                #     'episode reward: {:.4f}, episode length: {}'
                #     .format(i, i_episode, reward, length)
                # )
                break
            s = s_
