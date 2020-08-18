#!/usr/bin/env python
# coding: utf-8

# In this notebook we will implement a DQN agent that uses the DQN network defined in [DuellingDoubleDQN.py](DuellingDoubleDQN.py) and the replay buffer defined in [prioritized replay buffer](PrioritizedReplayBuffer.py). Notice that we also need to implement Importance Sampling in order to properly use the prioritized replay buffer.

# In[1]:


import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from DuellingDoubleDQN import DuellingDQN
from PrioritizedReplayBuffer import PrioritizedReplay


# In[2]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


# In[3]:


class DuellingDoubleDQNAgent():
    """
    device = cpu or gpu
    num_agents: number of agents 
    im_height: height of input image
    im_width: width of input image
    obs_in_channels: number of input channels of the grid image
    conv_dim: number of channels after passing through 1st conv. layer of the DQN 
    kernel_size: kernel size of the conv. layers in the DQN
    n_action: number of discrete actions
    buffer_size: replay buffer size; default 2^20 ~ 10^6; For creating sum tree it is better give buffer_size in powers of 2
    roll_out: length of roll out for n-step bootstrap
    replay_batch_size: batch_size of replay 
    epsilon: exploration rate
    epsilon_decay_rate: rate by which to scale down epsilon after every few steps
    tau: parameter for soft update of the target network
    gamma: discount factor for discouted rewards
    update_interval: interval after which to update the network with new parameters
    """
    
    def __init__(self, device = device, num_agents=1, im_height = 464, im_width = 464, obs_in_channels=4, conv_dim = 32,  
                 kernel_size = 6, n_actions = 5, buffer_size = 2**20, roll_out = 5, replay_batch_size = 32,
                 lr = 1e-4, epsilon = 0.3, epsilon_decay_rate = 0.999, tau = 1e-3, gamma = 1, update_interval = 4):
        super().__init__()
        self.device = device
        self.num_agents = num_agents
        self.im_height = im_height
        self.im_width = im_width
        self.in_channels = obs_in_channels
        self.conv_dim = conv_dim
        self.kernel_size = kernel_size
        self.n_actions = n_actions
        self.buffer_size = buffer_size
        self.roll_out = roll_out
        self.replay_batch_size = replay_batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.tau = tau
        self.gamma = gamma # we want the train to find the shortest possible path to its destination
                           # for every time step it gets a reward of -1
                           # it makes sense to keep gamma = 1 
        self.update_every = update_interval        
                
        self.local_net = [DuellingDQN(obs_in_channels, conv_dim, kernel_size, n_actions) for _ in range(num_agents)]
        self.target_net = []
        self.optimizer = []
        
        for agent in range(num_agents):
            local = self.local_net[agent]
            target = DuellingDQN(obs_in_channels, conv_dim, kernel_size, n_actions)
            
            # copy the local networks parameters to the target network
            for local_param, target_param in zip(local.parameters(), target.parameters()):
                target_param.data.copy_(local_param.data)
            
            self.target_net.append(target)
            
            local = local.to(device)
            target = target.to(device)
            
            # set the optimizer for the local network
            optim = torch.optim.Adam(local.parameters(), lr = lr)
        
        # loss function to compare the Q-value of the local and the target network
        self.criterion = nn.MSELoss()
        
        # steps counter to keep track of steps passed between updates
        self.t_step = 0
        
        # need to fix this to store images as memories. 
        # for the time being using a dummy value for n_states
        n_states = 264
        self.memory = PrioritizedReplay(buffer_size, n_states, n_actions, roll_out, num_agents)
        
    def act(self, state):
        # function to produce an action from the DQN
        # convert states to a torch tensor and move to the device
        # unsqueeze at index 1 to convert the state for each agent into a batch of size 1
        state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        actions_list = []
        with torch.no_grad():
            for idx in range(self.num_agents):
                self.local_net[idx].eval()
                actionQ = self.local_net[idx](state).cpu().detach().numpy()[0]
                    
                # choose action with epsilon-greedy policy
                random_num = np.random.uniform()
                if random_num > self.epsilon:
                    action = np.argmax(actionQ)
                else:
                    action = np.random.randint(self.n_actions+1)
                    
                actions_list.append(action)
                self.local_net[idx].train()
        actions_array = np.array(actions_list)        
        return actions_array # each entry is the action for the corresponding agent
    
    
    def step(self, new_memories):
        
        # new memories is a list of n tuples
        # here n = roll_out
        # each tuple corresponds to a step in the roll_out
        # each tuples contains: state, actions_dict, all_rewards, next_state, done
        # here state and next_state are given by an image of the entire network
        self.memory.add(new_memories)
        
        # update the networks after every self.update_every steps
        # make sure to check that the replay_buffer has enough memories
        self.t_step = (self.t_step+1)%self.update_every
        if self.t_step == 0 and self.memory.__len__() > 2*self.replay_batch_size:
            self.learn()
            self.epsilon = max(self.epsilon_decay_rate*self.epsilon, 0.1 )
        
    
    
    def learn(self):
        print("learning")
        pass

