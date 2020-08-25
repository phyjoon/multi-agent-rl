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
    alpha: TDerror's exponent when computing priorities of saved memories
    beta0: Initial value of the exponent for computing importance sampling weights
    """
    
    def __init__(self, device = device, num_agents=1, im_height = 464, im_width = 464, obs_in_channels=4, conv_dim = 32,  
                 kernel_size = 6, n_actions = 5, buffer_size = 2**20, roll_out = 5, replay_batch_size = 32,
                 lr = 1e-5, epsilon = 0.3, epsilon_decay_rate = 0.9999, tau = 1e-3, gamma = 1, update_interval = 4, 
                 alpha = 0.6, beta0 = 0.4):
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
        self.alpha = alpha
        self.beta0 = beta0
        # discounts to be applied at each step of roll_out
        self.discounts = torch.tensor([self.gamma**powr 
                                       for powr in range(self.roll_out)]).float().to(device)   
        self.update_every = update_interval        
                
        self.local_net = []
        self.target_net = []
        self.optimizer = []
        
        for agent in range(num_agents):
            local = DuellingDQN(obs_in_channels, conv_dim, kernel_size, n_actions)
            target = DuellingDQN(obs_in_channels, conv_dim, kernel_size, n_actions)
            
            # copy the local networks parameters to the target network
            for local_param, target_param in zip(local.parameters(), target.parameters()):
                target_param.data.copy_(local_param.data)
            
            local = local.to(device)
            self.local_net.append(local)
            
            target = target.to(device)
            self.target_net.append(target)
            
            # set the optimizer for the local network
            optim = torch.optim.Adam(self.local_net[-1].parameters(), lr = lr, betas = (0.5, 0.9999))
            self.optimizer.append(optim)
        
        # loss function to compare the Q-value of the local and the target network
        # The total loss has to be a weighted sum of the instance losses
        # The weights are obtained throught "importance sampling"
        # Thus it is better to keep reduction to be 'none' instead of the default value which
        # will return the average of all instance losses
        self.criterion = nn.MSELoss(reduction = 'none')
        # We will also be computing TDerrors for updating priorities
        self.TDErrors = nn.L1Loss(reduction = 'none')
        
        # steps counter to keep track of steps passed between updates
        self.t_step = 0
        
        # need to fix this to store images as memories. 
        # for the time being using a dummy value for n_states
        n_states = 264
        self.memory = PrioritizedReplay(buffer_size, n_states, n_actions, roll_out, num_agents, alpha)
        
        print("Created {} local networks".format(len(self.local_net)))
        print("Created {} target networks".format(len(self.target_net)))
        print("Created {} optimizers".format(len(self.optimizer)))
        
    def act(self, state):
        # function to produce an action from the DQN
        # convert states to a torch tensor and move to the device
        # unsqueeze at index 1 to convert the state for each agent into a batch of size 1
        state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        expected_state_shape = (1, self.in_channels, self.im_height, self.im_width)
        assert state.shape == expected_state_shape,        "Error: state's shape not same as expected. Expected shape {}, got {}".format(expected_state_shape, tuple(state.shape))
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
                    action = np.random.randint(self.n_actions)
                    
                actions_list.append(action)
                self.local_net[idx].train()
        actions_array = np.array(actions_list)        
        return actions_array # each entry is the action for the corresponding agent
    
    
    def step(self, new_memories):
        
        # new memories is a list of n tuples
        # here n = roll_out
        # each tuple corresponds to a step in the roll_out
        # each tuples contains: state, actions, all_rewards, next_state, done
        # here state and next_state are given by an image of the entire network
        # actions is the list of each agent's action in that step
        # thus actions[0] is the action of the 0-th agent in that step
        self.memory.add(new_memories)
        
        # update the networks after every self.update_every steps
        # make sure to check that the replay_buffer has enough memories
        self.t_step = (self.t_step+1)%self.update_every
        if self.t_step == 0 and self.memory.__len__() > 2*self.replay_batch_size:
            self.learn()
            self.epsilon = max(self.epsilon_decay_rate*self.epsilon, 0.01 )
            self.beta0 = min(self.beta0/self.epsilon_decay_rate, 1)
        
    
    
    def learn(self):
        # sample a batch of memories from the replay buffer
        batch, batch_idxs, priorities = self.memory.sample(self.replay_batch_size)
        # for a roll_out of n-steps, the batch has shape: (self.replay_batch_size, roll_out, 5)
        # The last dimension corresponds to shape of the tuple (state, action, all_rewards, next_state, done) for each step
        
        #in_states = np.stack(batch[:,0,0])
        in_states = torch.stack(list(map(lambda mem: torch.from_numpy(mem[0][0]), batch))).float().to(device)
        expected_in_shape = (self.replay_batch_size, self.in_channels, self.im_height, self.im_width)
        assert in_states.shape == expected_in_shape ,        "Error: shape of in_states is not same as expected. Expected shape: {}, got {}".format(expected_in_shape, tuple(in_states.shape))
        #in_states = torch.from_numpy(in_states).float().to(device)
        
        #actions0 = np.stack(batch[:,0,1])
        actions0 = torch.stack(list(map(lambda mem: torch.from_numpy(mem[0][1]), batch))).to(device)
        expected_actions_shape = (self.replay_batch_size, self.num_agents)
        assert actions0.shape == expected_actions_shape,         "Error: shape of actions0 not same as expected. Expected shape: {}, got {}".format(expected_actions_shape, tuple(actions0.shape))
        #actions0 = torch.from_numpy(actions0).to(device)
        
        # rewards for all the steps in the roll_out for all the agents
        #rewards = np.array(batch[:,:,2].tolist()) 
        rewards = torch.tensor(list(map(lambda mem: list(map(lambda tup: tup[2], mem)), batch))).float().to(device)
        expected_rewards_shape = (self.replay_batch_size, self.roll_out, self.num_agents)
        assert rewards.shape == expected_rewards_shape,         "Error: shape of rewards not same as expected. Expected shape: {}, got {}".format(expected_rewards_shape, tuple(rewards.shape))
        #rewards = torch.from_numpy(rewards).float().to(device)
        
        #fin_states = np.stack(batch[:,-1,3])
        fin_states = torch.stack(list(map(lambda mem: torch.from_numpy(mem[-1][3]), batch))).float().to(device)
        expected_fin_shape = (self.replay_batch_size, self.in_channels, self.im_height, self.im_width)
        assert fin_states.shape == expected_fin_shape ,        "Error: shape of fin_states is not same as expected. Expected shape: {}, got {}".format(expected_fin_shape, tuple(fin_states.shape))
        #fin_states = torch.from_numpy(fin_states).float().to(device)
        
        # each agent's done for the last step of the roll_out
        #dones = np.stack(batch[:,:,4].tolist())
        #expected_dones_shape = (self.replay_batch_size, self.roll_out, self.num_agents)
        #dones = np.stack(batch[:,-1,4].tolist()) 
        dones = torch.tensor(list(map(lambda mem: mem[-1][4], batch))).float().to(device)
        expected_dones_shape = (self.replay_batch_size, self.num_agents)
        assert dones.shape == expected_dones_shape,         "Error: shape of dones not same as expected. Expected shape: {}, got {}".format(expected_dones_shape, tuple(dones.shape))
        #dones = torch.from_numpy(dones).float().to(device)
        
        # compute the accumalated discounted reward over the roll_out period
        discounted_rewards = torch.matmul(self.discounts, rewards)
        expected_discounted_rew_shape = (self.replay_batch_size, self.num_agents)
        assert discounted_rewards.shape == expected_discounted_rew_shape,         "Error: shape of discounted_rewards not same as expected. Expected shape: {}, got {}".format(expected_discounted_rew_shape, tuple(discounted_rewards.shape))
        
        
        for idx in range(self.num_agents):
            with torch.no_grad():
                # need to choose nextQ using the local network's q-values
                self.local_net[idx].eval()
                nextActions = torch.max(self.local_net[idx](fin_states).detach(), axis = 1).indices.view(-1,1)
                self.local_net[idx].train()
                self.target_net[idx].eval()
                nextQ = self.target_net[idx](fin_states).detach().gather(1, nextActions)
                self.target_net[idx].train() 
                nextQ*=(1-dones[:, idx].view(-1,1))
                targetQ = discounted_rewards[:,idx].view(-1,1) + (self.gamma**self.roll_out)*nextQ
            
            
            # local agent's Q-value
            agent_actions = actions0[:, idx].view(-1,1)
            localQ = self.local_net[idx](in_states).gather(1, agent_actions)
            
            with torch.no_grad():
                # TD-errors
                TDerrors = torch.clamp(self.TDErrors(targetQ.detach(), localQ.detach()).view(-1), -5, 5)
                expected_TDerror_shape = (self.replay_batch_size,)
                assert TDerrors.shape == expected_TDerror_shape,                "Error: shape of TDerrors is not same as expected. Expected shape: {}, got {}".format(expected_TDerror_shape, TDerrors.shape)
                TDerrors = TDerrors.tolist()
                
                # update priorities according to the TDerrors
                self.memory.update_priority(TDerrors, batch_idxs)
                
                # compute importance sampling weights
                priorities_sum = self.memory.priority_tree.get_value(0)
                priorities = torch.from_numpy(priorities).float().to(device)
                assert priorities.shape == (self.replay_batch_size, ),                 "Error: shape of priorities is not same as expected. Expected shape: {}, got {}".format((batch_size,), priorities.shape)
                
                replay_probs = priorities/priorities_sum
                imp_sampling_weights = torch.pow(self.buffer_size*replay_probs.detach(), -self.beta0)
                max_weight = torch.max(imp_sampling_weights)
                imp_sampling_weights = (imp_sampling_weights/max_weight).float().to(device)
            
            losses = self.criterion(targetQ, localQ).view(-1)
            assert imp_sampling_weights.shape == losses.shape,            "Error: imp_sampling_weights and losses don't have the same shape"
            losses = imp_sampling_weights.detach()*losses
            assert losses.shape == (self.replay_batch_size, ),             "Error: shape of losses is not same as expected. Expected shape: {}, got {}".format((batch_size,), losses.shape)
            
            self.optimizer[idx].zero_grad()
            loss = losses.mean()
            loss.backward()
            self.optimizer[idx].step()
            
        
        # now apply soft-updates to the target network
        self.soft_updates()
            
        return 
    
    
    def soft_updates(self):
        
        with torch.no_grad():
            for idx in range(self.num_agents):
                for target_params, local_params in zip(self.target_net[idx].parameters(), 
                                                   self.local_net[idx].parameters()):
                    updates = self.tau*local_params.data + (1.0-self.tau)*target_params.data
                    target_params.data.copy_(updates)
    
