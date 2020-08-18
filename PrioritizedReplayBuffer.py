import numpy as np
from operator import itemgetter

class sum_tree():
    def __init__(self, num_memories):
        self.num_memories = num_memories
        self.num_levels = int(np.ceil(np.log2(num_memories)+1))
        self.num_vertices = int(2**self.num_levels-1)
        self.tree_array = np.zeros(shape = self.num_vertices) # the 0-th entry is the root; children of i: 2i+1, 2i+2
        
    def get_children(self, i):
        # get the values of the children of the i-th node
        try:
            return [self.tree_array[2*i+1], self.tree_array[2*i+2]]
        except:
            return [None, None]
    
    def get_parent(self, i):
        # get the value of the parent of the i-th node
        return self.tree_array[(i-1)//2]
    
    def get_value(self, i):
        # get the value of i-th node
        return self.tree_array[i]
    
    def update_val(self, i, val):
        # update the value of the i-th node
        # This will also require us to update the value of its parent in order to maintian the sum-tree property
        self.tree_array[i] = val
        if not i==0:
            parent = (i-1)//2
            new_parent_val = sum(self.get_children(parent))
            self.update_val(parent, new_parent_val)        
            
    def get_sample_id(self, priority, current_index = 0):
        # get a sample corresponding to the input priority by traversing the tree starting from node at current_index
        
        # print(priority, current_index)
        if priority > self.get_value(current_index):
            raise ValueError("priority should be less than value of current index")
            
        left_c, right_c = self.get_children(current_index)
        
        if left_c == None:
            # we have reached the leaf node
            return current_index
        
        if priority <= left_c:
            sample_id = self.get_sample_id(priority, 2*current_index+1)
            return sample_id
        else:    
            sample_id = self.get_sample_id(priority-left_c, 2*current_index+2)
            return sample_id
        
    def max_leaf_value(self):
        # get the maximum value amongsts all the leaves
        return max(self.tree_array[2**(self.num_levels-2)+1:])
    
    
    
    
    
class PrioritizedReplay():
    def __init__(self, buffer_size, n_states, n_actions, roll_out, n_agents, alpha = 0.6, epsilon = 0.0001):
        self.memory = [None]*buffer_size
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.roll_out = roll_out # roll_out = 1 corresponds to a single step
        self.alpha = alpha # this is the exponent \alpha in eq.1 of the Prioritized Replay paper 
        self.epsilon = epsilon # this the epsilon  that is added to priorities to avoid edge-case issues
                               # see the discussion below eq. 1 in Prioritized Replay paper
        
        # length of an array containg a single memory of any one player
        self.experience_length = 2*n_states+n_actions+roll_out+1 
        
        # num of memories added to the buffer thus far
        self.memory_ctr = 0
        
        # index the in memory where the next experience is to be added
        # runs from 0 to buffer_size-1 after which it resets to zero
        self.new_exp_idx = 0
        
        # sum tree for the priorities
        self.priority_tree = sum_tree(buffer_size)
        self.first_leaf_idx = 2**(self.priority_tree.num_levels-1)-1 # index of the first leaf in priority_tree.tree_array 
        
    def add(self, experience_list):
        # add a new experience to the memory
        # the list consists of (n-1)-tuples of state, action, reward, done and the n-state
        # here n is the roll_out length
        self.memory[self.new_exp_idx] = experience_list
        
        # get the maximal priority of all experiences in the buffer
        priority = max([*self.priority_tree.tree_array[self.first_leaf_idx:], self.epsilon])
        
        # now update the priority in the priority_tree
        tree_index = self.new_exp_idx + self.first_leaf_idx
        self.priority_tree.update_val(tree_index, priority)
        
        # move the new experience index to the next position
        self.new_exp_idx = (self.new_exp_idx+1)%self.buffer_size
        # update the memory counter
        self.memory_ctr = min(self.memory_ctr+1, self.buffer_size)
    
    def update_priority(self, TDErrors, experience_idxs):
        # *********** To Implement **************
        # update the priorities of the given experiences
        # priority = abs(TDError)**self.alpha + self.epsilon
        # experience_idxs are the indices of the experiences in self.memory whose priority has to be updated
        # TDErrors are the latest TDErrors of those experiences
        for TDError, idx in zip(TDErrors, experience_idxs):
            priority = abs(TDError)**self.alpha + self.epsilon
            # now update the priority in the priority_tree
            tree_index = idx + self.first_leaf_idx
            self.priority_tree.update_val(tree_index, priority)
            
    def sample(self, batch_size):
        
        # get sum of all priorities 
        priority_sum = self.priority_tree.get_value(0)
        sample_priorities = (priority_sum)*np.random.random(batch_size)
        
        
        #batch_idxs = np.array(list(map(lambda val: 
        #                              self.priority_tree.get_sample_id(val) - self.first_leaf_idx, sample_priorities)))
        batch_idxs = list(map(lambda val: self.priority_tree.get_sample_id(val) - self.first_leaf_idx, sample_priorities))
        
        batch = np.stack(list(itemgetter(*batch_idxs)(self.memory)))
        
        priorities = np.array(list(map(lambda batch_id: 
                                      self.priority_tree.get_value(batch_id + self.first_leaf_idx) , batch_idxs)))
        
        return  batch, batch_idxs, priorities  
        
        
        
    def __len__(self):
        return self.memory_ctr