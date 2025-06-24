'''
定义所有用到的经验回放缓冲区类，包括常规的经验回放缓冲区、优先经验回放缓冲区、多步TD经验回放缓冲区等。
'''
from collections import deque
import numpy as np
import torch
from .base import *
#定义常规的经验回放缓冲区
class NormBuffer(BaseBuffer):
    def __init__(self, config):
        super().__init__(config)
        self.size = self.config['agent']['experience_replay']['buffer_size']
        self.memorys = deque(maxlen=self.size)
    
    def add(self, experience):

        self.memorys.append(experience)
    
    def sample(self, batch_size):
        indexs = np.random.choice(len(self.memorys), batch_size, replace=False)
        experiences = [self.memorys[idx] for idx in indexs]
        
        #逐一取出经验，并组装成所需的格式，s,a,r,s_t+1,done，都需要是N*D的格式
        states, actions, rewards, next_states, dones = self._split_experiences(experiences, batch_size)
        
        return_dict = {
            'batch_data':(states, actions, rewards, next_states, dones),
                }
        return return_dict #返回字典，便于兼容各类buffer

    def _split_experiences(self, experiences,batch_size):
        
        #将指定多个经验，拆分成指定大小的batch_size，用于并行采样
        states = torch.tensor( [experience[0] for experience in experiences]).reshape(batch_size, -1).to(self.device)
        actions = torch.tensor([ experience[1] for experience in experiences]).reshape(batch_size, -1).to(torch.int64).to(self.device)
        rewards = torch.tensor([experience[2] for experience in experiences]).reshape(batch_size, -1).to(self.device)
        next_states = torch.tensor([experience[3] for experience in experiences]).reshape(batch_size, -1).to(self.device)
        dones = torch.tensor( [experience[4] for experience in experiences]).reshape(batch_size, -1).to(torch.float).to(self.device)
        
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.memorys)


# 定义优先经验回放缓冲区
class PriorityBuffer(NormBuffer):
    def __init__(self, config):
        super().__init__(config)
        er_config = self.config['agent']['experience_replay']
        self.size = er_config['buffer_size']
        self.tree = SumTree(self.size)
        self.alpha = er_config['alpha']  # 优先级指数，控制优先级差异，为0时，优先级相同，为1时，完全按照TD误差抽样
        self.beta = er_config['beta']   # 重要性采样指数，用于校正偏差，为0时，权重相同，为1时，按照优先级缩放
        self.beta_increment = er_config['beta_increment']  # beta的增量，随着训练逐渐增加到1
        self.epsilon = 10**-8  # 防止优先级为0
        self.max_priority = 1.0  # 将最大优先级初始化为1
    
    def add(self, experience, TD_error=None):
        # 添加经验到缓冲区
        priority = self.max_priority if TD_error is None else \
                        (np.abs(TD_error) + self.epsilon) ** self.alpha
                        
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        # 采样一批经验
        experiences = [] 
        indices = []
        priorities = []
        weights = []
        segment = self.tree.total_priority() / batch_size
        
        # 增加beta值，最大为1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 计算最小优先级的权重
        # 加上10**-8，主要为了避免除0
        priorities = np.array(self.tree.tree[-self.tree.capacity:]) + 10**-8
        min_prob = np.min(priorities) / self.tree.total_priority()
        max_weight = (min_prob * self.tree.size) ** (-self.beta)
        
        for i in range(batch_size):
            # 在每个段中采样
            a, b = segment * i, segment * (i + 1)
            v = np.random.uniform(a, b)
            
            idx, priority, experience = self.tree.get_leaf(v)
            
            # 计算权重
            prob = priority / self.tree.total_priority() #优先级归一化
            weight = (prob * self.tree.size) ** (-self.beta)  #计算缩放权重
            weight = weight / max_weight  # 权重归一化
            
            experiences.append(experience)
            indices.append(idx)
            weights.append(weight)
        
        states, actions, rewards, next_states, dones = self._split_experiences(experiences, batch_size)
        
        return_dict={
            'batch_data':(states, actions, rewards, next_states, dones),
            'indices':indices,
            'weights':torch.tensor(np.array(weights)).to(self.device) #因为权重也会参与到前向传播，所以转换tensor并移到统一设备上
        }
        
        return return_dict
    
    def update_priorities(self, indices, TD_errors):
        # 更新优先级
        for idx, error in zip(indices, TD_errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    
    def __len__(self):
        return self.tree.size


