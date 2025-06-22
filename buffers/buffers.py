'''
定义所有用到的经验回放缓冲区类，包括常规的经验回放缓冲区、优先经验回放缓冲区、多步TD经验回放缓冲区等。
'''
from collections import deque
import numpy as np
import torch


from .base import BaseBuffer
#定义常规的经验回放缓冲区
class NormBuffer(BaseBuffer):
    def __init__(self, config):
        super().__init__(config)
        self.size = self.config['agent']['experience_replay']['buffer_size']
        self.memorys = deque(maxlen=self.size)
    
    def collect(self, experience):
        self.memorys.append(experience)
    
    def sample(self, batch_size):
        indexs = np.random.choice(len(self.memorys), batch_size, replace=False)
        samples = np.array(self.memorys)[indexs]
        batch_data = torch.from_numpy(samples).to(self.device) #转为tensor
        
        state = batch_data[:,0:1]
        action =  batch_data[:,1:2].to(torch.int64) #转为int64
        reward =  batch_data[:,2:3]
        next_state =  batch_data[:,3:4]
        done =  batch_data[:,4:5]
        return state,action,reward,next_state,done #返回s,a,r,s_t+1,done,每一个数据形状均为N*1
    
    def __len__(self):
        return len(self.memorys)
