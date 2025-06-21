import torch 
from torch import nn
from networks.base import BaseNetwork

#定义经典的Q-table网络，还是基于pytorch实现
class QTable(BaseNetwork):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.table = nn.Parameter(torch.randn(state_size,action_size)/500)

    def forward(self, state):
        #确保state是整数型，因为是对表格进行索引
        state = state.to(torch.int) #将输入state，由形状N,1，变为N
        out = self.table[state].squeeze(dim=1)  
        return out   #输出为N，A

#定义基于embedding的Q-table实现
#将离散的状态，通过嵌入的方式连续化，然后输入深度网络内
class QTableEmbedding(BaseNetwork):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.embed = nn.Embedding(state_size,32)
        self.fc1 = nn.Linear(32,128)
        self.fc2 = nn.Linear(128,action_size)

    def forward(self, state):
        #确保state是整数型，因为是对表格进行索引
        state = state.to(torch.int).squeeze(dim=1) #输入为N*1，转换为N 
        embed = self.embed(state) #输出为N*D
        out = torch.relu( self.fc1(embed) )
        out = self.fc2(out) 
        
        return out  #输出为N*A
