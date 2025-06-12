import torch 
from torch import nn
from .Base import Base_Network

#定义经典DQN网络结构
class DQN(Base_Network):
    def __init__(self, state_size, hidden_dim,action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, path, weights_only=False):
        if weights_only:
            torch.save(self.state_dict(), path)
        else:
            # Save the entire model including architecture
            torch.save(self, path)

    def load(self, path, weights_only=False):
        if weights_only:
            self.load_state_dict(torch.load(path))
        else:
            # Load the entire model including architecture
            loaded_model = torch.load(path)
            self.__dict__.update(loaded_model.__dict__)


class Dueling_DQN(nn.Module):
    def __init__(self, state_size, action_size,hidden_dim = 128):
        super().__init__()
        #共享部分
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
        )
        
        #状态价值分支
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  
        )   
        
        #优势函数分支
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)  
        )

    def forward(self, x):
        x = self.shared(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Dueling DQN的输出是状态价值和优势函数的组合
        # 这里使用平均优势函数来计算Q值，和理论公式不一致，因为实践表明使用平均优势函数可以更好地稳定训练
        # Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
        return value + advantage - advantage.mean()  
    
    
