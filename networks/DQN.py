import torch 
from torch import nn
from .base import BaseNetwork
from .base import NoisyLinear

#定义经典DQN网络结构
class DQN(BaseNetwork):
    def __init__(self, state_size, action_size,hidden_dim = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DuelingDQN(BaseNetwork):
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


class DuelingNoiseDQN(BaseNetwork):
    #使用NoisyLinear层替代经典线性层，构建Dueling DQN网络结构
    def __init__(self, state_size, action_size,hidden_dim = 128):
        super().__init__()
        #共享部分
        self.shared = nn.Sequential(
            NoisyLinear(state_size, hidden_dim),
            nn.ReLU(),
        )
        
        #状态价值分支
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, 64),
            nn.ReLU(),
            NoisyLinear(64, 1)  
        )   
        
        #优势函数分支
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, 64),
            nn.ReLU(),
            NoisyLinear(64, action_size)  
        )

    def forward(self, x):
        x = self.shared(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Dueling DQN的输出是状态价值和优势函数的组合
        # 这里使用平均优势函数来计算Q值，和理论公式不一致，因为实践表明使用平均优势函数可以更好地稳定训练
        # Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
        return value + advantage - advantage.mean()  
    
    def reset_noise(self):
        # 重置所有NoisyLinear层的噪声
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
    

if __name__ == "__main__":
    # 测试Dueling_DQN网络结构
    state_size = 4  # 假设状态空间大小为4
    action_size = 2  # 假设动作空间大小为2
    model = DuelingNoiseDQN(state_size, action_size)
    
    # 测试输入
    test_input = torch.randn(1, state_size)  # 随机生成一个状态输入
    output = model(test_input)
    
    print("Output shape:", output.shape)  # 输出形状应该是 (1, action_size)