import torch 
from torch import nn
from abc import ABC, abstractmethod
import torch.nn.functional as F

#定义神经网络基类，主要实现公共方法，并定义公共接口
class Base_Network(nn.Module,ABC):
    @abstractmethod
    def __init__(self):
        pass
        
    @abstractmethod
    def forward(self, x):
        pass

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
            

#定义线性噪声层，用于噪声网络内，或者其他网络内

class NoisyLinear(nn.Module):
    ''''
    #公式是y = y=(W+W_noise​⊙ϵ^W)x+(b+b_noise​⊙ϵ^b)，其中W和b是经典线性层权重，W_noise和b_noise是噪声权重和偏置，ϵ^W和ϵ^b是噪声的缩放因子
    其中W_noise 由epsilon_in和epsilon_out笛卡尔积生成，严格按照原始论文实现
    '''
    def __init__(self, in_features, out_features, train=True,sigma_init=0.017):
        super().__init__()
        self.train = train
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init))

        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features)) #不参与梯度传播

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init))
        self.register_buffer("bias_epsilon", torch.zeros(out_features)) #不参与梯度传播

        self.reset_parameters()
        self.reset_noise()
        
    #初始化时，重置权重参数
    def reset_parameters(self):
        bound = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

    #每次前向传播时，需要重新生成噪声
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    #对随机噪声进行缩放
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, input):
        #只有在训练模式下，才会使用噪声参数
        if self.train:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)
    

