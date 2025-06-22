import torch 
from torch import nn
from abc import ABC, abstractmethod
import torch.nn.functional as F
from networks.base import BaseNetwork
from utils.functions import get_device

#定义agent基类，主要实现公共方法，并定义公共接口
class BaseAgent(ABC):
    @abstractmethod
    def __init__(self,config,**kwargs):
        self.device = get_device(config)
    
    @abstractmethod
    def select_action(self, state):
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs):
        pass
        
    def train(self):
        #遍历所有网络属性并调用它们的train方法
        #确保所有网络都处于训练模式
        self.training = True
        for attr in self.__dict__:
            if isinstance(attr,BaseNetwork):
                attr.train()
                
    def eval(self):
        #遍历所有网络属性并调用它们的eval方法
        #确保所有网络都处于评估模式
        self.training = False
        for attr in self.__dict__:
            if isinstance(attr,BaseNetwork):
                attr.eval()