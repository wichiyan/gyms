
import torch
import os
from torch import nn
import numpy as np
from .base_agent import BaseAgent
from networks.dqn import DQN, DuelingDQN, DuelingNoiseDQN
from utils.schedulers import exploration_rate_scheduler

class DQNAgent(BaseAgent):
    def __init__(self, env ,config, **kwargs):
        super().__init__(config,**kwargs)
        self.env = env
        self.config = config
        self.use_greedy = self._use_greedy()
        self._init_network()
        self.explore_scheduler_train = exploration_rate_scheduler(**config.get('explore_scheduler_train', {}))
        self.explore_scheduler_eval = exploration_rate_scheduler(**config.get('explore_scheduler_eval', {}))

        #记录agent训练和验证时的信息
        self.run_info = {
            'train':{
                'episode_losses':[],
            },
            'eval':{
                
            }
        }
        
    #选择动作
    def select_action(self, state ,episode,episodes):
        #将episode和episodes信息记录到agent，主要用于后续监控日志记录
        self.episode = episode
        self.episodes = episodes
        
        #如果不使用greedy策略，则直接将探索率设置为0
        if not self.use_greedy:
            self.explore_rate = 0
        
        #根据模型是在训练还是测试阶段，使用不同的探索机制
        if self.Q_network.training:
            self.explore_rate = self.explore_scheduler_train.get_exploration_rate(episode,episodes)
        else :
            self.explore_rate = self.explore_scheduler_eval.get_exploration_rate(episode,episodes)
        
        #随机选择动作
        if np.random.rand() < self.explore_rate:
            return self.env.action_space.sample()  # 探索
        else:
            return self._get_max_action(state)
        
    #更新策略网络    
    def update(self, state, action, reward, next_state, done):
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action).to(self.device)
        reward_tensor = torch.tensor(reward).to(self.device)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(self.device)

        # 计算当前Q值，如果是噪声网络，先重置噪声
        if self.network_type == 'dueling_noise' : self.Q_network.reset_noise()
        current_q_values = self.Q_network(state_tensor)
        current_q_value = current_q_values[0][action_tensor]
        
        # 计算目标Q值
        with torch.no_grad():
            #如果是噪声网络，先重置噪声
            if self.network_type == 'dueling_noise' : self.Q_network.reset_noise()
            next_q_values = self.Q_network(next_state_tensor)
            max_next_q_value = torch.max(next_q_values).item()
            target_q_value = reward_tensor + (1 - done_tensor) * 0.99 * max_next_q_value
        # 计算损失
        loss = self.criterion(current_q_value, target_q_value.float())

        # 记录损失
        self.run_info['train']['episode_losses'].append(loss.item())
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return loss.item()

    #判断是否使用greedy策略
    def _use_greedy(self):
        if self.training:
            return self.config['train']['explore_scheduler_train']['use_greedy']
        else:
            return self.config['eval']['explore_scheduler_eval']['use_greedy']
    
    #获取最大Q值对应动作
    def _get_max_action(self,state):
        state_tensor = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.Q_network(state_tensor)
        return torch.argmax(q_values).item() 

    
    #根据网络类型和状态、动作空间大小设置网络、优化器及损失函数
    def _init_network(self):
        #设置网络类型
        agent_config = self.config.get('agent', {})
        lr = self.config['train'].get('lr', 0.001)
        self.Q_network = self._get_network()
        
        #设置优化器和损失函数
        optim_name = agent_config.get('optimizer', 'Adam')
        self.optimizer = getattr(torch.optim,optim_name)\
                                    (self.Q_network.parameters(), lr=lr)

        self.criterion = nn.MSELoss()
        self.Q_network.to(self.device)

    def _get_network(self):
        
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        agent_config = self.config.get('agent', {})
        self.network_type = agent_config.get('network_type', 'dqn')
        hidden_dim = agent_config.get('hidden_dim', 128)
        
        if self.network_type == 'dqn':
            return DQN(state_size, action_size, hidden_dim=hidden_dim)
        elif self.network_type == 'dueling':
            return DuelingDQN(state_size, action_size, hidden_dim=hidden_dim)
        elif self.network_type == 'dueling_noise':
            return DuelingNoiseDQN(state_size, action_size, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown network type: {self.network_type},can be dqn,dueling,dueling_noise")
        
        
    def save_network(self, path, weights_only=True):
        """
        Save the current state of the Q-network to a file.
        
        Args:
            path (str): The file path where the network will be saved.
        """
        self.Q_network.save(path, weights_only)

    def load_network(self, path, weights_only=True):
        """
        Load the Q-network from a file.
        
        Args:
            path (str): The file path from which the network will be loaded.
        """

        self.Q_network.load(path,weights_only)


class DoubleDQNAgent(DQNAgent):

    #更新策略网络    
    def update(self, state, action, reward, next_state, done):
        #先将数据转换成tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action).to(self.device)
        reward_tensor = torch.tensor(reward).to(self.device)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(self.device)

        # 使用Q网络计算q(s_t,a_t)，如果噪声网络，先重置噪声
        if self.network_type == 'dueling_noise' : self.Q_network.reset_noise()
        current_q_values = self.Q_network(state_tensor)
        current_q_value = current_q_values[0][action_tensor]
        
        # 计算目标Q值
        with torch.no_grad():
            #先使用Q网络，算出s_t+1时最大动作
            if self.network_type == 'dueling_noise' : self.Q_network.reset_noise()
            max_action = self._get_max_action(next_state_tensor)
            
            #然后使用目标网络，算出在s_t+1时，以上动作的Q值，如果是噪声网络，就先重置噪声
            if self.network_type == 'dueling_noise' : self.target_network.reset_noise()
            next_q_values = self.target_network(next_state_tensor)
            max_next_q_value = next_q_values[:,max_action].item()
            
            #计算TD目标
            target_q_value = reward_tensor + (1 - done_tensor) * 0.99 * max_next_q_value
            
        # 计算损失
        loss = self.criterion(current_q_value, target_q_value.float())

        # 记录损失
        self.run_info['train']['episode_losses'].append(loss.item())
        
        # 更新Q网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #每隔指定episode，更新目标网络
        update_freq = self.config['train']['target_update_every_episode']
        if self.episode % update_freq ==0:
            self._soft_update_target_network()
        
        return loss.item()
    
    
    #根据网络类型和状态、动作空间大小设置网络、优化器及损失函数
    def _init_network(self):
        #设置网络类型
        agent_config = self.config.get('agent', {})
        lr = self.config['train'].get('lr', 0.001)

        #初始化网络，并将目标网络和Q网络参数设置一样
        self.Q_network = self._get_network()
        self.target_network = self._get_network()
        self.target_network.load_state_dict(  self.Q_network.state_dict()  )
        
        #设置优化器和损失函数，并将网络移到指定设备
        optim_name = agent_config.get('optimizer', 'Adam')
        self.optimizer = getattr(torch.optim,optim_name)\
                                    (self.Q_network.parameters(), lr=lr)

        self.criterion = nn.MSELoss()
        self.Q_network.to(self.device)
        self.target_network.to(self.device)

    #软更新目标网络参数
    def _soft_update_target_network(self):
        tau = self.config['train']['tau']
        for target_param, param in zip(self.target_network.parameters(), self.Q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self):
        #重载积累的train方法，因为双Q网络，只需要让Q网络处于训练状态，target网络不需要
        self.training = True
        self.Q_network.train()
        self.target_network.eval()


