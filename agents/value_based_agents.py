
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
        self._init_network()
        self.explore_shecduler_train = exploration_rate_scheduler(**config.get('explore_shedule_train', {}))
        self.explore_shecduler_eval = exploration_rate_scheduler(**config.get('explore_shedule_eval', {}))

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
        
        #根据模型是在训练还是测试阶段，使用不同的探索机制
        if self.Q_network.training:
            self.explore_rate = self.explore_shecduler_train.get_exploration_rate(episode,episodes)
        else :
            self.explore_rate = self.explore_shecduler_train.explore_shecduler_eval(episode,episodes)
        
        #随机选择动作
        if np.random.rand() < self.explore_rate:
            return self.env.action_space.sample()  # 探索
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.Q_network(state_tensor)
            return torch.argmax(q_values).item()
        
    #更新策略网络    
    def update(self, state, action, reward, next_state, done):
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action).to(self.device)
        reward_tensor = torch.tensor(reward).to(self.device)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(self.device)

        # 计算当前Q值
        current_q_values = self.Q_network(state_tensor)
        current_q_value = current_q_values[0][action_tensor]
        
        # 计算目标Q值
        with torch.no_grad():
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
    
    #根据网络类型和状态、动作空间大小设置网络、优化器及损失函数
    def _init_network(self):
        #设置网络类型
        agent_config = self.config.get('agent', {})
        network_type = agent_config.get('network_type', 'dqn')
        hidden_dim = agent_config.get('hidden_dim', 128)
        lr = self.config['train'].get('lr', 0.001)
        
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        if network_type == 'dqn':
            self.Q_network = DQN(state_size, action_size, hidden_dim=hidden_dim)
        elif network_type == 'dueling':
            self.Q_network = DuelingDQN(state_size, action_size, hidden_dim=hidden_dim)
        elif network_type == 'dueling_noise':
            self.Q_network = DuelingNoiseDQN(state_size, action_size, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown network type: {network_type},can be dqn,dueling,dueling_noise")
        
        #设置优化器和损失函数
        optim_name = agent_config.get('optimizer', 'Adam')
        self.optimizer = getattr(torch.optim,optim_name)\
                                    (self.Q_network.parameters(), lr=lr)

        self.criterion = nn.MSELoss()
        self.Q_network.to(self.device)
        
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

