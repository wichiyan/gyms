
import torch
import os
from torch import nn
import numpy as np
from agents.base_agent import BaseAgent
from networks.dqn import *
from networks.q_table import *
from utils.schedulers import exploration_rate_scheduler
from buffers.buffers import *

class DQNAgent(BaseAgent):
    def __init__(self, env ,config, **kwargs):
        super().__init__(config,**kwargs)
        self.env = env
        self.config = config
        self.use_greedy = self._use_greedy()
        self._init_network()
        self._init_buffer()
        
        #TD配置项
        self.use_multi_step = self.config['agent'].get('use_multi_step', False)
        self.n_step = self.config['agent'].get('TD_steps', 1)
        self.TD_steps_buffer = deque(maxlen=self.n_step)
        
        #探索率
        self.explore_scheduler_train = exploration_rate_scheduler(**config['train'].get('explore_scheduler_train', {}))
        self.explore_scheduler_eval = exploration_rate_scheduler(**config['eval'].get('explore_scheduler_eval', {}))
        self.training = False

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
        else :
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
        
    #收集经验，并放到经验池中
    def collect_experience(self, state, action, reward, next_state, done):
        #将收集到的经验存储到经验池中，此处需要判断使用的是什么类型的经验池，如果是优先级，需要计算优先级后存储
        #1、首先根据是否启用多步TD，决定是否将经验转化为多步TD经验
        if self.use_multi_step:
            experience = self._get_multi_step_experiences(state,action,reward,next_state,done)
        else:
            experience = (state, action, reward, next_state, done)
        
        #2、如果经验不为空，将经验存储到经验池中
        if experience:
            #如果是普通经验池，则直接存储
            if self.buffer_type == 'norm':
                if experience:
                    self.buffer.add(experience)
            #如果是优先级经验回放，则需要计算优先级后存储
            elif self.buffer_type == 'priority':
                #首先使用当前网络，计算TD误差
                TD_error = self._get_TD_error(*experience) #返回数据形状为N*1
                #然后将TD误差和经验存储到经验池中
                self.buffer.add(experience,TD_error.item()) #此处因为batch是1，所以可以使用item直接取值 
            else:
                raise ValueError('buffer type must be norm or priority,got{self.buffer_type}')

    #更新策略网络    
    def update(self):
        #首先从经验池选择一批次经验数据
        #如果经验池中经验数量不足，则直接返回
        if len(self.buffer) < self.config['train']['batch_size']:
            return 0
        
        #从经验池中随机选择一批次经验数据
        #抽取出经验数据中的状态、动作、奖励、下一个状态、是否结束，每一个都是N*1形状
        samples = self.buffer.sample(self.config['train']['batch_size']) 
        states,actions,rewards,next_states,dones = samples['batch_data']

        # 计算当前Q值，如果是噪声网络，先重置噪声
        if self.network_type == 'dueling_noise' : self.Q_network.reset_noise()
        current_q_values = self.Q_network(states) #输出为N*A
        current_q_value = current_q_values.gather(index=actions,dim=1) #输出为N*1
        
        # 计算目标Q值
        with torch.no_grad():
            #如果是噪声网络，先重置噪声
            if self.network_type == 'dueling_noise' : self.Q_network.reset_noise()
            next_q_values = self.Q_network(next_states) #输出为N*A
            max_next_q_value = torch.max(next_q_values,dim=1,keepdim=True)[0] #输出为N*1
            discount_rate = self.config['train']['reward_discount_rate']
            target_q_value = rewards + (1 - dones) * discount_rate * max_next_q_value #输出为N*1
            
        # 计算损失，因为考虑需要对损失进行加权，所以先不聚合
        loss = self.criterion(current_q_value, target_q_value.float()) #输出为N*1
        TD_loss = loss.clone().detach().cpu() #输出为N*1
        
        #如果是优先级经验回放，则对损失进行加权
        if self.buffer_type == 'priority':
            #对损失进行加权
            weights = samples['weights'].reshape(-1,1) #形状变为N*1
            loss = (loss*weights).mean()
        else:
            loss = loss.mean()

        # 记录损失
        self.run_info['train']['episode_losses'].append(loss.item())
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #更新优先级，具体是否需要更新，下面方法内部自行处理
        self._update_priorities(samples,TD_loss)

        return loss.item()

    def _update_priorities(self,samples,TD_errors):
        #TD_error形状为N*1，需要变为N,
        if self.buffer_type == 'priority':
            self.buffer.update_priorities(samples['indices'],TD_errors.squeeze(dim=1))
       
    #将普通经验，转化为多步TD经验   
    def _get_multi_step_experiences(self,state,action,reward,next_state,done):
        #1、首先将经验池加入到n_step_buffer中
        self.TD_steps_buffer.append((state,action,reward,next_state,done))
        #2、如果n_step缓冲池中的经验数量足够，或者游戏已经结束，则将n_step缓冲池中的经验统一添加到sumtree缓冲池中
        if len(self.TD_steps_buffer) == self.n_step or done:
            #先取出来state,action,next_state
            state = self.TD_steps_buffer[0][0]
            action = self.TD_steps_buffer[0][1]
            next_state = self.TD_steps_buffer[-1][3]
            
            #然后计算多步TD奖励
            reward = self._get_multi_step_reward(self.TD_steps_buffer)
            
            #清空n_step缓冲池
            self.TD_steps_buffer.clear()
            
            #返回转换为多步TD的经验
            return (state,action,reward,next_state,done)
        
        #如果n_step缓冲池中的经验数量不足，则返回None
        else:
            return None
            
    
    #判断是否使用greedy策略
    def _use_greedy(self):
        if self.env.mode == 'train':
            return self.config['train']['explore_scheduler_train']['use_greedy']
        else:
            return self.config['eval']['explore_scheduler_eval']['use_greedy']
    
    #获取最大Q值对应动作
    def _get_max_action(self,state):
        state_tensor = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.Q_network(state_tensor)
        return torch.argmax(q_values).item()

    def _get_TD_error(self,state,action,reward,next_state,done):
        #首先使用当前网络，计算TD误差，需要支持单个批量计算
        #先判断数据是否都是tensor,如果不是则转换并移动到指定设备上
        #如果不是tensor，肯定是从环境收集的新的单个经验，从经验池采集的都会确保数据类型和形状
        if not isinstance(state,torch.Tensor):
            state,action,reward,next_state,done = \
                            self._trans_experience_to_tensor(state,action,reward,next_state,done) 
        
        #计算目标Q值                         
        with torch.no_grad():
            #如果是噪声网络，先重置噪声
            if self.network_type == 'dueling_noise' : self.Q_network.reset_noise()
            current_q_values = self.Q_network(state) #输出为N*A
            current_q_value = current_q_values.gather(index=action,dim=1) #输出为N*1
            max_next_q_value = torch.max(self.Q_network(next_state),dim=1,keepdim=True)[0]  #输出为N*1
            discount_rate = self.config['train']['reward_discount_rate']   
            
            #此处优化，考虑兼容使用多步TD的情况 
            target_q_value = reward + (1 - done) * (discount_rate**self.n_step) * max_next_q_value #输出为N*1
        
        # 计算TD误差
        TD_error = torch.abs(current_q_value - target_q_value)
        return TD_error
    
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

        self.criterion = nn.MSELoss(reduction='none')
        self.Q_network.to(self.device)

    #计算多步TD奖励
    def _get_multi_step_reward(self,TD_steps_buffer):
        #计算多步TD奖励
        discount_rate = self.config['train']['reward_discount_rate']
        rewards = np.array([ experience[2] for experience in TD_steps_buffer])
        discount_rates = np.array([ discount_rate**i for i in range(len(rewards))])
        multi_step_reward = float(np.sum(rewards * discount_rates))
        
        return multi_step_reward
            

    def _get_network(self):
        #兼容不同环境
        observe_space = self.env.observation_space
        state_size = observe_space.shape[0] if observe_space.shape else observe_space.n
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
        elif self.network_type == 'qtable':
            return QTable(state_size,action_size)
        elif self.network_type == 'qtable_embed':
            return QTableEmbedding(state_size,action_size)
        else:
            raise ValueError(f"Unknown network type: {self.network_type},\
                             can be dqn,dueling,dueling_noise,,qtable,qtable_embed")
    
    def _trans_experience_to_tensor(self, state,action,reward,next_state,done):
        
        state = torch.tensor(state,dtype=torch.float32).reshape(1,-1).to(self.device)
        action = torch.tensor(action,dtype=torch.int64).reshape(1,-1).to(self.device)
        reward = torch.tensor(reward,dtype=torch.float32).reshape(1,-1).to(self.device)
        next_state = torch.tensor(next_state,dtype=torch.float32).reshape(1,-1).to(self.device)
        done = torch.tensor(done,dtype=torch.float32).reshape(1,-1).to(self.device)
        
        return state,action,reward,next_state,done
        
    #根据配置文件，初始化buffer
    def _init_buffer(self):
        agent_config = self.config.get('agent', {})
        self.buffer_type = agent_config['experience_replay'].get('buffer_type', 'norm')

        if self.buffer_type == 'norm':
            self.buffer = NormBuffer(self.config)
        elif self.buffer_type == 'priority':
            self.buffer = PriorityBuffer(self.config)
    

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
    def update(self):
            #首先从经验池选择一批次经验数据
        #如果经验池中经验数量不足，则直接返回
        if len(self.buffer) < self.config['train']['batch_size']:
            return 0
        
        #从经验池中随机选择一批次经验数据
        #抽取出经验数据中的状态、动作、奖励、下一个状态、是否结束，每一个都是N*1形状
        samples = self.buffer.sample(self.config['train']['batch_size']) 
        states,actions,rewards,next_states,dones = samples['batch_data']

        # 使用Q网络计算q(s_t,a_t)，如果噪声网络，先重置噪声
        if self.network_type == 'dueling_noise' : self.Q_network.reset_noise()
        current_q_values = self.Q_network(states) #输出为N*A
        current_q_value = current_q_values.gather(index=actions,dim=1) #输出为N*1
        
        # 计算目标Q值
        with torch.no_grad():
            #1、先使用Q网络，算出s_t+1时最大Q值对应动作
            if self.network_type == 'dueling_noise' : self.Q_network.reset_noise()
            next_q_values_Q = self.Q_network(next_states) #输出为N*A
            max_action = torch.argmax(next_q_values_Q,dim=1,keepdim=True) #输出为N*1
            
            #2、然后使用目标网络，算出在s_t+1时，以上动作的Q值，如果是噪声网络，就先重置噪声
            if self.network_type == 'dueling_noise' : self.target_network.reset_noise()
            next_q_values_target = self.target_network(next_states) #输出为N*A
            max_next_q_value = next_q_values_target.gather(index=max_action,dim=1) #输出为N*1
            
            #3、计算TD目标
            discount_rate = self.config['train']['reward_discount_rate']
            target_q_value = rewards + (1 - dones) * discount_rate * max_next_q_value #输出为N*1
            
        # 计算损失，因为考虑需要对损失进行加权，所以先不聚合
        loss = self.criterion(current_q_value, target_q_value.float()) #输出为N*1
        TD_loss = loss.clone().detach().cpu() #输出为N*1
        
        #如果是优先级经验回放，则对损失进行加权
        if self.buffer_type == 'priority':
            #对损失进行加权
            weights = samples['weights'].reshape(-1,1) #形状变为N*1
            loss = (loss*weights).mean()
        else:
            loss = loss.mean()

        # 记录损失
        self.run_info['train']['episode_losses'].append(loss.item())
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #更新优先级，具体是否需要更新，下面方法内部自行处理
        self._update_priorities(samples,TD_loss)

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


