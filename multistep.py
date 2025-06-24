import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

# 定义NoisyLinear层，用于噪声网络
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        bound = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

# 定义Dueling DQN网络结构
class DuelingNoiseDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=128):
        super().__init__()
        # 共享部分
        self.shared = nn.Sequential(
            NoisyLinear(state_size, hidden_dim),
            nn.ReLU(),
        )
        
        # 状态价值分支
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, 64),
            nn.ReLU(),
            NoisyLinear(64, 1)  
        )   
        
        # 优势函数分支
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
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
    def reset_noise(self):
        # 重置所有NoisyLinear层的噪声
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

# 定义多步经验回放缓冲区
class MultiStepReplayBuffer:
    def __init__(self, capacity, n_steps, gamma):
        self.capacity = capacity  # 缓冲区容量
        self.buffer = deque(maxlen=capacity)  # 使用deque实现缓冲区
        self.n_steps = n_steps  # 多步TD的步数
        self.gamma = gamma  # 折扣因子
        self.n_step_buffer = deque(maxlen=n_steps)  # 用于存储n步经验的临时缓冲区
    
    def add(self, state, action, reward, next_state, done):
        # 添加经验到n步缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # 如果n步缓冲区未满，或者当前经验不是终止状态，则不进行n步计算
        if len(self.n_step_buffer) < self.n_steps and not done:
            return
        
        # 计算n步回报
        state, action = self.n_step_buffer[0][:2]  # 获取初始状态和动作
        
        # 计算n步回报和最终状态
        n_step_reward = 0
        next_state = None
        done = False
        
        for i, (_, _, r, next_s, d) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                done = True
                next_state = next_s
                break
            next_state = next_s
        
        # 将n步经验添加到主缓冲区
        self.buffer.append((state, action, n_step_reward, next_state, done))
        
        # 如果当前经验是终止状态，清空n步缓冲区
        if done:
            self.n_step_buffer.clear()
    
    def sample(self, batch_size):
        # 从缓冲区中随机采样一批经验
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为numpy数组或tensor
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# 定义多步TD DQN智能体
class MultiStepDQNAgent:
    def __init__(self, state_size, action_size, n_steps=3, gamma=0.99, lr=0.001, buffer_size=100000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.n_steps = n_steps  # 多步TD的步数
        self.gamma = gamma  # 折扣因子
        self.batch_size = batch_size  # 批量大小
        
        # 创建在线网络和目标网络
        self.online_network = DuelingNoiseDQN(state_size, action_size)
        self.target_network = DuelingNoiseDQN(state_size, action_size)
        self.update_target_network()  # 初始化目标网络
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=lr)
        
        # 创建多步经验回放缓冲区
        self.memory = MultiStepReplayBuffer(buffer_size, n_steps, gamma)
        
        # 训练参数
        self.exploration_rate = 1.0  # 初始探索率
        self.exploration_rate_decay = 0.995  # 探索率衰减
        self.exploration_rate_min = 0.01  # 最小探索率
        self.target_update_frequency = 1000  # 目标网络更新频率
        self.train_frequency = 4  # 训练频率
        self.step_counter = 0  # 步数计数器
        
        # 记录训练过程
        self.loss_history = []  # 损失历史
        self.reward_history = []  # 奖励历史
    
    def update_target_network(self):
        # 更新目标网络
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def select_action(self, state):
        # 选择动作，使用ε-greedy策略
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)
        
        # 使用在线网络选择动作
        self.online_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            self.online_network.reset_noise()  # 重置噪声
            q_values = self.online_network(state_tensor)
            action = torch.argmax(q_values).item()
        self.online_network.train()
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        # 存储经验到缓冲区
        self.memory.add(state, action, reward, next_state, done)
    
    def train(self):
        # 如果缓冲区中的经验不足，则不进行训练
        if len(self.memory) < self.batch_size:
            return 0
        
        # 从缓冲区中采样一批经验
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 计算目标Q值
        self.target_network.eval()
        with torch.no_grad():
            # 使用Double DQN：在线网络选择动作，目标网络计算Q值
            self.online_network.reset_noise()
            online_next_q_values = self.online_network(next_states)
            next_actions = torch.argmax(online_next_q_values, dim=1)
            
            self.target_network.reset_noise()
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # 计算目标Q值：r + gamma^n * max(Q(s', a'))
            # 注意：对于多步TD，我们已经在缓冲区中计算了n步回报，所以这里的gamma是gamma^n
            targets = rewards + (self.gamma ** self.n_steps) * max_next_q_values * (1 - dones)
        
        # 计算当前Q值
        self.online_network.train()
        self.online_network.reset_noise()
        q_values = self.online_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算损失并更新网络
        loss = F.smooth_l1_loss(current_q_values, targets)  # Huber损失
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 10)
        self.optimizer.step()
        
        # 记录损失
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def update_exploration_rate(self):
        # 更新探索率
        self.exploration_rate = max(self.exploration_rate_min, 
                                   self.exploration_rate * self.exploration_rate_decay)
    
    def step(self, state, action, reward, next_state, done):
        # 存储经验
        self.store_experience(state, action, reward, next_state, done)
        
        # 增加步数计数器
        self.step_counter += 1
        
        # 每隔一定步数进行训练
        loss = 0
        if self.step_counter % self.train_frequency == 0:
            loss = self.train()
        
        # 每隔一定步数更新目标网络
        if self.step_counter % self.target_update_frequency == 0:
            self.update_target_network()
        
        return loss

# 训练函数
def train_agent(env_name, n_steps=3, num_episodes=1000, render=False):
    # 创建环境
    env = gym.make(env_name)
    if render:
        env = gym.make(env_name, render_mode="rgb_array")
        trigger = lambda t: (t+1) % 100 == 0
        env = RecordVideo(env, video_folder="./save_videos", episode_trigger=trigger, disable_logger=True)
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 创建智能体
    agent = MultiStepDQNAgent(state_size, action_size, n_steps=n_steps)
    
    # 训练循环
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 智能体学习
            loss = agent.step(state, action, reward, next_state, done)
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
        
        # 更新探索率
        agent.update_exploration_rate()
        
        # 记录奖励
        episode_rewards.append(episode_reward)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Exploration Rate: {agent.exploration_rate:.4f}")
    
    # 关闭环境
    env.close()
    
    # 保存模型
    torch.save(agent.online_network, f"./checkpoints/{env_name.split('-')[0]}_multistep_dqn.pth")
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title(f"Rewards over Episodes ({env_name})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(f"./rewards_{env_name.split('-')[0]}.png")
    plt.show()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(agent.loss_history)
    plt.title(f"Loss over Training Steps ({env_name})")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.savefig(f"./loss_{env_name.split('-')[0]}.png")
    plt.show()
    
    return agent

# 评估函数
def evaluate_agent(env_name, agent, num_episodes=10, render=True):
    # 创建环境
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)
    
    # 评估循环
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作（评估模式，不使用探索）
            agent.online_network.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.online_network(state_tensor)
                action = torch.argmax(q_values).item()
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
        
        # 记录奖励
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    # 关闭环境
    env.close()
    
    # 打印评估结果
    avg_reward = np.mean(total_rewards)
    print(f"Average Evaluation Reward: {avg_reward:.2f}")
    
    return avg_reward

# 主函数
if __name__ == "__main__":
    # 设置随机种子，保证结果可复现
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # 选择环境
    env_name = "LunarLander-v3"  # 可以替换为其他环境，如"CartPole-v1"、"Acrobot-v1"等
    
    # 训练智能体
    print(f"Training agent on {env_name} with multi-step TD learning...")
    agent = train_agent(env_name, n_steps=3, num_episodes=1000, render=True)
    
    # 评估智能体
    print(f"Evaluating agent on {env_name}...")
    evaluate_agent(env_name, agent, num_episodes=5, render=True)