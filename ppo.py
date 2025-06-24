import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
import time

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义策略网络
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.actor(state)

# 定义价值网络
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.critic(state)

# 定义PPO算法
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, 
                 eps_clip=0.2, K_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.policy_old = ActorNetwork(state_dim, action_dim)
        self.policy_old.load_state_dict(self.actor.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state)
        action_probs = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
    def update(self, memory):
        # 将收集的轨迹转换为张量
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()
        old_rewards = torch.tensor(memory.rewards).detach()
        old_dones = torch.tensor(memory.is_terminals).detach()
        
        # 计算折扣回报
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns)
        
        # 标准化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # 优化策略网络
        for _ in range(self.K_epochs):
            # 评估旧动作和状态
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            
            # 计算比率 (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 计算替代损失 (L^{CLIP})
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            critic_loss = self.MseLoss(state_values, returns)
            
            # 总损失
            loss = actor_loss + 0.5 *torch.tensor(critic_loss,dtype=torch.float32) - 0.01 * dist_entropy.mean()
            
            # 梯度下降
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.actor.state_dict())

# 定义内存类用于存储轨迹
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

# 主训练函数
def train():
    # 创建环境
    env_name = "LunarLander-v3"
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5,render_mode="rgb_array")
    trigger = lambda t: (t+1) % 1000 == 0
    env = RecordVideo(env, video_folder="./save_videos_lunar_PPO", episode_trigger=trigger, disable_logger=True)

    # 设置超参数
    max_episodes = 6000
    max_timesteps = 2000
    update_timestep = 4000
    log_interval = 100
    state_dim,action_dim = env.observation_space.shape[0], env.action_space.n
    # 初始化PPO和内存
    ppo = PPO(state_dim, action_dim)
    memory = Memory()
    
    # 记录训练进度
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # 训练循环
    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            
            # 选择动作
            action = ppo.select_action(state, memory)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            
            # 存储奖励和是否终止
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # 更新状态
            state = next_state
            
            # 如果达到更新时间步或者回合结束，则更新策略
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward
            if done:
                break
        avg_length += t
        
        # 记录日志
        if i_episode % log_interval == 0:
            avg_length = avg_length / log_interval
            running_reward = running_reward / log_interval
            
            print(f'Episode {i_episode}\tavg length: {avg_length:.2f}\tavg reward: {running_reward:.2f}')
            running_reward = 0
            avg_length = 0
            
        # 如果平均奖励足够高，则保存模型并退出
        # if running_reward > 200:
        #     print(f"Solved! Running reward is now {running_reward} and the last episode runs to {t} time steps!")
        #     torch.save(ppo.policy_old.state_dict(), f'./checkpoints/PPO_{env_name}.pth')
        #     # break
    
    # 保存最终模型
    torch.save(ppo.policy_old.state_dict(), f'./checkpoints/PPO_{env_name}_final.pth')

# 评估函数
def evaluate(render=True):
    env_name = "LunarLander-v3"
    
    if render:
        env =gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=15.0, turbulence_power=1.5,render_mode="human")
        # 可选：录制视频
        # env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    else:
        env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    #加载训练好的模型
    policy = ActorNetwork(state_dim, action_dim)
    policy.load_state_dict(torch.load(f'./checkpoints/PPO_{env_name}_final.pth'))
    
    n_episodes = 10
    max_timesteps = 1000
    results = []
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        total_reward = 0
        
        for t in range(max_timesteps):
            # 选择动作
            state_tensor = torch.FloatTensor(state)
            action_probs = policy(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample().item()
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state
            
            if done:
                if reward>0:
                    results.append(('win'))
                else:
                    results.append(('lose'))
                break
    #关闭环境
    env.close()
    #打印模拟结果
    wins =len( [ result[0] for result in results if result[0] =='win']  )
    print(f'此次共模拟{n_episodes}轮，其中成功{wins}，成功率{round(wins/n_episodes,2)}')


if __name__ == "__main__":
    # 训练模型
    train()
    # 评估模型
    evaluate(render=True)
    
    print("Training and evaluation completed.")