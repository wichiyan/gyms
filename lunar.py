#本代码使用DQN算法，处理Acrobot-v1环境
#因为状态是连续的，所以需要搭建神经网络拟合Q函数
import torch
import torch.nn as nn
import torch.functional as F
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from networks.DQN import Dueling_DQN

#创建环境
# env = gym.make("LunarLander-v3",render_mode="rgb_array")
env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=15.0, turbulence_power=1.5,render_mode="rgb_array")
trigger = lambda t: (t+1) % 1000 == 0
env = RecordVideo(env, video_folder="./save_videos_lunar", episode_trigger=trigger, disable_logger=True)

#创建Q网络
Q_network = Dueling_DQN(env.observation_space.shape[0], env.action_space.n)
#定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(Q_network.parameters(), lr=0.001)
exploration_rate = 1.0  # 初始探索率
#开始训练
num_episodes = 10000
# for episode in range(num_episodes):
#     state, _ = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         #将状态转换为tensor
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
#         #选择动作，使用ε-greedy策略
#         if np.random.rand() < exploration_rate:
#             action = env.action_space.sample()
#         else:
#             #将状态转换为tensor
#             with torch.no_grad():
#                 q_values = Q_network(state_tensor)
#                 action = torch.argmax(q_values).item()
        
#         #执行动作
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
        
#         #更新Q网络
#         next_state_tensor = torch.tensor(next_state).unsqueeze(0)
#         with torch.no_grad():
#             next_q_values = Q_network(next_state_tensor)
#             max_next_q_value = torch.max(next_q_values).item()
        
#         #计算Q(s_t，a_t)以及目标Q值
#         q_values = Q_network(state_tensor)
#         # reward = reward if not done else 100  # Acrobot-v1的奖励是-1，直到成功
#         target_q_value = reward + 0.99 * max_next_q_value * (1 - int(done))
        
#         #计算损失
#         loss = criterion(q_values[0][action], torch.tensor(target_q_value, dtype=torch.float32))
        
#         #反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         state = next_state
#         total_reward += reward
        
#     # exploration_rate = 0.001 + \
#     #                 (1.0 - 0.001) * np.exp(-0.00005 * episode)
#     exploration_rate = 1.0 + (0.01-1.0)*episode/num_episodes   
#     if episode % 200 == 0:
#         print(f'Episode {episode }, Total Reward: {total_reward}, Exploration Rate: {exploration_rate:.4f},reward:{reward},done:{done}')

#关闭环境
# env.close()

#此处选择保存模型权重和结构
# torch.save(Q_network,'./checkpoints/lunar_dqn.pth')

#加载模型，注意设置weights_only=False，因为默认是True，即只加载权重，此时会和上面代码冲突，报错
Q_network = torch.load('./checkpoints/lunar_dqn.pth',weights_only=False)

#开始评估
#开始模拟测试
# env = gym.make('LunarLander-v3',render_mode='human')
env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=15.0, turbulence_power=1.5,render_mode="human")

estim_count = 10
#总共模拟20轮，每一轮都记录最后是成功了还是失败了，如果成功了，总共走的步数
results = []
for episode in range( estim_count ):
    #初始化环境
    state ,info = env.reset()
    #使用训练好的Q_table，按照最大化原则指导智能体走位
    done = False
    take_action_count = 0
    #玩每一轮游戏，直到结束
    while not done:
        #选择当前状态，使得Q值最大的动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if np.random.rand() < 0.01:
            action = env.action_space.sample()
        else:
            #将状态转换为tensor
            with torch.no_grad():
                q_values = Q_network(state_tensor)
                action = torch.argmax(q_values).item()
        
        #让智能体走位
        new_state,reward,done,truncated,info = env.step(action)
        #刷新环境
        env.render()
        take_action_count+=1
        
        #更新当前状态
        state = new_state
        
        #判断，如果游戏结束，则记录该局游戏结果
        if done:
            #如果最后一步奖励是1，则游戏赢，否则，输，同时将总共走的步数记录下来
            if reward>0:
                results.append(('win',take_action_count))
            else:
                results.append(('lose',take_action_count))
#关闭环境
env.close()
#打印模拟结果
wins =len( [ result[0] for result in results if result[0] =='win']  )
mean_action_count = np.mean(  [ result[1] for result in results if result[0] =='win']  )
print(f'此次共模拟{estim_count}轮，其中成功{wins}，成功率{round(wins/estim_count,2)}，平均成功消耗步数{mean_action_count}')
