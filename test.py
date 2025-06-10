# 基于Double Q-Network的FrozenLake强化学习实现
# Double Q-Network算法通过维护两个Q表来减少Q学习中的过度估计问题
# 一个Q表用于选择动作，另一个Q表用于评估动作的价值

import gymnasium as gym
import numpy as np
import time
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from collections import deque
import random
import matplotlib.pyplot as plt

# 创建环境参数
is_slippery = True  # 控制环境是否有随机性：True表示冰面滑，有随机性；False表示没有随机性
map_name = "8x8"    # 地图大小，可选"4x4"或"8x8"
desc = None         # 使用默认地图布局，也可以自定义地图

# 创建FrozenLake环境
env = gym.make("FrozenLake-v1", desc=desc, map_name=map_name, is_slippery=is_slippery)

# 算法超参数
discount_rate = 0.99    # 折扣因子，控制未来奖励的重要性
lr = 0.1               # 学习率
exploration_rate = 1.0  # 初始探索率
min_exploration_rate = 0.01  # 最小探索率
exploration_decay = 0.0005   # 探索率衰减系数
target_update_rate = 0.6     # 目标网络更新率

# 初始化两个Q表（Double Q-Network的核心）
# Q1用于选择动作，Q2用于评估动作价值，两者交替更新
Q1 = np.random.uniform(size=(env.observation_space.n, env.action_space.n)) / 500
Q2 = np.random.uniform(size=(env.observation_space.n, env.action_space.n)) / 500

# 用于记录训练过程中的数据
episode_rewards = []  # 记录每个episode的累积奖励
td_errors_history = []  # 记录每个episode的平均TD误差
steps_history = []  # 记录每个episode的步数

# 训练过程
num_episodes = 30000
for episode in range(num_episodes):
    state, info = env.reset()  # 初始化环境，返回初始状态
    td_errors = []  # 记录当前episode的TD误差
    action_count = 0  # 记录当前episode的步数
    done = False  # 游戏是否结束
    episode_reward = 0  # 当前episode的累积奖励
    
    # 每个episode进行游戏，直到结束
    while not done:
        # ε-greedy策略选择动作
        if np.random.uniform(0, 1) < exploration_rate:
            # 探索：随机选择动作
            action = env.action_space.sample()
        else:
            # 利用：选择Q1值最大的动作
            action = np.argmax(Q1[state, :])
        
        # 执行动作，获取下一状态和奖励
        new_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward  # 累积奖励
        action_count += 1  # 步数加1
        
        # Double Q-learning更新（交替使用Q1和Q2）
        if episode % 2 == 0:  # 偶数episode更新Q1
            # 使用Q1选择最佳动作
            best_action = np.argmax(Q1[new_state, :])
            # 使用Q2评估该动作的价值
            target = reward + discount_rate * Q2[new_state, best_action] * (1 - done)
            # 计算TD误差
            td_error = target - Q1[state, action]
            # 更新Q1
            Q1[state, action] = Q1[state, action] + lr * td_error
        else:  # 奇数episode更新Q2
            # 使用Q2选择最佳动作
            best_action = np.argmax(Q2[new_state, :])
            # 使用Q1评估该动作的价值
            target = reward + discount_rate * Q1[new_state, best_action] * (1 - done)
            # 计算TD误差
            td_error = target - Q2[state, action]
            # 更新Q2
            Q2[state, action] = Q2[state, action] + lr * td_error
        
        td_errors.append(td_error)  # 记录TD误差
        state = new_state  # 更新当前状态
    
    # 记录训练数据
    episode_rewards.append(episode_reward)
    td_errors_history.append(np.mean(td_errors))
    steps_history.append(action_count)
    
    # 衰减探索率
    exploration_rate = min_exploration_rate + \
                      (1.0 - min_exploration_rate) * np.exp(-exploration_decay * episode)
    
    # 每隔2000个episode打印训练进度
    if episode % 2000 == 0:
        print(f'当前第{episode}个episode，此次episode均TD误差为：{np.mean(td_errors):.6f}，总共走{action_count}步')

# 关闭训练环境
env.close()

# 合并两个Q表作为最终策略（取平均）
final_Q = (Q1 + Q2) / 2

# 保存训练好的Q表
np.save('./models/q_double_network_frozenlake.npy', final_Q)

# 可视化训练过程
plt.figure(figsize=(15, 5))

# 绘制TD误差变化
plt.subplot(1, 3, 1)
plt.plot(range(0, num_episodes, 100), td_errors_history[::100])
plt.title('TD误差变化')
plt.xlabel('Episode')
plt.ylabel('平均TD误差')

# 绘制步数变化
plt.subplot(1, 3, 2)
plt.plot(range(0, num_episodes, 100), steps_history[::100])
plt.title('步数变化')
plt.xlabel('Episode')
plt.ylabel('步数')

# 绘制奖励变化
plt.subplot(1, 3, 3)
plt.plot(range(0, num_episodes, 100), episode_rewards[::100])
plt.title('奖励变化')
plt.xlabel('Episode')
plt.ylabel('累积奖励')

plt.tight_layout()
plt.savefig('./double_q_learning_training.png')

# 测试训练好的智能体
print("\n开始测试训练好的智能体...")

# 创建测试环境（带可视化）
env = gym.make("FrozenLake-v1", desc=desc, map_name=map_name, render_mode='human', is_slippery=is_slippery)

# 测试参数
test_episodes = 10  # 测试轮数
results = []  # 记录测试结果

# 进行测试
for episode in range(test_episodes):
    state, info = env.reset()  # 初始化环境
    done = False
    steps = 0
    
    # 使用训练好的策略进行游戏
    while not done:
        # 选择Q值最大的动作
        action = np.argmax(final_Q[state, :])
        
        # 执行动作
        new_state, reward, done, truncated, info = env.step(action)
        env.render()  # 渲染环境
        steps += 1
        state = new_state  # 更新状态
        
        # 如果游戏结束，记录结果
        if done:
            if reward > 0:
                results.append(('win', steps))
            else:
                results.append(('lose', steps))
            
            # 短暂暂停，便于观察
            time.sleep(0.5)

# 计算并打印测试结果
wins = len([result for result in results if result[0] == 'win'])
win_steps = [result[1] for result in results if result[0] == 'win']
mean_win_steps = np.mean(win_steps) if win_steps else 0

print(f'测试结果：共测试{test_episodes}轮，成功{wins}轮，成功率{wins/test_episodes:.2f}')
if wins > 0:
    print(f'成功时的平均步数：{mean_win_steps:.2f}')

# 关闭环境
env.close()