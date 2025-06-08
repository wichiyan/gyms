import numpy as np
import gymnasium as gym

# # Monkey patch if missing，主要修复numpy报错
# if not hasattr(np, 'bool8'):
#     np.bool8 = np.bool_

# 创建环境
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)

# 初始化Q表 (状态数 x 动作数)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# 超参数设置
learning_rate = 0.1        # 学习率
discount_factor = 0.99     # 折扣因子
exploration_rate = 1.0     # 初始探索率
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay = 0.001  # 探索率衰减率
num_episodes = 10000       # 训练总轮数
max_steps = 100            # 每轮最大步数

# 训练过程
for episode in range(num_episodes):
    # 重置环境获取初始状态
    state = env.reset()[0]
    done = False
    
    for step in range(max_steps):
        # 探索与利用的权衡
        exploration_threshold = np.random.uniform(0, 1)
        if exploration_threshold < exploration_rate:
            action = env.action_space.sample()  # 随机探索
        else:
            action = np.argmax(q_table[state, :])  # 利用已知最优
        
        # 执行动作
        new_state, reward, done, truncated, info = env.step(action)
        
        # Q值更新公式
        current_q = q_table[state, action]
        future_q = np.max(q_table[new_state, :])
        
        # Bellman方程更新Q值
        if done:
            q_table[state, action] = current_q + learning_rate * (reward - current_q)
        else:
            q_table[state, action] = current_q + learning_rate * (reward + discount_factor * future_q - current_q)
        
        state = new_state
        
        if done or truncated:
            break
    
    # 指数衰减探索率
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay * episode)
    
    # 每1000轮输出一次进度
    if (episode + 1) % 1000 == 0:
        print(f"Episode: {episode+1}/{num_episodes}, Exploration Rate: {exploration_rate:.3f}")

print("\n训练完成！")
print(q_table)
# # 测试训练好的智能体
# test_episodes = 10
# successes = 0
# env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", render_mode='human')
# for episode in range(test_episodes):
#     state = env.reset()[0]
#     done = False
    
#     print(f"\n测试轮次 {episode+1}")
#     env.render()
    
#     while not done:
#         # 始终选择最优动作
#         action = np.argmax(q_table[state, :])
#         new_state, reward, done, truncated, info = env.step(action)
        
#         # 更新状态
#         state = new_state
#         env.render()
        
#         if done:
#             if reward == 1.0:
#                 successes += 1
#                 print("目标达成！")
#             else:
#                 print("掉入冰洞！")
#             break

# print(f"\n成功率: {successes}/{test_episodes} ({successes/test_episodes*100:.2f}%)")

# # 关闭环境
# env.close()