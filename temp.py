import gymnasium as gym
import numpy as np
from collections import deque
import random
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


# 环境初始化
desc=generate_random_map(size=12)
env = gym.make("FrozenLake-v1", desc=desc,is_slippery=False, render_mode=None)
n_states = env.observation_space.n
n_actions = env.action_space.n

# 超参数
alpha = 0.1          # 学习率
gamma = 0.99         # 折扣因子
epsilon = 1.0        # 初始探索率
epsilon_decay = 0.999
epsilon_min = 0.01
n_episodes = 25000
max_steps = 100
n_step = 1           # 多步TD的步数

# 初始化 Q 表
Q = np.random.uniform(size=(env.observation_space.n,env.action_space.n))/500

# 训练主循环
for episode in range(n_episodes):
    state, _ = env.reset()
    done = False

    # 多步TD的轨迹缓存
    state_buffer = deque()
    action_buffer = deque()
    reward_buffer = deque()

    for t in range(max_steps):
        # ε-贪婪策略选择动作
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # 执行动作
        next_state, reward, done, truncated, _ = env.step(action)

        # 存入轨迹
        state_buffer.append(state)
        action_buffer.append(action)
        reward_buffer.append(reward)

        if len(reward_buffer) >= n_step:
            # 计算 n 步回报
            G = 0
            for i in range(n_step):
                G += (gamma ** i) * reward_buffer[i]

            # 加上 Q 值的 bootstrapping
            if not done:
                G += (gamma ** n_step) * np.max(Q[next_state])

            # 多步 TD 更新
            s_tau = state_buffer[0]
            a_tau = action_buffer[0]
            Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

            # 移除最旧的一个
            state_buffer.popleft()
            action_buffer.popleft()
            reward_buffer.popleft()

        # 更新状态
        state = next_state

        if done:
            # 清空剩余轨迹（注意：不完整轨迹也要更新）
            while len(reward_buffer) > 0:
                G = 0
                for i in range(len(reward_buffer)):
                    G += (gamma ** i) * reward_buffer[i]
                if not done:
                    G += (gamma ** len(reward_buffer)) * np.max(Q[state])
                s_tau = state_buffer[0]
                a_tau = action_buffer[0]
                Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

                state_buffer.popleft()
                action_buffer.popleft()
                reward_buffer.popleft()
            break

    # ε 衰减
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # 打印进度
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}, ε: {epsilon:.3f}")

# 训练结束
print("训练完成！")

# 评估策略的成功率
env = gym.make("FrozenLake-v1", desc=desc,is_slippery=False, render_mode='human')

test_episodes = 20
successes = 0

for _ in range(test_episodes):
    state, _ = env.reset()
    done = False
    for _ in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        if done:
            if reward == 1.0:
                successes += 1
            break

print(f"测试成功率: {successes / test_episodes:.2f}")
