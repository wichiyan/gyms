import gym
import time
import numpy as np

# Monkey patch if missing，主要修复numpy报错
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# 创建环境并指定渲染模式
env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("CartPole-v1")

# 重置环境
observation, info = env.reset(seed=42)

for _ in range(1000):
    # # 选择动作
    # action = env.action_space.sample()
    # print(f'环境采样的动作是{action}')
    #随机选择1或者0
    if np.random.rand(1) < 0.5:
        action = 0
    else :
        action = 1
    # time.sleep(1)
    # 执行动作（新API返回5个值）
    observation, reward, terminated, truncated, info = env.step(action)
    env.render() #将最新状态渲染到窗口
    print(f'执行以上动作后，得到的环境奖励为{reward}')
    # 游戏结束条件
    if terminated or truncated:
        print(f'游戏结束')
        # observation, info = env.reset() #重置环境
env.close()