
import gymnasium as gym
import os,sys
from gymnasium.wrappers import RecordVideo

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from agents.value_based_agents import DQNAgent
from utils import functions

def is_successful_episode(total_reward, terminated, truncated, threshold=500):
    if truncated:
        return True  # 撑满最大步数
    if terminated:
        return False  # 中途失败
    return total_reward >= threshold  # 也可以用总reward判断

if __name__ == "__main__":
    
    config_path = "../configs/value_based/dqn.yaml"
    config = functions.get_config(config_path)
    render_mode = config['env'].get('render_mode', 'human')
    env = gym.make(**config['env'],render_mode=render_mode)
    results=[]
    #创建智能体
    agent = DQNAgent(env, config)
    agent.eval()
    agent.load_network(config['train'].get('load_path', 'dqn_agent.pth'), weights_only=True)
    
    episodes = config['eval']['episodes']
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        total_step = 0 
        while not done:
            #选择动作
            action = agent.select_action(state, episode, episodes)
            #执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_step += 1    
            #切换状态
            state = next_state
        result = is_successful_episode(total_reward, terminated, truncated)
        results.append(result)
        if result:
            print(f"Episode {episode + 1}/{episodes} - Success! Total Reward: {total_reward}, Steps: {total_step}")
        else:
            print(f"Episode {episode + 1}/{episodes} - Failed. Total Reward: {total_reward}, Steps: {total_step}")  
    #打印成功率
    success_count = sum(results)
    success_rate = success_count / episodes
    print(f"Total Episodes: {episodes}, Success Count: {success_count}, Success Rate: {success_rate:.2f}")
    env.close()
