
import gymnasium as gym
import os,sys
from gymnasium.wrappers import RecordVideo

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from agents.value_based_agents import DQNAgent
from utils import functions
from envs.env import Env

def is_successful_episode(total_reward, terminated, truncated, threshold=500):
    if terminated:
        return True  # 成功
    if truncated:
        return False  # 中途失败
    # return total_reward >= threshold  # 也可以用总reward判断

if __name__ == "__main__":
    
    config_path = "../configs/value_based/dqn.yaml"
    config = functions.get_config(config_path)
    
    #创建环境
    env = Env(config,mode='eval')
    #创建智能体
    agent = DQNAgent(env, config)
    agent.eval()
    agent.load_network(config['train'].get('save_path', 'dqn_agent.pth'), weights_only=True)
    
    #开始训练
    results=[]
    episodes = config['eval']['episodes']
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        while not done:
            #选择动作
            action = agent.select_action(state, episode, episodes)
            #执行动作
            next_state, reward, done, info = env.step(action) 
            #切换状态
            state = next_state
