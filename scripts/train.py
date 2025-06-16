
import gymnasium as gym
import os,sys
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from agents.value_based_agents import DQNAgent
from utils import functions
from envs.env import Env
from monitors.tb_monitor import TBMonitor

if __name__ == "__main__":
    config_path = "../configs/value_based/dqn.yaml"
    config = functions.get_config(config_path)
    #创建环境
    env = Env(config,mode='train')
    #创建智能体
    agent = DQNAgent(env, config)
    agent.train()
    #创建监控器
    tb_monitor = TBMonitor(env,agent,config,name='dqn_base_Acrobot-v1')
    #开始训练
    episodes = config['train']['episodes']
    for episode in tqdm(range(episodes)):
        #重置环境
        state, _ = env.reset()
        done = False
        while not done:
            #选择动作
            action = agent.select_action(state, episode, episodes)
            #执行动作
            next_state, reward, done, info = env.step(action)
            #智能体更新策略，更新探索率
            agent.update(state, action, reward, next_state, done)
            #切换状态
            state = next_state
            
        #监控信息
        tb_monitor.monitor(info)
        
    env.close()
    #保存智能体
    save_path = config['train'].get('save_path', 'dqn_agent.pth')
    agent.save_network(save_path, weights_only=True)

