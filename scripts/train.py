
import gymnasium as gym
import os,sys
import numpy as np
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from agents.value_based_agents import DQNAgent
from utils import functions
from envs.env import Env
from monitors.tb_monitor import TBMonitor

if __name__ == "__main__":
    config = functions.get_config("../configs/value_based/dqn.yaml")
    #创建环境
    env = Env(config,mode='train')
    #创建智能体
    agent = DQNAgent(env, config)
    agent.train()  #切换模式
    #创建监控器
    tb_monitor = TBMonitor(env,agent,config,name='dqn_base_Acrobot-v1_dueling-again')
    #开始训练
    episodes = config['train']['episodes']
    for episode in tqdm(range(episodes), colour ='red'):
        #重置环境
        state, _ = env.reset()
        done = False
        losses = []
        while not done:
            #选择动作
            action = agent.select_action(state, episode, episodes)
            #执行动作
            next_state, reward, done, info = env.step(action)
            #智能体更新策略，更新探索率
            loss = agent.update(state, action, reward, next_state, done)
            losses.append(loss)
            #切换状态
            state = next_state
            
        #监控信息
        tb_monitor.run_info.update(info)
        tb_monitor.run_info.update({
            'mean_episode_loss':np.mean(losses)
            })
        tb_monitor.monitor()
        
    env.close()
    #保存智能体
    save_path = config['train'].get('save_path', 'dqn_agent.pth')
    agent.save_network(save_path, weights_only=True)

