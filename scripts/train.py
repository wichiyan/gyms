
import gymnasium as gym
import os,sys
from gymnasium.wrappers import RecordVideo

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from agents.value_based_agents import DQNAgent
from utils import functions

if __name__ == "__main__":
    config_path = "../configs/value_based/dqn.yaml"
    config = functions.get_config(config_path)
    #创建环境
    if config['train'].get('record_video', False) :
        env = gym.make(**config['env'],render_mode="rgb_array")
        trigger = lambda t: (t+1) % 500 == 0
        env = RecordVideo(env, video_folder="../runs/videos/save_videos_lunar_dueling_noise_doubleQ",
                        episode_trigger=trigger, disable_logger=True)
    else:
        env = gym.make(**config['env'])

    #创建智能体
    agent = DQNAgent(env, config)
    agent.train()
    episodes = config['train']['episodes']
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
            
            #智能体更新策略，更新探索率
            agent.update(state, action, reward, next_state, done)
                        
            #切换状态
            state = next_state

        if episode % 100 == 0:
            #每10个回合打印一次
             print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Steps: {total_step}")
        
    env.close()
    #保存智能体
    save_path = config['train'].get('save_path', 'dqn_agent.pth')
    agent.save_network(save_path, weights_only=True)

