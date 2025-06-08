#本文使用Q-learning算法，基于Q-table进行学习，并使用经验回放技巧，先随机探索，然后当经验池足够大时，
# 开始使用经验池中的数据进行训练
#Q-table相当于最优动作价值函数，最终会使用该表指导智能体行动
#本文具体训练时，使用TD算法，并且行为策略和目标策略不一致
#行为策略：采用greedy，目标策略：最优动作价值函数
#最终算法评判标准
#1、模拟游戏，最终达到目标的占比
#2、模拟游戏，达到目标的平均动作数

import gymnasium as gym
import numpy as np 
from collections import deque
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


# 假设你已经有了一个优先级数组 priorities，形状和 replay_buffer 一致
priorities = np.array([transition.priority for transition in replay_buffer])
probs = priorities ** alpha
probs /= probs.sum()

# 按照 probs 分布来采样 indices
indices = np.random.choice(len(replay_buffer), size=batch_size, replace=False, p=probs)

# 计算 importance-sampling 权重（可选，用于校正偏差）
weights = (1 / (len(replay_buffer) * probs[indices])) ** beta
weights /= weights.max()  # 归一化

# 采样出 transitions 和对应的权重
transitions = [replay_buffer[i] for i in indices]


# desc=["SFFF", "FHFH", "FFFH", "HFFG"]
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
def get_ut_by_records(records,discount_rate):
    rewards = np.array([record[2] for record in records])
    discounts = np.array([discount_rate**i for i in range(len(records))])
    ut = (rewards*discounts).sum()
    return ut

def get_explore_threshold(episode, explore_threshold_max, explore_threshold_min, explore_decay_rate):
    """
    计算当前episode的探索阈值
    """
    return explore_threshold_min + (explore_threshold_max - explore_threshold_min) \
            * np.exp(-explore_decay_rate * episode)

#创建经验池
experience_pool = deque(maxlen=5000)  # 最大经验池大小为10000

#创建环境--冰湖
is_slippery = False
map_name = "4x4"
desc=generate_random_map(size=20)
desc=None
explore_threshold_max = 1.0
explore_threshold_min = 0.005
explore_decay_rate = 0.001
explore_threshold = 1.0
# desc = None
env = gym.make("FrozenLake-v1",map_name=map_name,desc=desc,is_slippery=is_slippery) 

#尝试使用uniform随机初始化，因为Q表内的值，不可能小于0，考虑到reward最大值是1，为避免对真实情况干扰太大，缩小到0-0.01之间
Q_table = np.random.uniform(size=(env.observation_space.n,env.action_space.n))/100
#考虑到最后一个状态所有动作奖励都是0，因为是终态，所以把这个先验假设体现在Q_table中
# Q_table[-1,:] = 0.0

discount_rate = 0.99
lr = 0.05
exploration_rate = 0.01
total_td_errors = []

#训练过程
for episode in range(20000):
    state,info = env.reset() #初始化环境，返回初始状态
    td_errors = []
    action_count = 0
    done = False
    #每一个episode，都要确保玩到最后
    while not done:
        #首先基于s_t，按照greedy策略选择动作s_t
        if np.random.uniform(0,1) < explore_threshold:
            action = env.action_space.sample()
        else :
            action = np.argmax(Q_table[state,:])
        
        #开始行动
        new_state,reward,done,truncated,info = env.step(action)
        record = (state, action, reward, new_state, done)
        #放到经验池中
        experience_pool.append(record)
        #更新当前状态
        state = new_state
        
        #判断经验池是否足够大，如果足够大，则开始使用经验池中的数据进行训练
        if len(experience_pool) < 2000:
            #如果经验池不够大，则跳过本次训练
            continue
        #从经验池中随机采样一个数据，并进行更新，总共更新5次，使用经验数据更新Q表
        for _ in range(20):
            idx = np.random.choice(range(len(experience_pool)),size=1)[0]
            record_replay = experience_pool[idx]
            s_t, a_t = record_replay[0], record_replay[1]
            #计算负梯度
            #负梯度 = u_t + discount_rate * max(Q_table[new_state,:]) * (1-done) - Q_table[state,action]    
            neg_grad = record_replay[2] + \
            discount_rate*np.max(Q_table[record_replay[3],:])*(1-record_replay[4]) -  Q_table[s_t,a_t]
            #使用负梯度，以及随机梯度算法，更新Table
            Q_table[s_t,a_t] = Q_table[s_t,a_t] + lr*neg_grad

        #使用在线数据更新Q表
        neg_grad = record[2] + discount_rate*np.max(Q_table[record[3],:])*(1-record[4]) -  Q_table[s_t,a_t]
        td_errors.append(neg_grad)
        
    if len(td_errors) > 0:
        total_td_errors.append(np.mean(td_errors))    
    #探索率逐步衰减
    explore_threshold = get_explore_threshold(episode, explore_threshold_max,
                                              explore_threshold_min, explore_decay_rate)
    
    #每隔100个episode打印一下结果                
    if episode%1000 == 0 and len(td_errors)>0:
        print(f'当前第{episode}个episode，此次episode均td误差为：{np.mean(td_errors)}，总共走{action_count}步')

#最后关闭环境，下面会再初始化一个同样的环境
env.close()
print(Q_table)
#保存Q_table表
np.save('./models/q_lr_frozenlake_total_random_policy.npy',Q_table)

from matplotlib import pyplot as plt
plt.plot(total_td_errors)

#最后，对学习到的Q-table进行可视化
Q_vec_sum = Q_table.mean(axis=1)
Q_vec_sum = Q_vec_sum.reshape(8,8)
plt.imshow(c)
plt.hot()
plt.colorbar()
plt.show()

#加载指定已训练好的Q_table表
# Q_table = np.load('./models/q_lr_frozenlake_total_random_policy.npy')
#开始模拟测试
env = gym.make("FrozenLake-v1",desc=desc,map_name=map_name,render_mode='human',is_slippery=is_slippery)
estim_count = 20
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
        action = np.argmax(Q_table[state,:])
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
#打印模拟结果
wins =len( [ result[0] for result in results if result[0] =='win']  )
mean_action_count = np.mean(  [ result[1] for result in results if result[0] =='win']  )
print(f'此次共模拟{estim_count}轮，其中成功{wins}，成功率{round(wins/estim_count,2)}，\
      平均成功消耗步数{mean_action_count}')

