#本文使用Q-learning算法，基于Q-table进行学习
#Q-table相当于最优动作价值函数，最终会使用该表指导智能体行动
#本文具体训练时，使用TD算法，并且行为策略和目标策略不一致
#行为策略：采用greedy，目标策略：最优动作价值函数
#最终算法评判标准
#1、模拟游戏，最终达到目标的占比
#2、模拟游戏，达到目标的平均动作数

import gymnasium as gym
import numpy as np 
# desc=["SFFF", "FHFH", "FFFH", "HFFG"]
import time
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

#创建环境--冰湖
is_slippery = False
map_name = "8x8"
# desc = ["SFFFFFFF", "FHFHFFFF", "FFFFFFFF", "HFFFFFHG","SFFFFFFF", "FHFFHFFH", "FFFHFFFF", "HFFFFFFG"]
desc = None
desc = generate_random_map(12)
env = gym.make("FrozenLake-v1",desc=desc,map_name=map_name,is_slippery=is_slippery) #is_slippery控制环境是否随机切换状态，True是，False否

#尝试使用uniform随机初始化，因为Q表内的值，不可能小于0，考虑到reward最大值是1，为避免对真实情况干扰太大，缩小到0-0.1之间
Q_table = np.random.uniform(size=(env.observation_space.n,env.action_space.n))/500
Q_target = np.random.uniform(size=(env.observation_space.n,env.action_space.n))/500

discount_rate = 0.99
lr = 0.1
exploration_rate = 1.0

#训练过程
for episode in range(20000):
    state,info = env.reset() #初始化环境，返回初始状态
    td_errors = []
    action_count = 0
    done = False
    #每一个episode，都要确保玩到最后
    while not done:
        #使用多步TD算法
        #即每隔多少步，集中更新一次参数，在这个期间
        # for t in range(10):
        #使用greedy算法，选择下一步动作
        if np.random.uniform(0,1) < exploration_rate:
            action = env.action_space.sample()
        else :
            action = np.argmax(Q_table[state,:])
            
        #让环境做动作，产生下一个状态
        new_state,reward,done,truncated,info = env.step(action)
        
        action_count+=1
        #主要是这里，在计算max(Q(s,a))的时候，使用target网络，主要为了减少自举和高估带来的影响，注意只是减少，而不是消除
        neg_grad = reward + discount_rate*np.max(Q_target[new_state,:])*(1-done) -  Q_table[state,action]
            
        td_errors.append(neg_grad)
        
        #使用负梯度，以及随机梯度算法，更新Table
        Q_table[state,action] = Q_table[state,action] + lr*neg_grad
        
        #切换状态
        state = new_state        

        #按照一定频次更新target网络，比如每隔5步更新一次
        if action_count%5==0:
            Q_target = 0.6*Q_target+ (1-0.6)*Q_table
            
    exploration_rate = 0.01 + \
                    (1.0 - 0.01) * np.exp(-0.0005 * episode)
    
    #每隔100个episode打印一下结果                
    if episode%2000 == 0:
        print(f'当前第{episode}个episode，此次episode均td误差为：{np.mean(td_errors)}，总共走{action_count}步')

#最后关闭环境，下面会再初始化一个同样的环境
env.close()
print(Q_table)

#保存Q_table表
np.save('./models/q_lr_frozenlake_total_random_policy.npy',Q_table)


#加载指定已训练好的Q_table表
# Q_table = np.load('./models/q_lr_frozenlake_total_random_policy.npy')
#开始模拟测试
env = gym.make("FrozenLake-v1",desc=desc,map_name=map_name,render_mode='human',is_slippery=is_slippery)
estim_count = 10
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
print(f'此次共模拟{estim_count}轮，其中成功{wins}，成功率{round(wins/estim_count,2)}，平均成功消耗步数{mean_action_count}')

