#本文使用Q-learning算法，基于Q-table进行学习，并使用多步TD算法更新表格
#Q-table相当于最优动作价值函数，最终会使用该表指导智能体行动
#本文具体训练时，使用TD算法，并且行为策略和目标策略不一致
#行为策略：采用greedy，目标策略：最优动作价值函数
#最终算法评判标准
#1、模拟游戏，最终达到目标的占比
#2、模拟游戏，达到目标的平均动作数

import gymnasium as gym
import numpy as np 
# desc=["SFFF", "FHFH", "FFFH", "HFFG"]
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


#创建计算多步TD回报辅助函数
def get_ut_by_records(records,discount_rate):
    rewards = np.array([record[2] for record in records])
    discounts = np.array([discount_rate**i for i in range(len(records))])
    ut = (rewards*discounts).sum()
    return ut

#创建环境--冰湖
is_slippery = False
map_name = "8x8"
desc=generate_random_map(size=12)
# desc = None
env = gym.make("FrozenLake-v1",map_name=map_name,desc=desc,is_slippery=is_slippery) #is_slippery控制环境是否随机切换状态，True是，False否

#尝试使用uniform随机初始化，因为Q表内的值，不可能小于0，考虑到reward最大值是1，为避免对真实情况干扰太大，缩小到0-0.01之间
Q_table = np.random.uniform(size=(env.observation_space.n,env.action_space.n))/500
#考虑到最后一个状态所有动作奖励都是0，因为是终态，所以把这个先验假设体现在Q_table中
# Q_table[-1,:] = 0.0

discount_rate = 0.99
lr = 0.05
exploration_rate = 0.01
TD_steps = 1
total_td_errors = []

#训练过程
for episode in range(20000):
    state,info = env.reset() #初始化环境，返回初始状态
    td_errors = []
    action_count = 0
    done = False
    #每一个episode，都要确保玩到最后
    while not done:
        #初始化多步TD记录
        records = []
        #首先基于s_t，按照greedy策略选择动作s_t
        if np.random.uniform(0,1) < exploration_rate:
            action = env.action_space.sample()
        else :
            action = np.argmax(Q_table[state,:])
        s_t,a_t = state,action
        
        for i in range(TD_steps):
            #做动作
            new_state_temp,reward,done,truncated,info = env.step(action)
            action_count+=1
            #记录
            records.append((state,action,reward,new_state_temp))
            #判断游戏是否结束，如果结束，则提前终止
            if done:
                break
            #切状态
            state = new_state_temp
            
            #选择下一步动作
            if np.random.uniform(0,1) < exploration_rate:
                action = env.action_space.sample()
            else :
                action = np.argmax(Q_table[state,:])            

        #基于以上记录，使用多步TD算法更新
        #多步TD算法，即让智能体做多个动作，收集更多的奖励反馈，让TD-target更加精确
        #q(s,a) ≈ r_t + γmax(q(s_t+1))
        #如果游戏未结束：负梯度 = r_t + γmax(q(s_t+1)) - q(s,a) ；如果游戏结束：负梯度 = r_t - q(s,a)
        #首先统一计算u_t这一项，因为是统一的
        u_t = get_ut_by_records(records,discount_rate)
        if done:
            #如果游戏结束，则说明此轮的records，最后一组元祖，就是最终状态，因为我们假设最终状态下所有动作的值都是0，所以就不需要加Q(s_t+m,a_t+m)了
            neg_grad = u_t - Q_table[s_t,a_t]
        else:
            #如果游戏没有结束，则最后一步需要查表求最大值，并计算进TD error内
            #之所以使用len(records)，不使用TD_steps，是因为要考虑游戏可能提前结束
            neg_grad = u_t + (discount_rate**(len(records)))*np.max(Q_table[records[-1][3],:]) -  Q_table[s_t,a_t]
            
        td_errors.append(neg_grad)
        
        #使用负梯度，以及随机梯度算法，更新Table
        Q_table[s_t,a_t] = Q_table[s_t,a_t] + lr*neg_grad
        
        #更新后，让智能体的状态从s_t+1开始，即records里面最后一条记录的最后一个状态，进入下一轮的多步TD更新
        state = records[0][3]
    total_td_errors.append(np.mean(td_errors))    
    #探索率逐步衰减
    exploration_rate = 0.001 + \
                    (0.001 - 0.001) * np.exp(-0.00005 * episode)
    
    #每隔100个episode打印一下结果                
    if episode%1000 == 0:
        print(f'当前第{episode}个episode，此次episode均td误差为：{np.mean(td_errors)}，总共走{action_count}步')

#最后关闭环境，下面会再初始化一个同样的环境
env.close()
print(Q_table)
#保存Q_table表
np.save('./models/q_lr_frozenlake_total_random_policy.npy',Q_table)

from matplotlib import pyplot as plt
plt.plot(total_td_errors)
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
print(f'此次共模拟{estim_count}轮，其中成功{wins}，成功率{round(wins/estim_count,2)}，平均成功消耗步数{mean_action_count}')


# #最后，对学习到的Q-table进行可视化
# Q_vec_sum = Q_table.mean(axis=1)
# Q_vec_sum = Q_vec_sum.reshape(8,8)
# plt.imshow(c)
# plt.hot()
# plt.colorbar()