import gymnasium as gym
import numpy as np
from collections import deque
import random
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt

# 定义SumTree数据结构，用于优先经验回放
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验池容量
        self.tree = np.zeros(2 * capacity - 1)  # 存储优先级和和
        self.data = np.zeros(capacity, dtype=object)  # 存储经验数据
        self.data_pointer = 0  # 数据指针，指向下一个要存储的位置
        self.size = 0  # 当前存储的经验数量
    
    def add(self, priority, data):
        # 添加新数据和优先级
        tree_idx = self.data_pointer + self.capacity - 1  # 叶节点索引
        self.data[self.data_pointer] = data  # 存储数据
        self.update(tree_idx, priority)  # 更新优先级
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity  # 更新指针
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, tree_idx, priority):
        # 更新优先级和传播变化
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # 传播变化到根节点
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get_leaf(self, v):
        # 根据优先级采样
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # 如果到达叶节点，返回
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            # 向下遍历树
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def total_priority(self):
        # 返回总优先级
        return self.tree[0]

# 定义优先经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # 优先级指数，控制采样偏差程度
        self.beta = beta  # 重要性采样指数，用于校正偏差
        self.beta_increment = beta_increment  # beta的增量，随着训练逐渐增加到1
        self.epsilon = epsilon  # 防止优先级为0
        self.max_priority = 1.0  # 初始最大优先级
    
    def add(self, experience, error=None):
        # 添加经验到缓冲区
        priority = self.max_priority if error is None else (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        # 采样一批经验
        datas = []
        indices = []
        priorities = []
        weights = []
        segment = self.tree.total_priority() / batch_size
        
        # 增加beta值，最大为1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 计算最小优先级的权重
        # 加上10**-8，主要为了避免除0
        min_prob = np.min(self.tree.tree[-self.tree.capacity:] + 10**-8 ) / self.tree.total_priority()
        max_weight = (min_prob * self.tree.size) ** (-self.beta)
        
        for i in range(batch_size):
            # 在每个段中采样
            a, b = segment * i, segment * (i + 1)
            v = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get_leaf(v)
            
            # 计算权重
            prob = priority / self.tree.total_priority()
            weight = (prob * self.tree.size) ** (-self.beta)
            weight = weight / max_weight  # 归一化权重
            
            datas.append(data)
            indices.append(idx)
            priorities.append(priority)
            weights.append(weight)
        
        return datas, indices, np.array(weights)
    
    def update_priorities(self, indices, errors):
        # 更新优先级
        for idx, error in zip(indices, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.size

# 计算探索阈值
def get_explore_threshold(episode, explore_threshold_max, explore_threshold_min, explore_decay_rate):
    """
    计算当前episode的探索阈值
    """
    return explore_threshold_min + (explore_threshold_max - explore_threshold_min) * np.exp(-explore_decay_rate * episode)

# 主函数
def main():
    # 环境参数
    is_slippery = False  # 是否有随机性
    map_name = "8x8"  # 地图大小
    desc = None  # 使用默认地图
    desc = generate_random_map(20)
    # 创建环境
    env = gym.make("FrozenLake-v1", map_name=map_name, desc=desc, is_slippery=is_slippery)
    
    # 超参数
    discount_rate = 0.99  # 折扣因子
    lr = 0.1  # 学习率
    explore_threshold_max = 1.0  # 最大探索阈值
    explore_threshold_min = 0.01  # 最小探索阈值
    explore_decay_rate = 0.001  # 探索衰减率
    explore_threshold = 1.0  # 初始探索阈值
    num_episodes = 20000  # 训练轮数
    batch_size = 32  # 批量大小
    update_frequency = 1  # 更新频率
    
    # 优先经验回放参数
    buffer_capacity = 10000
    alpha = 0.6  # 优先级指数，为0则均匀抽样，为1则完全按照TD误差抽样，小于1则缩小TD误差，大于1则放大TD误差
    beta = 0.4  # 重要性采样指数，为0则不进行修正，用来修正非均匀采样带来的偏差
    beta_increment = 10**-4  # beta增量，逐渐增加到1，为1的时候，即标准1/n*p修正
                             # 主要是为了前期减少权重修正，避免前期因噪声导致的大TD太过主导采样
    
    # 初始化Q表
    Q_table = np.random.uniform(size=(env.observation_space.n, env.action_space.n)) / 100
    
    # 创建优先经验回放缓冲区
    replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha, beta, beta_increment)
    
    # 记录训练过程中的TD误差
    total_td_errors = []
    
    # 训练过程
    for episode in range(num_episodes):
        state, info = env.reset()  # 初始化环境，返回初始状态
        td_errors = []
        action_count = 0
        done = False
        
        # 每一个episode，都要确保玩到最后
        while not done:
            action_count += 1
            
            # 基于当前状态选择动作（ε-greedy策略）
            if np.random.uniform(0, 1) < explore_threshold:
                action = env.action_space.sample()  # 随机探索
            else:
                action = np.argmax(Q_table[state, :])  # 贪婪选择
            
            # 执行动作
            new_state, reward, done, truncated, info = env.step(action)
            
            # 计算TD误差
            td_error = reward + discount_rate * np.max(Q_table[new_state, :]) * (1 - done) - Q_table[state, action]
            
            # 将经验存入回放缓冲区
            experience = (state, action, reward, new_state, done)
            replay_buffer.add(experience, td_error)
            
            # 更新当前状态
            state = new_state
            
            #以上只做采样，不做模型更新
            
            # 当缓冲区足够大且到达更新频率时，从缓冲区采样并更新Q表
            if len(replay_buffer) > 5000 and action_count % update_frequency == 0:
                # 从优先经验回放缓冲区采样
                batch, indices, weights = replay_buffer.sample(batch_size)
                
                # 计算新的TD误差并更新Q表
                new_errors = []
                for i, (s, a, r, next_s, d) in enumerate(batch):
                    # 计算TD目标
                    td_target = r + discount_rate * np.max(Q_table[next_s, :]) * (1 - d)
                    # 计算TD误差
                    td_error = td_target - Q_table[s, a]
                    # 使用重要性采样权重更新Q表
                    Q_table[s, a] += lr * weights[i] * td_error
                    # 记录新的TD误差用于更新优先级
                    new_errors.append(td_error)
                
                # 更新优先级
                replay_buffer.update_priorities(indices, new_errors)
                
                # 记录TD误差
                td_errors.extend(new_errors)
        
        # 记录平均TD误差
        if td_errors:
            total_td_errors.append(np.mean(np.abs(td_errors)))
        
        # 探索率衰减
        explore_threshold = get_explore_threshold(episode, explore_threshold_max, explore_threshold_min, explore_decay_rate)
        
        # 每隔1000个episode打印结果
        if episode % 1000 == 0 and td_errors:
            print(f'当前第{episode}个episode，此次episode均TD误差为：{np.mean(np.abs(td_errors))}，总共走{action_count}步')
    
    # 关闭环境
    env.close()
    
    # 保存Q表
    np.save('./models/q_lr_frozenlake_pri_replay.npy', Q_table)
    
    # 绘制TD误差变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(total_td_errors)
    plt.title('TD Error over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average TD Error')
    plt.savefig('./td_error_plot.png')
    plt.show()
    
    # 测试训练好的智能体
    test_agent(env, Q_table, map_name, desc, is_slippery)

# 测试函数
def test_agent(env, Q_table, map_name, desc, is_slippery):
    # 创建渲染环境
    env = gym.make("FrozenLake-v1", desc=desc, map_name=map_name, render_mode='human', is_slippery=is_slippery)
    estim_count = 20  # 测试轮数
    
    # 记录测试结果
    results = []
    for episode in range(estim_count):
        state, info = env.reset()
        done = False
        take_action_count = 0
        
        while not done:
            # 选择最优动作
            action = np.argmax(Q_table[state, :])
            # 执行动作
            new_state, reward, done, truncated, info = env.step(action)
            # 渲染环境
            env.render()
            take_action_count += 1
            # 更新状态
            state = new_state
            
            # 记录结果
            if done:
                if reward > 0:
                    results.append(('win', take_action_count))
                else:
                    results.append(('lose', take_action_count))
    
    # 计算成功率和平均步数
    wins = len([result[0] for result in results if result[0] == 'win'])
    if wins > 0:
        mean_action_count = np.mean([result[1] for result in results if result[0] == 'win'])
        print(f'此次共模拟{estim_count}轮，其中成功{wins}，成功率{round(wins/estim_count, 2)}，平均成功消耗步数{mean_action_count}')
    else:
        print(f'此次共模拟{estim_count}轮，没有成功案例')

# 运行主函数
if __name__ == "__main__":
    main()