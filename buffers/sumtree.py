
import numpy as np
# 定义SumTree数据结构，专门用于优先经验回放，主要是加速抽样
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验池容量
        self.tree = np.zeros(2 * capacity - 1)  # 存储经验对应优先级
        self.data = np.zeros(capacity, dtype=object)  # 存储经验数据，数据为元祖，元祖内可以是任意数据
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
        # 根据优先级采样，采样1个，返回叶节点索引、优先级和对应数据
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
