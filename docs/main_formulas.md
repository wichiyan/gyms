# 基础

## 基本概念


# value-based

基于值的算法，一般是基于环境奖励$R_t$，学习一个动作价值函数，或者最大动作价值函数，该函数可以判断在状态$s_t$下，什么动作得分最高，然后基于这个得分，指导智能体行动。

## 基本公式

### 目标函数--$Q(s_t,a_t)$

$$
\begin{align}
Q_{\pi}(s_t,a_t) &= E_{S_{t+1},A_{t+1}}[R_t + \gamma·Q_{\pi}(S_{t+1},A_{t+1})]\\
&\approx r_t + \gamma·Q_{\pi}(s_{t+1},a_{t+1})  \\
&\approx u_t 
\end{align}
$$

以上是对动作价值函数的近似，也即SARSA类型算法的基础目标函数，其中：

* (2)是单步蒙特卡洛近似，一般指导TD算法设计，好处是可以执行一步训练一步，坏处是存在自举偏差；
* (3)一局游戏后蒙特卡洛近似，一般指导蒙特卡洛算法设计，好处是无偏，坏处是每次玩一局后才能更新，且方差大；
* 之所以可以使用蒙特卡洛近似，是因为$Q_{\pi}$的随机性来源于状态转移函数和动作，动作是基于策略随机选择的，肯定满足动作概率分布；状态是环境根据动作随机切换的，且满足状态转移分布；所以($s_t,a_t,r_t,s_{t+1}$)可以认为是一次随机抽样，这就类似SGD；

### 目标函数--$Q^{*}(s_t,a_t)$

$$
\begin{align}
Q^{*}(s_t,a_t) &= E_{S_{t+1}\backsim p(·|s_t,a_t)}[ R_t + \gamma·max_{a\in A}Q^* (S_{t+1},a)|S_t=s_t,A_t=a_t ] \\
&\approx r_t + \gamma·max_{a\in A}Q^* (S_{t+1},a)
\end{align}
$$

以上是对最优动作价值函数的预估，用于Q-learning其中：

* $r_t$是在状态$s_t$下执行$a_t$得到的环境奖励
* $max_{a\in A}Q^* (S_{t+1},a)$是在$s_{t+1}$状态下，所有动作最大的值

### 状态价值函数--V(s)

$$
\begin{align}
V_{\pi}(s_t) &= E_{a_t \backsim \pi(·|s_t;\theta)}[ E_{s_{t+1}\backsim p(·|s_t,a_t)}[ R_t + \gamma · V_{\pi}(s_{t+1}) ] ] \\
&\approx r_t + \gamma·V_{\pi}(s_{t+1})
\end{align}
$$

# policy-based

## 基本公式

### 目标函数

即基于策略的相关算法，需要优化的函数，主要假设是需要找到一个策略函数，能让状态价值的期望最大，即在该策略指导下，最终所有状态的均值最大。

$$
\begin{align}
J(\theta) &= E_S[V_{\pi}(S)] \\
&= E_S [ E_{A\backsim \pi(·|S；\theta)} [Q_{\pi}(S,A) ] ] \\
&= E_S [ \sum_{\alpha \in A} \pi(a|s;\theta) · Q_{\pi}(s,a) ] \\
\end{align}
$$

### 策略梯度

即对目标函数进行求导，得出梯度，用于后续使用梯度上升对网络进行更新，以下是简化推导，主要得到实际建模使用的通用策略梯度公式，后续所有算法都是基于该策略梯度公式
**首先求$V_{\pi}(S)$的梯度**

$$
\begin{align}
\frac{\partial V_{\pi}(S)}{\partial \theta} &= \frac{\partial}{\partial \theta}  \sum_{\alpha \in A} \pi(a|s;\theta) · Q_{\pi}(s,a) \\
&= \sum_{\alpha \in A} \frac{ \partial \pi (a|s;\theta) · Q_{\pi}(s,a)}{\partial \theta}\\
&= \sum_{\alpha \in A} \frac{\partial \pi (a|s;\theta) }{\partial \theta} · Q_{\pi}(s,a)  + \sum_{\alpha \in A} \pi (a|s;\theta) ·  \frac{ \partial Q_{\pi}(s,a)}{\partial \theta}  \\
&\approx \sum_{\alpha \in A} \frac{\partial \pi (a|s;\theta) }{\partial \theta} · Q_{\pi}(s,a) \\
&\approx \sum_{\alpha \in A} \pi(a|s;\theta)· \frac{1}{\pi(a|s;\theta)} · \frac{\partial \pi (a|s;\theta) }{\partial \theta} · Q_{\pi}(s,a) \\
&\approx \sum_{\alpha \in A} \pi(a|s;\theta) · \frac{\partial ln \pi(a|s;\theta) }{\partial \theta} · Q_{\pi}(s,a)\\
&\approx E_{a\in\pi(·|s;\theta)} [ \frac{\partial ln \pi(a|s;\theta) }{\partial \theta} · Q_{\pi}(s,a) ]
\end{align}
$$

以上最终的近似，有严格数学证明，此处从略

**然后求对目标函数的梯度**

$$
\begin{align}
\frac{\partial J(\theta)}{\partial \theta} &= E_{S} (\frac{\partial V_{\pi}(S)}{\partial \theta}) \\
&= E_{S}[ E_{a\in\pi(·|s;\theta)} [ \frac{\partial ln \pi(a|s;\theta) }{\partial \theta} · Q_{\pi}(s,a) ] ]
\end{align}
$$

### 策略梯度蒙特卡洛近似形式

因为标准策略梯度的求解比较难，毕竟实际中状态转移函数一般不可知，纵使可知通过连加(离散情况)或积分(连续情况)计算量也比较大，考虑到实际训练时，经过的轮数足够大，所以可以使用蒙特卡洛近似的数值方式来近似目标函数梯度

$$
\begin{align}
\frac{\partial J(\theta)}{\partial \theta} &\approx g(s,a;\theta) \\
&\approx Q_{\pi}(s,a) · \frac{\partial ln \pi (a|s;\theta) }{\partial \theta}
\end{align}
$$

## 基本算法

### REINFORCE

### actor-critic$\delta$
