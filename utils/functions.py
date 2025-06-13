
import torch

def soft_update(target, source, tau):
    """
    Perform a soft update of the target network parameters. \n
    usually used in Target Networks or Double Target Networks. \n
    fumula: target_param = tau * source_param + (1.0 - tau) * target_param
    Args:
        target (torch.nn.Module): The target network.
        source (torch.nn.Module): The source network.
        tau (float): The interpolation factor (0 < tau < 1).
    """
    #在无梯度环境进行单纯数据更新，更安全
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
        

def get_exploration_rate(episode, num_episodes,policy='exp' ,decay_rate= 0.0001,initial_rate=1.0, final_rate=0.01):
    """
    used in reinforcement learning to calculate the exploration rate (epsilon) for epsilon-greedy strategies. \n
    This function supports both exponential and linear decay strategies. 
    Args:
        episode (int): current episode number 
        num_episodes (int): total episode number  
        policy (str): exploration rate decay policy, option :'exp'、'linear',default is 'exp' 
        decay_rate (float): decay rate,only used if policy is 'exp' 
        initial_rate (float): initial exploration rate 
        final_rate (float): final exploration rate, the minimum value of epsilon 
    """
    if policy == 'exp' or policy == 'exponential':
        epsilon = final_rate + (initial_rate - final_rate) * torch.exp(-decay_rate * episode)
        return epsilon
    
    elif policy == 'linear':
        decay_rate = (initial_rate - final_rate) / num_episodes
        epsilon = max(final_rate,initial_rate - decay_rate * episode)
        return epsilon
    
    else:
        raise ValueError("Unsupported exploration rate policy. Use 'exp' or 'linear'.")
    
