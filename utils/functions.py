
import torch
from box import Box

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
        

def get_config(yaml_path, default_config=None):
    """
    Load configuration from a YAML file. \n
    If the file does not exist, return the default configuration.
    
    Args:
        yaml_path (str): Path to the YAML configuration file.
        default_config (dict, optional): Default configuration to return if the file does not exist.
        
    Returns:
        dict: Loaded configuration.
    """
    import os
    import yaml
    
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r',encoding='utf-8') as file:
            config = yaml.safe_load(file)
    else:
        raise ValueError(f"Configuration file {yaml_path} does not exist. Returning default configuration.")
    
    config = Box(config)
    
    return config


def get_device(config):
    """
    Get the device to use for training.
    
    Args:
        config (dict): Configuration.
        
    Returns:
        torch.device: The device to use for training.
    """
    device = config['global'].get('device', 'auto')
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == 'cpu':
        device = torch.device("cpu")
    elif device == 'cuda':
        device = torch.device("cuda")
    else:
        raise ValueError(f"Unknown device type: {device}")
    
    return device