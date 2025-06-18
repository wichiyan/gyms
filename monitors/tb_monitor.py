from torch.utils.tensorboard import SummaryWriter
from box import Box

class TBMonitor:
    def __init__(self,env,agent,config,name):
        
        self.env = env
        self.agent = agent
        self.config = config
        self.name = name
        self.run_info = Box()
        log_dir = config['monitor_logging'].get('log_dir',None)
        self.writer = SummaryWriter(log_dir+self.name)

    def monitor(self):
        #每一个episode结束后，此处需要统一记录以下信息：
        #1、环境返回的info，里面包含当前episode的累计奖励、总步长、总时间
        #2、agent的探索率
        monitot_interval = self.config['monitor_logging'].get('monitor_interval',1)
        
        current_episode = self.agent.episode
        if current_episode % monitot_interval == 0:
            total_reward = self.run_info['episode'].get('r',0)
            episode_length = self.run_info['episode'].get('l',0)
            episode_time = self.run_info['episode'].get('t',0)
            explore_rate = self.agent.explore_rate
            mean_losses = self.run_info['mean_episode_loss']
            
            mode = 'train' if self.agent.training else 'eval'
            
            self.writer.add_scalar(f'{mode}/total_reward',total_reward,current_episode)
            self.writer.add_scalar(f'{mode}/mean_episode_loss',mean_losses,current_episode)
            self.writer.add_scalar(f'{mode}/episode_length',episode_length,current_episode)
            self.writer.add_scalar(f'{mode}/episode_time',episode_time,current_episode)
            self.writer.add_scalar(f'{mode}/explore_rate',explore_rate,current_episode)
        
        