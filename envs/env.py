import copy

import gymnasium as gym
from gymnasium.wrappers import RecordVideo,RecordEpisodeStatistics

#封装环境类
class Env:
    def __init__(self,config,mode):
        '''
        self wrapped env class
        agrs:
            config: dict,project config,dict like
            mode: str, can be train,eval
        '''
        super().__init__()
        
        assert mode in ('train','eval'), f'mode must be one of  train or eval, not {mode}'
        env_kwagrs = copy.deepcopy(config['env'])
        #根据mode判断需要以什么配置参数创建环境
        if config[mode].get('record_video', False) :
            env_kwagrs['render_mode'] ='rgb_array'
            self.env = self._make_env(**env_kwagrs)
            
            video_dir = config[mode].get('record_video_dir','./videos')
            record_every_episode = config[mode].get('record_every_episode',500)
            
            trigger = lambda t: (t+1) % record_every_episode == 0
            self.env = RecordVideo(self.env, video_folder=video_dir,episode_trigger=trigger, disable_logger=True)
        else:
            render_mode = config[mode].get('render_mode', None)
            env_kwagrs['render_mode'] = render_mode
            self.env = self._make_env(**env_kwagrs)
        
        #封装统一记录episode统计信息    
        self.env = RecordEpisodeStatistics(self.env)
        
                
    def step(self,action):
        next_state,reward,terminated,truncated,info = self.env.step(action)
        done = terminated or truncated
        return next_state,reward,done,info
    
    def _make_env(self,**kwargs):
        try:
            return gym.make(**kwargs)
        except gym.error.Error as e:
            raise ValueError(f"Failed to create the environment {self.env_name}: {e}")
            
    def __getattr__(self, name):
        return getattr(self.env, name)
        
