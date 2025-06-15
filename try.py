import os
import yaml
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    YAML配置文件加载器
    提供加载、访问和管理YAML配置文件的功能
    """
    def __init__(self, config_path: str):
        """
        初始化配置加载器
        
        Args:
            config_path: YAML配置文件的路径
        """
        self.config_path = config_path
        self.config_data = {}
        self.load_config()
    
    def load_config(self) -> None:
        """
        加载YAML配置文件
        如果文件不存在或格式错误，将抛出相应异常
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
                
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config_data = yaml.safe_load(file)
                
            if self.config_data is None:
                self.config_data = {}
                
        except yaml.YAMLError as e:
            raise ValueError(f"YAML格式错误: {str(e)}")
        except Exception as e:
            raise Exception(f"加载配置文件时出错: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项的值
        
        Args:
            key: 配置项的键，支持点号分隔的嵌套键 (例如 'database.host')
            default: 如果键不存在，返回的默认值
            
        Returns:
            配置项的值，如果键不存在则返回默认值
        """
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有配置项
        
        Returns:
            包含所有配置项的字典
        """
        return self.config_data
    
    def reload(self) -> None:
        """
        重新加载配置文件
        """
        self.load_config()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件的简便函数
    
    Args:
        config_path: YAML配置文件的路径
        
    Returns:
        包含配置数据的字典
        
    Raises:
        FileNotFoundError: 如果配置文件不存在
        ValueError: 如果YAML格式错误
        Exception: 其他加载错误
    """
    loader = ConfigLoader(config_path)
    return loader.get_all()


# 使用示例
if __name__ == "__main__":
    try:
        # 方法1：使用ConfigLoader类
        config_loader = ConfigLoader("config.yaml")
        
        # 获取配置项
        db_host = config_loader.get("database.host", "localhost")
        port = config_loader.get("server.port", 8080)
        
        print(f"数据库主机: {db_host}")
        print(f"服务器端口: {port}")
        
        # 获取所有配置
        all_config = config_loader.get_all()
        print("所有配置:", all_config)
        
        # 方法2：使用简便函数
        config_data = load_yaml_config("config.yaml")
        print("通过函数加载的配置:", config_data)
        
    except Exception as e:
        print(f"错误: {str(e)}")