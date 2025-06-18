import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Union, Dict, Any, List


class ColoredFormatter(logging.Formatter):
    """
    彩色日志格式化器
    """
    # 颜色代码
    COLORS = {
        'BLACK': '\033[0;30m',
        'RED': '\033[0;31m',
        'GREEN': '\033[0;32m',
        'YELLOW': '\033[0;33m',
        'BLUE': '\033[0;34m',
        'MAGENTA': '\033[0;35m',
        'CYAN': '\033[0;36m',
        'WHITE': '\033[0;37m',
        'RESET': '\033[0m'
    }
    
    # 日志级别对应的颜色
    LEVEL_COLORS = {
        'DEBUG': COLORS['BLUE'],
        'INFO': COLORS['GREEN'],
        'WARNING': COLORS['YELLOW'],
        'ERROR': COLORS['RED'],
        'CRITICAL': COLORS['MAGENTA']
    }
    
    def format(self, record):
        """
        格式化日志记录
        """
        # 获取原始格式化的消息
        log_message = super().format(record)
        
        # 添加颜色
        levelname = record.levelname
        if levelname in self.LEVEL_COLORS:
            color_code = self.LEVEL_COLORS[levelname]
            log_message = f"{color_code}{log_message}{self.COLORS['RESET']}"
        
        return log_message


class ProjectLogger:
    """
    项目级别日志记录器，支持控制台和文件输出，可配置日志轮转
    """
    # 单例实例
    _instances: Dict[str, 'ProjectLogger'] = {}
    
    # 日志级别映射
    LEVEL_MAP = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    # 默认日志格式
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def get_instance(cls, name: str = 'project', **kwargs) -> 'ProjectLogger':
        """
        获取日志记录器实例（单例模式）
        
        Args:
            name: 日志记录器名称
            **kwargs: 其他初始化参数
            
        Returns:
            ProjectLogger实例
        """
        if name not in cls._instances:
            cls._instances[name] = cls(name=name, **kwargs)
        return cls._instances[name]
    
    def __init__(self, 
                 name: str = 'project', 
                 level: str = 'info',
                 log_dir: str = 'logs',
                 log_file: Optional[str] = None,
                 format_str: Optional[str] = None,
                 date_format: str = '%Y-%m-%d %H:%M:%S',
                 console_output: bool = True,
                 file_output: bool = True,
                 colored_console: bool = True,
                 date_in_filename: bool = True,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 rotation_unit: str = 'size',  # 'size' or 'time'
                 when: str = 'D',  # 按天轮转
                 encoding: str = 'utf-8'):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别，可选值：debug, info, warning, error, critical
            log_dir: 日志文件目录
            log_file: 日志文件名，如果为None则自动生成
            format_str: 日志格式字符串
            date_format: 日期格式字符串
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            colored_console: 是否使用彩色控制台输出
            date_in_filename: 是否在文件名中包含日期
            max_bytes: 单个日志文件最大字节数（按大小轮转时使用）
            backup_count: 备份文件数量
            rotation_unit: 轮转单位，'size'按大小轮转，'time'按时间轮转
            when: 时间轮转单位，可选值：S秒、M分钟、H小时、D天、W星期、midnight午夜
            encoding: 文件编码
        """
        self.name = name
        self.level = self._get_level(level)
        self.log_dir = log_dir
        self.format_str = format_str or self.DEFAULT_FORMAT
        self.date_format = date_format
        self.console_output = console_output
        self.file_output = file_output
        self.colored_console = colored_console
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.rotation_unit = rotation_unit
        self.when = when
        self.encoding = encoding
        
        # 创建日志目录
        if self.file_output and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 生成日志文件名
        if log_file is None:
            date_str = datetime.now().strftime('%Y%m%d') if date_in_filename else ''
            self.log_file = os.path.join(log_dir, f"{name}_{date_str}.log" if date_in_filename else f"{name}.log")
        else:
            self.log_file = os.path.join(log_dir, log_file)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False  # 避免日志重复输出
        
        # 清除已有的处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 创建格式化器
        standard_formatter = logging.Formatter(self.format_str, datefmt=self.date_format)
        colored_formatter = ColoredFormatter(self.format_str, datefmt=self.date_format) if colored_console else standard_formatter
        
        # 添加控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(colored_formatter if colored_console else standard_formatter)
            self.logger.addHandler(console_handler)
        
        # 添加文件处理器
        if file_output:
            if rotation_unit == 'size':
                # 按大小轮转
                file_handler = RotatingFileHandler(
                    filename=self.log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding=encoding
                )
            else:
                # 按时间轮转
                file_handler = TimedRotatingFileHandler(
                    filename=self.log_file,
                    when=when,
                    backupCount=backup_count,
                    encoding=encoding
                )
            
            file_handler.setLevel(self.level)
            file_handler.setFormatter(standard_formatter)
            self.logger.addHandler(file_handler)
    
    def _get_level(self, level: str) -> int:
        """
        获取日志级别
        
        Args:
            level: 日志级别字符串
            
        Returns:
            日志级别整数值
        """
        return self.LEVEL_MAP.get(level.lower(), logging.INFO)
    
    def debug(self, msg: Any, *args, **kwargs):
        """
        记录DEBUG级别日志
        """
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: Any, *args, **kwargs):
        """
        记录INFO级别日志
        """
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: Any, *args, **kwargs):
        """
        记录WARNING级别日志
        """
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: Any, *args, **kwargs):
        """
        记录ERROR级别日志
        """
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: Any, *args, **kwargs):
        """
        记录CRITICAL级别日志
        """
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: Any, *args, exc_info=True, **kwargs):
        """
        记录异常信息
        """
        self.logger.exception(msg, *args, exc_info=exc_info, **kwargs)
    
    def set_level(self, level: str):
        """
        设置日志级别
        
        Args:
            level: 日志级别字符串
        """
        level_value = self._get_level(level)
        self.logger.setLevel(level_value)
        for handler in self.logger.handlers:
            handler.setLevel(level_value)
    
    def add_file_handler(self, file_path: str, level: str = 'info', **kwargs):
        """
        添加额外的文件处理器
        
        Args:
            file_path: 日志文件路径
            level: 日志级别
            **kwargs: 其他参数
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 创建处理器
        if kwargs.get('rotation_unit', self.rotation_unit) == 'size':
            handler = RotatingFileHandler(
                filename=file_path,
                maxBytes=kwargs.get('max_bytes', self.max_bytes),
                backupCount=kwargs.get('backup_count', self.backup_count),
                encoding=kwargs.get('encoding', self.encoding)
            )
        else:
            handler = TimedRotatingFileHandler(
                filename=file_path,
                when=kwargs.get('when', self.when),
                backupCount=kwargs.get('backup_count', self.backup_count),
                encoding=kwargs.get('encoding', self.encoding)
            )
        
        # 设置级别和格式化器
        handler.setLevel(self._get_level(level))
        handler.setFormatter(logging.Formatter(
            kwargs.get('format_str', self.format_str),
            datefmt=kwargs.get('date_format', self.date_format)
        ))
        
        # 添加到记录器
        self.logger.addHandler(handler)
        
        return handler


# 便捷函数
def get_logger(name: str = 'project', **kwargs) -> ProjectLogger:
    """
    获取日志记录器实例
    
    Args:
        name: 日志记录器名称
        **kwargs: 其他参数
        
    Returns:
        ProjectLogger实例
    """
    return ProjectLogger.get_instance(name=name, **kwargs)


# 使用示例
if __name__ == '__main__':
    # 创建日志记录器
    logger = get_logger(
        name='example',
        level='debug',
        colored_console=True,
        rotation_unit='time',
        when='D'
    )
    
    # 记录不同级别的日志
    logger.debug('这是一条调试日志')
    logger.info('这是一条信息日志')
    logger.warning('这是一条警告日志')
    logger.error('这是一条错误日志')
    logger.critical('这是一条严重错误日志')
    
    # 添加额外的文件处理器
    logger.add_file_handler(
        file_path=os.path.join('logs', 'errors.log'),
        level='error',
        rotation_unit='size',
        max_bytes=5 * 1024 * 1024  # 5MB
    )
    
    # 记录错误日志（会同时写入主日志文件和错误日志文件）
    logger.error('这条错误日志会写入两个文件')
    
    try:
        # 故意引发一个异常
        1 / 0
    except Exception as e:
        # 记录异常信息
        logger.exception(f'发生异常: {e}')