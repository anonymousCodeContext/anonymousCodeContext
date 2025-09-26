#!/usr/bin/env python3

import time
import functools
import logging
import threading
from typing import Callable, Any, Optional, Dict
from datetime import datetime
from langchain.callbacks.base import BaseCallbackHandler



class TimerLogger:
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TimerLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.log_file_path = None
            self.logger = None
            self._setup_logger()
    
    def _setup_logger(self):
        self.logger = logging.getLogger('timer_logger')
        self.logger.setLevel(logging.INFO)
        
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        if self.log_file_path:
            handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
        else:
            handler = logging.StreamHandler()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def set_log_file(self, file_path: Optional[str] = None):
        if file_path != self.log_file_path:
            self.log_file_path = file_path
            self._setup_logger()
    
    def log(self, message: str, level: str = "INFO"):
        if not self.logger:
            self._setup_logger()
        
        level = level.upper()
        if level == "DEBUG":
            self.logger.debug(message)
        elif level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        else:
            self.logger.info(message)


# INSERT_YOUR_CODE
class MultiStartEndTimer:

    def __init__(self, name: str = "MultiStartEndTimer"):

        self.name = name
        self._first_start_time = None
        self._last_end_time = None
        self._has_started = False
        self._has_ended = False

    def start(self):
        if not self._has_started:
            self._first_start_time = time.time()
            self._has_started = True
        return self

    def end(self):
        if self._has_started:
            self._last_end_time = time.time()
            self._has_ended = True
        return self

    def get_elapsed_ms(self) -> float:

        if self._has_started and self._has_ended and self._first_start_time is not None and self._last_end_time is not None:
            return (self._last_end_time - self._first_start_time) * 1000
        return 0.0

    def reset(self):
        self._first_start_time = None
        self._last_end_time = None
        self._has_started = False
        self._has_ended = False


class Timer:
    
    def __init__(self, name: str = "Timer"):

        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None
        self.is_running = False
    
    def start(self):
        if self.is_running:
            return self
        
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_ms = None
        self.is_running = True
        
        return self
    
    def end(self) -> float:

        if not self.is_running:
            return 0.0
        
        self.end_time = time.time()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        self.is_running = False
        
        return self.elapsed_ms
    
    def get_elapsed_ms(self) -> Optional[float]:
        return self.elapsed_ms
    
    def get_current_elapsed_ms(self) -> Optional[float]:
        if not self.is_running or self.start_time is None:
            return None
        
        current_time = time.time()
        return (current_time - self.start_time) * 1000
    
    def reset(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None
        self.is_running = False
        
        return self
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


class MultiStartEndTimer:

    def __init__(self, name: str = "MultiTimer"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None

    def start(self):
 
        if self.start_time is None:
            self.start_time = time.time()

        return self

    def end(self) -> float:

        if self.start_time is None:
            return 0.0
        self.end_time = time.time()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return self.elapsed_ms

    def get_elapsed_ms(self) -> Optional[float]:
        return self.elapsed_ms

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None
        return self


class TimeCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        # self.start_time = None
        # self.end_time = None
        self.timer = Timer()

    def on_llm_start(self, *args, **kwargs):
        self.timer.start()

    def on_llm_end(self, *args, **kwargs):
        self.timer.end()
    
    def on_llm_error(self, *args, **kwargs):
        self.timer.end()
    
    def get_elapsed_ms(self):
        return self.timer.get_elapsed_ms()


def timer_decorator(func_name: Optional[str] = None, log_level: str = "INFO"):

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            name = func_name or func.__name__
            logger = TimerLogger()
            
            start_time = time.time()
            start_datetime = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                
                log_msg = (f"[TIMER] {name} - Elapsed time: {elapsed_ms:.2f}ms "
                          f"(Start: {start_datetime.strftime('%H:%M:%S.%f')[:-3]}, "
                          f"End: {datetime.now().strftime('%H:%M:%S.%f')[:-3]})")
                
                logger.log(log_msg, log_level)
                return result
                
            except Exception as e:
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                
                error_msg = (f"[TIMER] {name} - Error occurred, elapsed time: {elapsed_ms:.2f}ms - "
                            f"Error: {str(e)}")
                
                logger.log(error_msg, "ERROR")
                raise
        
        return wrapper
    return decorator


def async_timer_decorator(func_name: Optional[str] = None, log_level: str = "INFO"):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            name = func_name or func.__name__
            logger = TimerLogger()
            
            start_time = time.time()
            start_datetime = datetime.now()
            
            try:
                result = await func(*args, **kwargs)
                
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                
                log_msg = (f"[TIMER] {name} - Elapsed time: {elapsed_ms:.2f}ms "
                          f"(Start: {start_datetime.strftime('%H:%M:%S.%f')[:-3]}, "
                          f"End: {datetime.now().strftime('%H:%M:%S.%f')[:-3]})")
                
                logger.log(log_msg, log_level)
                return result
                
            except Exception as e:
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                
                error_msg = (f"[TIMER] {name} - Error occurred, elapsed time: {elapsed_ms:.2f}ms - "
                            f"Error: {str(e)}")
                
                logger.log(error_msg, "ERROR")
                raise
        
        return wrapper
    return decorator


def timer(func: Callable) -> Callable:
    """Simplified synchronous function timer decorator"""
    return timer_decorator()(func)


def async_timer(func: Callable) -> Callable:
    """Simplified asynchronous function timer decorator"""
    return async_timer_decorator()(func)


def performance_timer(func_name: Optional[str] = None, log_args: bool = False):
    """
    Performance analysis decorator that records more detailed information.
    
    Args:
        func_name: Custom function name.
        log_args: Whether to log function arguments (note: may contain sensitive information).
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            name = func_name or func.__name__
            logger = TimerLogger()
            start_time = time.time()
            start_datetime = datetime.now()
            
            args_info = ""
            if log_args:
                args_str = ", ".join([str(arg) for arg in args])
                kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                args_info = f" - Arguments: ({args_str}{', ' + kwargs_str if kwargs_str else ''})"
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                
                log_msg = f"[PERFORMANCE] {name}{args_info} - Elapsed time: {elapsed_ms:.2f}ms"
                logger.log(log_msg, "INFO")
                
                return result
                
            except Exception as e:
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                
                error_msg = (f"[PERFORMANCE] {name}{args_info} - Error occurred, "
                            f"elapsed time: {elapsed_ms:.2f}ms - Error: {str(e)}")
                logger.log(error_msg, "ERROR")
                raise
        
        return wrapper
    return decorator


def set_timer_log_file(file_path: Optional[str] = None):
    """Set the global timer log file path"""
    logger = TimerLogger()
    logger.set_log_file(file_path)


def create_timer(name: str = "Timer") -> Timer:
    """Create a new timer instance"""
    return Timer(name=name)


__all__ = [
    'Timer',
    'TimerLogger', 
    'timer_decorator',
    'async_timer_decorator',
    'timer',
    'async_timer',
    'performance_timer',
    'set_timer_log_file',
    'create_timer'
]