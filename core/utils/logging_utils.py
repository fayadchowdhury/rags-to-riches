import logging
from pathlib import Path

def setup_logger(log_name: str, log_dir: str, log_level: int = logging.DEBUG) -> logging.Logger:
    '''
    Take a logger name, log file name and log level
    Return configured logger
    '''

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(f"{log_dir}/{log_name}.log", mode="a")
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger