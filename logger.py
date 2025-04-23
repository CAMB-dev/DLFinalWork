import logging
import os
from datetime import datetime


def get_logger(logger_name, base_dir='logs') -> logging.Logger:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_dir = os.path.join(base_dir, logger_name)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f'{logger_name}_{timestamp}.log')

    # 设置 logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if not logger.handlers:
        # 文件 handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)

        # 控制台 handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info(f'Log file at: {log_path}')
    return logger
