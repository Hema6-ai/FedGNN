import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "results", level: int = logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{timestamp}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
