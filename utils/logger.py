import logging
from datetime import datetime

# Logger setup
logger = logging.getLogger("GenesisLogger")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_info(message):
    logger.info(message)

def log_error(message):
    logger.error(message)
