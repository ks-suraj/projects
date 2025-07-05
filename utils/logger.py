import logging
import os

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Configure Logger
logging.basicConfig(
    filename="logs/genesis_ai.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

def log_info(message):
    print(f"[INFO] {message}")  # Also print in terminal
    logging.info(message)

def log_error(message):
    print(f"[ERROR] {message}")
    logging.error(message)
