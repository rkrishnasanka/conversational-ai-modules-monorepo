import logging
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

# Define a function to set up logging
def setup_logger(
    name="app_logger",
    log_dir="logs",
    log_filename="app.log",
    log_level=logging.DEBUG,
    when="midnight",
    backup_count=7,
    fmt="%(asctime)s - %(message)s"
):
    """
    Sets up a logger with both console and TimedRotatingFileHandler.
    
    - `log_dir`: Directory to store logs.
    - `log_filename`: Log file name.
    - `log_level`: Logging level (DEBUG, INFO, etc.).
    - `when`: Rotation time (`midnight`, `H` for hourly, etc.).
    - `backup_count`: Number of rotated logs to keep.
    - `fmt`: Log message format.
    """

    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Full log file path
    log_file_path = log_path / log_filename

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        return logger

    # Define log format
    formatter = logging.Formatter(fmt)

    # Console Handler (for real-time visibility)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # File Handler (TimedRotatingFileHandler for automatic log rotation)
    file_handler = TimedRotatingFileHandler(
        log_file_path, when=when, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    return logger

def setup_default_logging():
    """Configure default logging for the package."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
