import logging
from rich.logging import RichHandler
from src.config import Config

def setup_logger(name: str = "project") -> logging.Logger:
    """Setup a dual-target logger writing to console (Rich) and log file.
    
    Args:
        name: Name of the logger instance.
        
    Returns:
        logging.Logger: Styled and structured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicating handlers if this is called multiple times
    if logger.handlers:
        return logger

    # Ensure log directories exist
    Config.create_required_dirs()

    # Formatter for log file
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler - details logged down to DEBUG index level
    file_handler = logging.FileHandler(Config.LOG_FILE_PATH, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console Handler - prints clean visual info/warnings/errors using Rich
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False
    )
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Single default logger export
logger = setup_logger()
