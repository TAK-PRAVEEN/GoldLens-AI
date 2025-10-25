import logging
import os

def setup_logging(log_file: str = 'goldlens.log', level: str = 'INFO'):
    """
    Setup and configure logging for GoldLens AI.

    Parameters:
        log_file (str): Path to the log file (default: 'goldLens.log')
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        logger (logging.Logger): Configured logger instance
    """
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)

    # Set the logging level dynamically
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger('GoldLensAI')
    # logger.info('Logging initialized for GoldLens AI.')

    return logger