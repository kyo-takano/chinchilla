"""
Contains a utility function `get_logger`. This module also filters out noisy debug messages 
from `matplotlib` and suppresses redundant warnings from `numpy` and `matplotlib`.
"""

from __future__ import annotations  # PEP 604 backport

import logging
import warnings

from rich.console import Console
from rich.logging import RichHandler

# Ignore the noisy matplotlib debug messages.
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# For NumPy's precision errors & Matplotlib with non-GUI backend
warnings.filterwarnings("once", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_logger(level: int | str, name: str) -> logging.Logger:
    """
    Sets up a logger with the specified log level.
    This logger uses RichHandler for `rich` formatted logging output to the console.

    Args:
        level (int | str): Logging level, e.g., 20 or logging.INFO, 30 or logging.WARNING.
        name (str, optional): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Configure the logger for this library
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(stderr=True), markup=True)],
        force=True,
    )
    # Hook warnings to logger
    logging.captureWarnings(True)
    return logging.getLogger(name)
