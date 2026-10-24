
"""Utility functions."""

import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_env_var(key: str, default: Optional[str] = None) -> str:
    """
    Retrieve a value from environment variables, using a default value if provided.

    Args:
        key: The name of the environment variable.
        default: The default value, if provided.

    Returns:
        The environment variable value or the default value.
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(
            f"Environment variable {key} is not set and no default value was provided."
        )
    return value
