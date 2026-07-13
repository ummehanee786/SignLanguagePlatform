"""
logging_config.py

Basic logging setup for the backend. Instead of scattering print()
statements everywhere, the rest of the app can do:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Something happened")

and it will be consistently formatted and routed through here.
"""

import logging

from app.core.config import settings


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(settings.app_name)
    logger.info("Logging configured. Debug mode: %s", settings.debug)
    return logger