# pruning_toolkit/__init__.py
import logging

# Configure basic logging for the library.
# Users of the library can override this configuration.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Initializing pruning_toolkit package.")