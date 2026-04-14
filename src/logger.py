import logging
import multiprocessing
from logging.handlers import QueueHandler, QueueListener


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that writes only to the log file.
    (Console output is now handled by print() for milestones)
    """
    def emit(self, record):
        # This handler is effectively disabled for console output
        # as we want all detailed logging to go only to the file.
        pass

def setup_logging(verbose: bool = False):
    """
    Sets up the logging system.
    Returns the log queue and the listener.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = logging.Formatter('%(asctime)s [%(levelname)s] [PID:%(process)d] %(message)s')

    # 1. Create the Queue for inter-process communication
    log_queue = multiprocessing.Queue(-1)

    # 2. Setup handlers for the Main Process (Listener)
    # We remove the TqdmLoggingHandler from the listener to prevent logs in CLI

    # File handler - the only destination for structured logs
    file_handler = logging.FileHandler('photo_organizer.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(log_format)

    # 3. Create and start the QueueListener
    # Only file_handler is passed here
    listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
    listener.start()

    # Configure the main process's root logger to use the QueueHandler
    root = logging.getLogger()
    root.setLevel(log_level)

    # Clear existing handlers to avoid duplicate logs
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    root.addHandler(QueueHandler(log_queue))

    # Silence noisy third-party libraries in the main process
    _silence_noisy_loggers()

    return log_queue, listener

def worker_logging_config(queue):
    """
    Configures the worker process to send all logs to the main process queue.
    """
    h = QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG) # Let the listener decide the final level

    # Silence noisy third-party libraries in the worker process
    _silence_noisy_loggers()

def _silence_noisy_loggers():
    """Helper to set noisy library log levels to WARNING."""
    noisy_loggers = ['PIL', 'face_recognition', 'matplotlib', 'urllib3', 'requests']
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
