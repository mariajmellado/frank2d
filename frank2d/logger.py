"""
This module provides a simple logging utility for the Frank2D package. 
It allows for printing informational, warning, and error messages based on a verbosity setting.
"""

import logging

logging.basicConfig(format='%(message)s', force=True)

class Logger():
    def __init__(self, verbose):
        """
        Initializes the Show class.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, messages will be printed; otherwise, they will be ignored. Default is False.
        callback : function, optional
            A callback function to be called with the message. Default is None.
        """
        self._verbose = verbose
    
    def info(self, message):
        """Prints an informational message if verbose is True."""
        logging.info(message)
        if self._verbose:
            print(message)
    
    def warning(self, message):
        """Prints a warning message."""
        logging.warning("[WARNING] " + message)
    
    def error(self, message):
        """Prints an error message and raises a ValueError."""
        logging.error("[ERROR] " + message)
        raise ValueError(message)