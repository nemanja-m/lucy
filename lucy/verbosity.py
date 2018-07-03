import os
from contextlib import redirect_stdout

VERBOSE = False


def set_verbosity(verbose):
    """Turn on/off verbose output."""
    global VERBOSE
    VERBOSE = verbose


def verbose(func):
    """Redirect output to /dev/null if VERBOSE is False."""
    def decorator(*args, **kwargs):
        if not VERBOSE:
            with open(os.devnull, 'w') as void:
                with redirect_stdout(void):
                    return func(*args, **kwargs)
        return func(*args, **kwargs)
    return decorator
