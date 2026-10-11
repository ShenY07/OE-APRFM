"""Measure elapsed time for a function call."""

import time


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"The executed time of {func.__name__}: {elapsed:.2e} seconds.")
        return result

    return wrapper
