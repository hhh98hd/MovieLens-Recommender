import time
import tracemalloc
from functools import wraps

def measure_execution_time(func):
    """Decorator to measure the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def measure_memory(func):
    """Decorator to measure the memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Memory usage of {func.__name__}: {current / 10**6:.4f} MB; Peak: {peak / 10**6:.4f} MB")
        return result
    return wrapper