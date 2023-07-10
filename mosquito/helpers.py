import contextlib
import logging
import os
import timeit


def get_dir(*paths) -> str:
    directory = os.path.join(*paths)
    os.makedirs(directory, exist_ok=True)
    return directory

@contextlib.contextmanager
def time_activity(activity_name: str):
    logging.info("[Timing] %s started.", activity_name)
    start = timeit.default_timer()
    yield
    duration = timeit.default_timer() - start
    
    logging.info("[Timing] %s finished (Took %.2fs).", activity_name, duration)