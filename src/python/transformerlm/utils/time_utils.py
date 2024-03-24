import time
from contextlib import contextmanager


class TimeUtils:

    @classmethod
    @contextmanager
    def trace_time(cls, msg: str):
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            print(f"{msg} took {end - start:.4f} seconds")
