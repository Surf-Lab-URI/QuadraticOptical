"""Ordered fork executor; read-only inherited arrays, unchanged numerical calls."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import multiprocessing as mp

_INHERITED_CALLABLE = None


def _run_inherited(argument):
    if _INHERITED_CALLABLE is None:
        raise RuntimeError('No callable was installed before fork.')
    return _INHERITED_CALLABLE(argument)


class ForkProcessExecutor:
    """Ordered map with a fresh fork when the task callable changes.

    Maps must be fully consumed before submitting another map, exactly as the
    current PIV drivers do with list(pool.map(...)). Do not nest this executor
    or invoke its maps concurrently from multiple parent threads.
    """
    def __init__(self, max_workers=4):
        if max_workers < 1:
            raise ValueError('max_workers must be positive')
        self.max_workers = max_workers
        self.pool = None
        self.callable = None

    def __enter__(self):
        return self

    def _close(self, terminate=False):
        if self.pool is not None:
            if terminate:
                self.pool.terminate()
            else:
                self.pool.close()
            self.pool.join()
            self.pool = None

    def map(self, fn, iterable):
        global _INHERITED_CALLABLE
        if self.pool is None or self.callable is not fn:
            self._close()
            _INHERITED_CALLABLE = fn
            self.callable = fn
            self.pool = mp.get_context('fork').Pool(processes=self.max_workers)
        return self.pool.imap(_run_inherited, iterable, chunksize=1)

    def __exit__(self, exc_type, exc_value, traceback):
        self._close(terminate=exc_type is not None)
        return False


