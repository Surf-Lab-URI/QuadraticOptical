"""Optional fork-based seed evaluation and fitting; numerical drivers unchanged.

Example: python work/image_only_cm/parallel_fit_fields.py --pairs 80 --workers 4

Use only for a new launch or an explicitly coordinated resume. This does not
attach to or modify running jobs. Original PTVModel.evaluate runs unchanged on
ordered query chunks in inherited-model workers. Each seed pool closes before
local image fitting starts; the latter uses the verified ForkProcessExecutor.
Input/code/settings signatures and atomic checkpoints remain those produced by
fit_fields.py, since the numerical model and its inputs are unchanged.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import multiprocessing as mp
import numpy as np

import fit_fields
from ptv_model import PTVModel as OriginalPTVModel
from fork_executor import ForkProcessExecutor

_INHERITED_MODEL = None
_ORIGINAL_SEED_STAGE = fit_fields.seed_stage
_ORIGINAL_RUN_PAIR = fit_fields.run_pair


def _evaluate_chunk(argument):
    query, scale, order = argument
    if _INHERITED_MODEL is None:
        raise RuntimeError('The particle model must be installed before fork.')
    return _INHERITED_MODEL.evaluate(query, scale=scale, order=order)


class ParallelPTVModel:
    """Read-only proxy with a persistent, explicitly closed fork worker pool."""
    def __init__(self, path, max_workers=4):
        if max_workers < 1:
            raise ValueError('max_workers must be positive.')
        self.model = OriginalPTVModel(path)
        self.max_workers = int(max_workers)
        self.pool = None

    @property
    def points(self):
        return self.model.points

    def evaluate(self, query, scale=10, order=2):
        global _INHERITED_MODEL
        query = np.atleast_2d(query)
        if len(query) < 2 or self.max_workers == 1:
            return self.model.evaluate(query, scale=scale, order=order)
        if self.pool is None:
            _INHERITED_MODEL = self.model
            self.pool = mp.get_context('fork').Pool(processes=self.max_workers)
        # np.array_split preserves every source row and its original order.
        chunks = np.array_split(query, min(self.max_workers, len(query)))
        results = self.pool.map(_evaluate_chunk,
                                [(chunk, scale, order) for chunk in chunks], chunksize=1)
        keys = list(results[0])
        if any(list(result) != keys for result in results):
            raise RuntimeError('Original model returned inconsistent result keys.')
        return {key: np.concatenate([result[key] for result in results], axis=0)
                for key in keys}

    def close(self, terminate=False):
        global _INHERITED_MODEL
        if self.pool is not None:
            if terminate:
                self.pool.terminate()
            else:
                self.pool.close()
            self.pool.join()
            self.pool = None
        if _INHERITED_MODEL is self.model:
            _INHERITED_MODEL = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(terminate=exc_type is not None)
        return False


def parallel_seed_stage(directory, inputs, common, checkpoint_every, max_workers=4):
    """Run the untouched stage with a temporary constructor-only substitution."""
    original_constructor = fit_fields.PTVModel
    proxies = []
    def constructor(path):
        model = ParallelPTVModel(path, max_workers=max_workers)
        proxies.append(model)
        return model
    fit_fields.PTVModel = constructor
    failed = True
    try:
        result = _ORIGINAL_SEED_STAGE(directory, inputs, common, checkpoint_every)
        failed = False
        return result
    finally:
        fit_fields.PTVModel = original_constructor
        for model in proxies:
            model.close(terminate=failed)


def run_pair(pair, args):
    original_seed_stage = fit_fields.seed_stage
    def seed_stage(directory, inputs, common, checkpoint_every):
        return parallel_seed_stage(directory, inputs, common, checkpoint_every,
                                   max_workers=args.workers)
    fit_fields.seed_stage = seed_stage
    try:
        return _ORIGINAL_RUN_PAIR(pair, args)
    finally:
        fit_fields.seed_stage = original_seed_stage


def main():
    fit_fields.run_pair = run_pair
    fit_fields.ThreadPoolExecutor = ForkProcessExecutor
    return fit_fields.main()


if __name__ == '__main__':
    raise SystemExit(main())
