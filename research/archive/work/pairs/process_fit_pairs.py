"""Run the unchanged fitting driver with inherited-array fork workers.

Example: python work/pairs/process_fit_pairs.py --pairs 80 --workers 4
All numerical stages, inputs, signatures, and checkpoints remain those of
fit_pairs.py. Only its context-manager/map executor is rebound at import time.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from process_executor import ForkProcessExecutor
import fit_pairs


if __name__ == '__main__':
    fit_pairs.ThreadPoolExecutor = ForkProcessExecutor
    fit_pairs.main()
