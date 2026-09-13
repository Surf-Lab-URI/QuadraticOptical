"""Run only fresh ±48-track seeds and main fits via unchanged frozen routines."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from types import SimpleNamespace
import argparse,sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'pairs'))
import fit_fields
from process_executor import ForkProcessExecutor


def run(pair,workers=4):
    parent=HERE/str(pair);branch=parent/'bootstrap48'
    if not (branch/'ptv_tracks.npz').is_file():
        raise FileNotFoundError('Complete fresh ±48 tracks are required first.')
    tracks=fit_fields.verify_tracks(branch/'ptv_tracks.npz',fit_fields.file_hash(parent/'inputs.npz'))
    if int(tracks['bootstrap_radius'])!=48:
        raise ValueError('Expected the independent ±48 bootstrap branch.')
    target=branch/'inputs.npz'
    if not target.exists():os.link(str(parent/'inputs.npz'),str(target))
    assert fit_fields.file_hash(target)==fit_fields.file_hash(parent/'inputs.npz')
    alias_root=HERE/'bootstrap48_runs';alias_root.mkdir(exist_ok=True)
    alias=alias_root/str(pair)
    if not alias.exists():alias.symlink_to(branch,target_is_directory=True)
    if alias.resolve()!=branch.resolve():raise ValueError('Unexpected branch alias target.')
    fit_fields.ThreadPoolExecutor=ForkProcessExecutor
    return fit_fields.run_pair(str(pair),SimpleNamespace(base_dir=alias_root,
        stages=['seeds','main'],workers=workers,checkpoint_every=512))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['80','100'],required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();run(args.pair,args.workers)
