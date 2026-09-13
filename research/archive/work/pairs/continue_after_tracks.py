"""Continue this active calculation as soon as its complete tracks are saved."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--pair', required=True, choices=['80', '100'])
a = p.parse_args()
folder = root/'work/pairs'/a.pair
state = folder/'continuation_status.json'

def status(stage, **extra):
    payload = dict(pair=a.pair, stage=stage, updated_unix=time.time(), **extra)
    temporary = state.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(state)
    print(json.dumps(payload), flush=True)

status('waiting_for_complete_tracks')
while not (folder/'ptv_tracks.npz').exists():
    time.sleep(5)
for stage, args in [
    ('fitting', ['process_fit_pairs.py', '--pairs', a.pair, '--workers', '4', '--checkpoint-every', '2048']),
    ('finalizing', ['finalize_pairs.py', '--pairs', a.pair]),
]:
    status(stage)
    command = [sys.executable, '-u', str(root/'work/pairs'/args[0])] + args[1:]
    result = subprocess.run(command, cwd=str(root))
    if result.returncode:
        status('failed', failed_stage=stage, return_code=result.returncode)
        raise SystemExit(result.returncode)
status('complete')
