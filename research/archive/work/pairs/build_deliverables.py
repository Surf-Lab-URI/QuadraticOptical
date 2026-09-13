"""Export both completed pairs with a shared diagonal-gradient color scale."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
from export_pairs import run as export
from report_pairs import run as report

ROOT=Path(__file__).resolve().parents[2]
values=[]
for pair in [80,100]:
    with np.load(ROOT/'work/pairs'/str(pair)/'results.npz') as f:
        gradient=f['gradient'];dt=float(f['DT'])
        for key,i in [('gradient_accepted_xx',0),('gradient_accepted_yy',1)]:
            values.append(np.abs(gradient[f[key].astype(bool),i,i])/dt)
combined=np.concatenate(values)
limit=float(np.ceil(max(.2,float(np.percentile(combined,99.5)))*5)/5) if len(combined) else .2
records=[]
for pair in [80,100]:
    records.append(export(pair,limit));report(pair)
(ROOT/'outputs/pairs_80_100_summary.json').write_text(json.dumps(dict(pairs=records,shared_gradient_color_limit_assumed_per_s=limit),indent=2))
print('Both pairs exported with shared gradient limit:',limit,flush=True)
