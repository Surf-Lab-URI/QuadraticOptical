"""Check initialization sensitivity at the exact saved figure sample points."""
from pathlib import Path
import argparse,json
import numpy as np
from compare_bootstrap_field import compare,summarize
from finalize_fields import ConservativeEvaluator,FastLocalField
from fit_fields import read_npz,file_hash,atomic_npz,atomic_json

HERE=Path(__file__).resolve().parent


def run(pair):
    base=HERE/str(pair);path=base/'plot_samples.npz';hash_before=file_hash(path)
    a=read_npz(path);e=ConservativeEvaluator(base);r=e.evaluate_conservative(a['query'])
    for k in ['accepted','gradient_accepted_xx','gradient_accepted_yy']:
        assert np.array_equal(a[k],r[k]),k
    for k in ['disp','gradient']:
        assert np.allclose(a[k],r[k],rtol=0,atol=1e-12,equal_nan=True),k
    alt=FastLocalField(base/'bootstrap48/main.npz');c=compare(a['query'],a,alt)
    summary=summarize(a['query'],a,c)
    summary.update(pair=pair,status='passed',plot_samples_sha256=hash_before,
        original_plot_values_and_flags_reproduced=True,
        gradient_difference_units='pixel/pixel; divide by DT for 1/s',
        note='Relevant diagonal-gradient gates are component-specific; raw cross derivatives are not promoted.')
    arrays=dict(query=a['query'],primary_displacement=a['disp'],primary_gradient=a['gradient'],
        primary_accepted=a['accepted'],primary_gradient_xx_accepted=a['gradient_accepted_xx'],
        primary_gradient_yy_accepted=a['gradient_accepted_yy'],x_axis_px=a['x_axis_px'],depth_axis_px=a['depth_axis_px'],
        DX=a['DX'],DT=a['DT'],**c)
    assert file_hash(path)==hash_before
    atomic_npz(base/'bootstrap_plot_comparison.npz',**arrays)
    atomic_json(base/'bootstrap_plot_comparison.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='greatest_accepted_differences'},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['80','100'],required=True)
    run(p.parse_args().pair)
