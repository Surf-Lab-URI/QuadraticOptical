"""Describe NCC threshold crossings and fixed-domain integral extrema only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,json
import numpy as np
from finalize_fields import ConservativeEvaluator
from fit_fields import read_npz,atomic_json

HERE=Path(__file__).resolve().parent


def run(pair):
    base=HERE/str(pair);z=read_npz(base/'bootstrap_field_comparison.npz')
    ok=z['integration_primary_accepted'];ncc=z['integration_alt_ncc']
    with np.errstate(invalid='ignore'):bad=ok&~(ncc>=.6)
    q=z['integration_query'][bad]
    r=ConservativeEvaluator(base).evaluate_conservative(q) if len(q) else None
    a=[]
    for k,(i,j) in enumerate(np.argwhere(bad)):
        a.append(dict(depth_m=float(z['depth_m'][i]),query=q[k].tolist(),
            primary_ncc=float(r['ncc'][k]),alt_ncc=float(ncc[i,j]),
            displacement_difference_px=float(z['integration_displacement_difference_px'][i,j]),
            gradient_difference_pixel_per_pixel=z['integration_absolute_gradient_difference'][i,j].tolist(),
            alt_good_patch_share=float(z['integration_alt_good_patch_share'][i,j]),
            primary_fb=float(r['fb'][k]),primary_accepted=bool(r['accepted'][k])))
    report=dict(pair=pair,alt_ncc_below_threshold=a,
        note='Additional alternative-NCC diagnostics are not a new acceptance requirement. Integral extrema use the exact primary domains.')
    for name,key,primary in [
        ('covered','covered_integral_delta_assumed_m2_per_s','primary_covered_integral_assumed_m2_per_s'),
        ('common','common_integral_delta_assumed_m2_per_s','primary_common_integral_assumed_m2_per_s')]:
        d=z[key]
        if not np.isfinite(d).any():
            report[name+'_largest_integral_change']=None;continue
        i=np.nanargmax(abs(d));v=z[primary][i]
        report[name+'_largest_integral_change']=dict(depth_m=float(z['depth_m'][i]),
            delta_assumed_m2_per_s=float(d[i]),primary_assumed_m2_per_s=float(v),
            delta_m2_per_s=float(d[i]),primary_m2_per_s=float(v),
            relative_percent=float(100*abs(d[i]/v)) if v!=0 else None)
    atomic_json(base/'bootstrap_field_quality_exceptions.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['80','100'],required=True)
    run(p.parse_args().pair)
