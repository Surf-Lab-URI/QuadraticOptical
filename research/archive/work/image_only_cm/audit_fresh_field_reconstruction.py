"""Reconstruct fresh image-only fields/flags with original scalar algorithms.

Only algorithm definitions are reused from the earlier audit module. Its run,
native-PIV and legacy-self-test functions are never called. Every data path in
this audit is confined to the current image_only_cm/<pair> directory.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'work/pairs'))
from audit_final_fields import recompute,comparison,gradient_check,sha

def load(path,keys=None):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in (keys if keys is not None else z.files)}

def run(pair):
    started=time.time();directory=HERE/str(pair)
    allowed=['va','vb','surface_a','surface_b','points','origin0','DX','DT','image_only','supplied_velocity_used']
    inp=load(directory/'inputs.npz',allowed)
    if not bool(inp['image_only']) or bool(inp['supplied_velocity_used']):raise ValueError('Fresh image-only inputs required.')
    r=load(directory/'results.npz');p=load(directory/'integration_profile.npz')
    data_paths=[directory/(k+'.npz') for k in ['inputs','results','integration_profile','main','affine','large_window','margin14','reverse','ptv_tracks']]
    before={v.name:sha(v) for v in data_paths}
    q=r['query'];maximum=.01/float(inp['DX'])+1e-9
    fresh,models,_=recompute(directory,inp,q,maximum)
    checks={k:comparison(v,r[k]) for k,v in fresh.items() if k!='in_feature_hull'}

    # Deterministic broad sampling plus both outer image edges at every depth,
    # the inferred-surface row, common-domain start and exact 1cm row.
    x=p['horizontal_sample_x_px'];h=p['depth_m'];ny=len(h);nx=len(x)
    sample=set(np.linspace(0,ny*nx-1,1800).astype(int).tolist())
    for row in range(ny):
        for col in [0,1,nx-2,nx-1]:sample.add(row*nx+col)
    common_row=int(np.argmin(np.abs(h-20*float(inp['DX']))))
    for row in [0,common_row,ny-1]:
        sample.update((row*nx+np.linspace(0,nx-1,180).astype(int)).tolist())
    ids=np.array(sorted(sample),int);rows=ids//nx;cols=ids%nx
    curveq=np.c_[x[cols],p['sample_query_y_px'][rows,cols]]
    curve,_,_=recompute(directory,inp,curveq,maximum)
    curve_checks=dict(displacement= comparison(curve['disp'],p['sample_displacement_px'][rows,cols]),
        accepted= comparison(curve['accepted'],p['sample_accepted'][rows,cols]),
        u_units=comparison(curve['disp'][:,0]*float(inp['DX'])/float(inp['DT']),p['sample_primary_u_assumed_m_per_s'][rows,cols]),
        variant_u_units=comparison(curve['alternative_displacements'][:,:,0]*float(inp['DX'])/float(inp['DT']),p['sample_variant_u_assumed_m_per_s'][:,rows,cols]))
    fd,fdarrays=gradient_check(models['main'],q,fresh['depth'],inp['va'].shape,float(inp['DX']),float(inp['DT']))
    after={v.name:sha(v) for v in data_paths}
    report=dict(pair=pair,passed=all(v['passed'] for v in checks.values()) and all(v['passed'] for v in curve_checks.values()) and fd['passed'] and before==after,
        regular_grid_nodes=len(q),regular_grid_checks=checks,curve_query_count=len(curveq),curve_query_checks=curve_checks,
        derivative_check=fd,counts={k:int(fresh[k].sum()) for k in ['accepted','gradient_accepted_xx','gradient_accepted_yy']},
        data_hashes=before,data_unchanged=before==after,elapsed_seconds=time.time()-started,
        prior_prediction_or_supplied_velocity_arrays_read=False,
        algorithm_note='Original LocalField and independently written evidence reconstruction; only fresh image_only_cm data supplied to those generic routines.')
    (directory/'independent_field_reconstruction.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(directory/'independent_field_reconstruction_samples.npz',curve_flat_indices=ids,curve_query=curveq,**{'fd_'+k:v for k,v in fdarrays.items()})
    print(json.dumps({k:report[k] for k in ['pair','passed','regular_grid_nodes','curve_query_count','counts','data_unchanged','elapsed_seconds']},indent=2),flush=True)
    if not report['passed']:raise SystemExit(1)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',type=int,choices=[80,100],required=True);a=p.parse_args();run(a.pair)
