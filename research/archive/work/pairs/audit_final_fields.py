"""Independent, read-only audit of completed fresh-pair fields and exports.

Uses the original scalar LocalField, never the finalizer's evaluation/screening
helpers. Run only when all final model/result files exist; no fitting is done.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import h5py
from scipy.spatial import cKDTree,Delaunay
from scipy.ndimage import map_coordinates
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work'))
from field_model import LocalField

MODEL_NAMES=['main','affine','large_window','margin14','reverse']
TOL=1e-10

def read(path,keys=None):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in (keys if keys is not None else z.files)}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def comparison(a,b,tol=TOL):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return dict(passed=False,reason='shape mismatch',actual_shape=list(a.shape),saved_shape=list(b.shape))
    same_nonfinite=np.array_equal(np.isfinite(a),np.isfinite(b)) and np.array_equal(np.isnan(a),np.isnan(b))
    finite=np.isfinite(a)&np.isfinite(b)
    if a.dtype.kind in 'biu' and b.dtype.kind in 'biu':
        different=int(np.count_nonzero(a!=b));maximum=float(different>0)
    else:
        maximum=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
        different=int(np.count_nonzero(np.abs(a[finite]-b[finite])>tol))
    return dict(passed=bool(same_nonfinite and maximum<=tol),max_absolute_difference=maximum,
                elements_differing_over_tolerance=different,nonfinite_pattern_identical=bool(same_nonfinite),tolerance=tol)

def stats(x):
    a=np.asarray(x);a=a[np.isfinite(a)]
    return dict(n=int(a.size),minimum=float(a.min()),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max())) if a.size else dict(n=0)

def visibility(mask,q,strict_target=False):
    # Independently enforce the original solver's target interpolation domain.
    bounds=np.array(mask.shape[::-1],float)
    inside=np.isfinite(q).all(axis=1)&(q>=0).all(axis=1)&(q<=bounds-1).all(axis=1)
    if strict_target:inside&=(q>=1).all(axis=1)&(q<bounds-2).all(axis=1)
    out=np.zeros(len(q),bool)
    out[inside]=map_coordinates(mask.astype(float),q[inside,::-1].T,order=1,mode='constant',cval=0)>.99
    return out

def valid_share(model,query):
    good=(model.data['mindet']>.05)&(model.data['support_fraction']>.85)
    out=np.zeros(len(query))
    for j,q in enumerate(query):
        if not np.isfinite(q).all():continue
        neighbors=model.tree.query_ball_point(q,model.R)
        if not neighbors:continue
        radius=np.linalg.norm(model.points[neighbors]-q,axis=1)/model.R
        weights=(1-radius)**4*(1+4*radius)
        denominator=weights.sum()
        if denominator>1e-12:out[j]=np.sum(weights*good[neighbors])/denominator
    return out

def recompute(directory,inp,q,max_depth=None):
    models={k:LocalField(directory/(k+'.npz')) for k in MODEL_NAMES}
    d,g,ncc=models['main'].evaluate(q,diagnostics=True)
    print('audit scalar primary evaluated',len(q),flush=True)
    ad=[];ag=[]
    for name in MODEL_NAMES[1:4]:
        a,b=models[name].evaluate(q);ad.append(a);ag.append(b)
        print('audit scalar variant evaluated',name,flush=True)
    ad=np.array(ad);ag=np.array(ag);target=q+d
    reverse,_=models['reverse'].evaluate(target)
    with np.errstate(invalid='ignore'):
        af=np.isfinite(ad).all(axis=(0,2))&np.isfinite(ag).all(axis=(0,2,3))
        spread=np.max(np.linalg.norm(ad-d[None],axis=2),axis=0)
        sxx=np.max(np.abs(ag[:,:,0,0]-g[None,:,0,0]),axis=0)
        syy=np.max(np.abs(ag[:,:,1,1]-g[None,:,1,1]),axis=0)
        fb=np.linalg.norm(d+reverse,axis=1)
    tracks=read(directory/'ptv_tracks.npz',['points','accepted'])
    features=tracks['points'][tracks['accepted'].astype(bool)]
    tree=cKDTree(features);nearest=tree.query(q)[0]
    count=np.array([len(x) for x in tree.query_ball_point(q,25)])
    in_hull=Delaunay(features).find_simplex(q)>=0
    support=in_hull&(nearest<=12)&(count>=6)
    fshare=valid_share(models['main'],q);rshare=valid_share(models['reverse'],target)
    depth=q[:,1]-np.interp(q[:,0],np.arange(len(inp['surface_a'])),inp['surface_a'])
    target_depth=target[:,1]-np.interp(target[:,0],np.arange(len(inp['surface_b'])),inp['surface_b'])
    det=np.full(len(q),np.nan);finite_g=np.isfinite(g).all(axis=(1,2))
    det[finite_g]=np.linalg.det(np.eye(2)[None]+g[finite_g])
    source=visibility(inp['va'],q);destination=visibility(inp['vb'],target,True)
    with np.errstate(invalid='ignore'):
        gates=dict(alternatives_available=af,forward_valid_share=fshare>=.95,
            reverse_valid_share=rshare>=.95,minimum_source_depth=depth>=12,
            target_depth=target_depth>=10,feature_support=support,ncc=ncc>=.6,
            forward_backward=fb<=1,alternative_displacement_spread=spread<=1.5,
            deformation_determinant=det>.2,finite_displacement=np.isfinite(d).all(axis=1))
        if max_depth is not None:gates['maximum_source_depth']=depth<=max_depth
        evidence=np.logical_and.reduce(list(gates.values()))
        accepted=evidence&source&destination
        gradxx=accepted&(depth>=20)&(sxx<=.08)&(nearest<=10)
        gradyy=accepted&(depth>=20)&(syy<=.08)&(nearest<=10)
    out=dict(disp=d,gradient=g,ncc=ncc,fb=fb,depth=depth,target_depth=target_depth,
        method_spread=spread,gradient_spread_xx=sxx,gradient_spread_yy=syy,
        alternatives_available=af,local_valid_share=fshare,reverse_valid_share=rshare,
        support=support,nearest_feature=nearest,feature_count25=count,determinant=det,
        source_visible=source,target_visible=destination,evidence_pass=evidence,
        accepted=accepted,gradient_accepted_xx=gradxx,gradient_accepted_yy=gradyy,
        alternative_displacements=ad,alternative_gradients=ag,in_feature_hull=in_hull)
    return out,models,gates

def stratified_indices(q,depth,shape,finite,target=2000):
    """Fixed deterministic spatial sampling, enriched at the mask and edges."""
    rng=np.random.RandomState(20260911)
    priority=rng.permutation(len(q));rank=np.empty(len(q),int);rank[priority]=np.arange(len(q))
    chosen=set();categories={}
    def add(name,mask,n):
        ii=np.flatnonzero(mask&finite);ii=ii[np.argsort(rank[ii])[:n]]
        chosen.update(ii.tolist());categories[name]=ii
    height,width=shape
    for ix in range(8):
        for iy in range(8):
            mask=(q[:,0]>=width*ix/8)&(q[:,0]<width*(ix+1)/8)&(q[:,1]>=height*iy/8)&(q[:,1]<height*(iy+1)/8)
            add('spatial_%d_%d'%(ix,iy),mask,28)
    add('source_depth_below40',depth<40,256)
    add('source_depth40to80',(depth>=40)&(depth<80),128)
    add('left_image_edge',q[:,0]<32,160)
    add('right_image_edge',q[:,0]>width-33,160)
    add('bottom_image_edge',q[:,1]>height-33,160)
    # Also include all deepest eighth and each full-width quarter in reporting.
    remaining=target-len(chosen)
    if remaining>0:
        unseen=finite.copy()
        if chosen:unseen[list(chosen)]=False
        add('remaining_uniform',unseen,remaining)
    indices=np.array(sorted(chosen),int)
    coverage={name:int(np.isin(indices,ii).sum()) for name,ii in categories.items() if not name.startswith('spatial_')}
    coverage['width_quarters']=[int(np.sum((q[indices,0]>=width*j/4)&(q[indices,0]<width*(j+1)/4))) for j in range(4)]
    coverage['vertical_eighths']=[int(np.sum((q[indices,1]>=height*j/8)&(q[indices,1]<height*(j+1)/8))) for j in range(8)]
    return indices,coverage

def gradient_check(model,q,depth,shape,DX,DT,minimum=1000):
    value,gradient=model.evaluate(q)
    finite=np.isfinite(value).all(axis=1)&np.isfinite(gradient).all(axis=(1,2))
    ids,coverage=stratified_indices(q,depth,shape,finite)
    xy=q[ids];gg=gradient[ids];runs=[]
    for eps in [1e-2,1e-3]:
        plusx=model.evaluate(xy+[eps,0])[0];minusx=model.evaluate(xy-[eps,0])[0]
        plusy=model.evaluate(xy+[0,eps])[0];minusy=model.evaluate(xy-[0,eps])[0]
        fd=np.stack([(plusx-minusx)/(2*eps),(plusy-minusy)/(2*eps)],axis=2)
        valid=np.isfinite(fd).all(axis=(1,2));err=fd[valid]-gg[valid]
        # Independent SI finite differences: horizontal u/X and upward w/Z.
        du_dx=(plusx[:,0]*DX/DT-minusx[:,0]*DX/DT)/(2*eps*DX)
        # Increasing physical Z means decreasing image y; w=-dy*DX/DT.
        dw_dz=((-minusy[:,1]*DX/DT)-(-plusy[:,1]*DX/DT))/(2*eps*DX)
        record=dict(step_pixels=eps,valid_queries=int(valid.sum()),
            invalid_perturbed_queries=int((~valid).sum()),all_four_entries_absolute_error=stats(np.abs(err)),
            Gxx_absolute_error=stats(np.abs(fd[valid,0,0]-gg[valid,0,0])),
            Gyy_absolute_error=stats(np.abs(fd[valid,1,1]-gg[valid,1,1])),
            direct_SI_du_dx_absolute_error=stats(np.abs(du_dx[valid]-gg[valid,0,0]/DT)),
            direct_SI_dw_dz_absolute_error=stats(np.abs(dw_dz[valid]-gg[valid,1,1]/DT)))
        record['passed']=bool(valid.sum()>=minimum and np.max(np.abs(err))<1e-5 and
            np.max(np.abs(du_dx[valid]-gg[valid,0,0]/DT))<1e-5/DT and
            np.max(np.abs(dw_dz[valid]-gg[valid,1,1]/DT))<1e-5/DT)
        runs.append(record)
    return dict(passed=all(x['passed'] for x in runs),sampled_queries=len(ids),minimum_required=minimum,
                index0=ids.tolist(),coverage=coverage,tests=runs),dict(indices=ids,query=xy,analytic_gradient=gg)

def native_check(pair,inp,q,results,recomputed):
    path=Path('historical-user-files/Downloads')/('ExpLCL_1_03_%d_PIV.mat'%pair)
    with h5py.File(path,'r') as f:
        x=f['compVel/xPIV'][:].ravel()-1;y=f['compVel/zPIV'][:].ravel()-1
        ix=np.searchsorted(x,q[:,0]);iy=np.searchsorted(y,q[:,1]);exact=(ix<len(x))&(iy<len(y))
        ii=np.flatnonzero(exact);exact[ii]&=(x[ix[ii]]==q[ii,0])&(y[iy[ii]]==q[ii,1])
        disp=np.full((len(q),2),np.nan);cor=np.full(len(q),np.nan)
        disp[exact,0]=f['compVel/delta_x'][:][ix[exact],iy[exact]]
        disp[exact,1]=-f['compVel/delta_z'][:][ix[exact],iy[exact]]
        cor[exact]=f['compVel/dcor'][:][ix[exact],iy[exact]]
    checks={'input_native_at_grid':comparison(disp,inp['classical_at_points']),
            'input_native_correlation_at_grid':comparison(cor,inp['classical_dcor_at_points']),
            'saved_native_at_grid':comparison(disp,results['classical_displacement_native_at_grid'])}
    # Classical targets use actual image bounds, not the hybrid solver's strict
    # interpolation/gradient border. This native field was fitted elsewhere.
    source=visibility(inp['va'],q);target=visibility(inp['vb'],q+disp,False)
    vector_finite=np.isfinite(disp).all(axis=1);cor_finite=np.isfinite(cor)
    available=exact&vector_finite&source&target
    error=np.linalg.norm(recomputed['disp']-disp,axis=1)
    expected=dict(classical_ncc_at_grid=cor,classical_exact_node_at_grid=exact,
        classical_source_visible_at_grid=source,classical_target_visible_at_grid=target,
        classical_finite_ncc_at_grid=cor_finite,classical_available_at_grid=available,
        classical_comparison_error=error)
    for key,value in expected.items():
        checks[key]=comparison(value,results[key]) if key in results else dict(passed=False,reason='missing result key')
    native=dict(disp=disp,dcor=cor,exact=exact,source_visible=source,target_visible=target,
                finite_vector=vector_finite,finite_correlation=cor_finite,available=available,error=error)
    return checks,native

def export_check(pair,inp,recomputed,native,require=False):
    directory=ROOT/'outputs'/('pair_%d'%pair)
    paths=[directory/'conservative_field.npz',directory/'conservative_field.mat']
    if not all(p.exists() for p in paths):
        return dict(status='missing',passed=not require,required=require,missing=[str(p) for p in paths if not p.exists()])
    d=recomputed['disp'].copy();d[~recomputed['accepted']]=np.nan
    xx=recomputed['gradient'][:,0,0].copy();xx[~recomputed['gradient_accepted_xx']]=np.nan
    yy=recomputed['gradient'][:,1,1].copy();yy[~recomputed['gradient_accepted_yy']]=np.nan
    DX=float(inp['DX']);DT=float(inp['DT'])
    expected=dict(displacement_raw_px_per_pair=recomputed['disp'],displacement_conservative_px_per_pair=d,
        displacement_gradient_raw=recomputed['gradient'],du_dx_pixel_gradient_conservative=xx,
        dw_dz_pixel_gradient_conservative=yy,velocity_uw_conservative_assumed_mps=d*np.array([1,-1])*DX/DT,
        du_dx_conservative_assumed_per_s=xx/DT,dw_dz_conservative_assumed_per_s=yy/DT,
        accepted_vector=recomputed['accepted'].astype(np.uint8),accepted_du_dx=recomputed['gradient_accepted_xx'].astype(np.uint8),
        accepted_dw_dz=recomputed['gradient_accepted_yy'].astype(np.uint8),classical_native_displacement_at_grid_px=native['disp'],
        classical_native_available_at_grid=native['available'].astype(np.uint8),
        comparison_endpoint_difference_px=native['error'],comparison_available=(native['available']&recomputed['accepted']).astype(np.uint8),
        grid_xy_zero_based_px=inp['points'],grid_xy_matlab_one_based_px=inp['points']+1)
    reports={}
    for path in paths:
        saved=read(path) if path.suffix=='.npz' else loadmat(path,squeeze_me=True)
        reports[path.suffix]={k:comparison(v,saved[k]) if k in saved else dict(passed=False,reason='missing key') for k,v in expected.items()}
    return dict(status='checked',passed=all(v['passed'] for f in reports.values() for v in f.values()),checks=reports,
        sign_convention='u=dx DX/DT; w=-dy DX/DT; X=x DX and Z=-y DX, so both diagonal derivatives are Gii/DT.')

def run(pair,require_exports=False,self_test=False):
    started=time.time();directory=ROOT/'work'/('full_width' if self_test else 'pairs/%d'%pair)
    needed=[directory/(k+'.npz') for k in MODEL_NAMES]+[directory/'inputs.npz',directory/'results.npz',directory/'ptv_tracks.npz']
    missing=[str(p) for p in needed if not p.exists()]
    if missing:raise FileNotFoundError('Audit requires completed files; it does not poll: '+', '.join(missing))
    initial_hash={p.name:sha(p) for p in needed}
    inp=read(directory/'inputs.npz',['va','vb','availability_a','availability_b','surface_a','surface_b','points','origin0','DX','DT']+
             ([] if self_test else ['classical_at_points','classical_dcor_at_points']))
    saved=read(directory/'results.npz');q=inp['points']
    if self_test:
        # A broad existing-field subset exercises the same audit logic cheaply.
        idx=np.unique(np.linspace(0,len(q)-1,1600).astype(int));q=q[idx]
        result_idx=200+idx
        oldn=len(saved['query'])
        saved={k:(v[:,result_idx] if k.startswith('alternative_') and v.ndim>1 else v[result_idx])
                   if isinstance(v,np.ndarray) and ((k.startswith('alternative_') and v.ndim>1 and v.shape[1]==oldn) or (v.ndim>0 and v.shape[0]==oldn)) else v for k,v in saved.items()}
        saved['gradient_spread_xx']=saved['gradient_spread'];saved['gradient_accepted_xx']=saved['gradient_accepted']
    else:
        if not np.array_equal(saved['query'],q):raise ValueError('Result queries must equal the entire prepared grid.')
        if not np.array_equal(saved['query_full'],q+inp['origin0']):raise ValueError('Full-image result coordinates disagree with origin.')
    recomputed,models,gates=recompute(directory,inp,q,354 if self_test else None)
    comparisons={}
    for k,v in recomputed.items():
        if k=='in_feature_hull':continue
        if self_test and k in ['gradient_spread_yy','gradient_accepted_yy']:continue
        comparisons[k]=comparison(v,saved[k]) if k in saved else dict(passed=False,reason='missing result key')
    derivatives,derivative_arrays=gradient_check(models['main'],q,recomputed['depth'],inp['va'].shape,float(inp['DX']),float(inp['DT']))
    native_report={};native=None
    if not self_test:
        native_report,native=native_check(pair,inp,q,saved,recomputed)
    export=export_check(pair,inp,recomputed,native,require_exports) if not self_test else dict(status='not applicable',passed=True)
    final_hash={p.name:sha(p) for p in needed}
    counts={k:int(recomputed[k].sum()) for k in ['accepted','gradient_accepted_xx','gradient_accepted_yy','evidence_pass']}
    report=dict(pair=pair if not self_test else 'legacy123_subset_self_test',all_grid_nodes_audited=len(q),
        implementation='Original scalar work/field_model.py LocalField; independently reconstructed diagnostics and gates.',
        model_and_input_hashes=initial_hash,inputs_and_results_unchanged=initial_hash==final_hash,
        model_recomputed_comparisons=comparisons,finite_difference_gradient_check=derivatives,
        native_PIV_checks=native_report,export_checks=export,counts=counts,
        criterion_failure_counts={k:int((~v).sum()) for k,v in gates.items()},
        no_maximum_depth_applied=not self_test,elapsed_seconds=time.time()-started)
    report['passed']=bool(initial_hash==final_hash and all(v['passed'] for v in comparisons.values()) and derivatives['passed'] and
        all(v['passed'] for v in native_report.values()) and export['passed'])
    out=ROOT/'work/pairs' if self_test else directory
    prefix='final_fields_audit_self_test' if self_test else 'final_fields_audit'
    (out/(prefix+'.json')).write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/(prefix+'_gradient_sample.npz'),**derivative_arrays)
    print(json.dumps({k:v for k,v in report.items() if k not in ['model_and_input_hashes','model_recomputed_comparisons','finite_difference_gradient_check','native_PIV_checks','export_checks']},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',type=int,choices=[80,100]);p.add_argument('--self-test',action='store_true');p.add_argument('--require-exports',action='store_true');a=p.parse_args()
    if not a.self_test and a.pair is None:p.error('--pair is required unless --self-test is used')
    result=run(a.pair,a.require_exports,a.self_test)
    if not result['passed']:raise SystemExit(1)
