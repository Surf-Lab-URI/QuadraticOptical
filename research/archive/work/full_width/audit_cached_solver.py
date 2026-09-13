"""Check cached-gradient solver against untouched original fit_local."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,inspect,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work'))
from extended_flow import fit_local as original
from fit_local_cached import fit_local as cached

o=inspect.getsource(original)
c=inspect.getsource(cached).replace('stride=1,target_gradient=None,target_mask_float=None):','stride=1):')
c=c.replace('by,bx=np.gradient(B) if target_gradient is None else target_gradient','by,bx=np.gradient(B)')
c=c.replace('    target_mask_float=vb.astype(float) if target_mask_float is None else target_mask_float\n','')
c=c.replace('map_coordinates(target_mask_float,coordinates','map_coordinates(vb.astype(float),coordinates')
assert o==c, 'Cached helper differs beyond the documented cache substitutions'
z=np.load(ROOT/'work/large_frame/inputs.npz')
f=np.load(ROOT/'work/large_frame/main.npz')
A,B,va,vb=[z[k] for k in ['A','B','va','vb']]
grads={'forward':np.gradient(B),'reverse':np.gradient(A)}
report=[]
for i,r,order,direction in [(0,13,2,'forward'),(100,13,1,'forward'),
    (350,19,2,'forward'),(900,13,2,'forward'),(1500,13,2,'reverse'),
    (2183,19,1,'reverse')]:
    pp=f['params'][i]
    seed=np.zeros((2,3));seed[:,0]=pp[:,0]+[.31,-.24]
    seed[:,1:3]=pp[:,1:3]*r/13*.8
    point=f['points'][i]
    aa,bb,vv,ww=A,B,va,vb
    if direction=='reverse':
        aa,bb,vv,ww=B,A,vb,va
        point=point+pp[:,0]
        seed[:,0]=-pp[:,0]+[.31,-.24]
        seed[:,1:3]=(np.linalg.inv(np.eye(2)+pp[:,1:3]/13)-np.eye(2))*r*.8
    args=dict(center=point,seed=seed[:,0],r=r,order=order,reg=.00005,
              maxiter=35,seed_affine=seed)
    p1,s1=original(aa,bb,vv,ww,**args)
    p2,s2=cached(aa,bb,vv,ww,target_gradient=grads[direction],target_mask_float=ww.astype(float),**args)
    assert np.array_equal(p1,p2)
    assert s1==s2
    report.append(dict(index=i,radius=r,order=order,direction=direction,
                       coefficients_exact=True,all_statistics_exact=True))
full=np.load(ROOT/'work/full_width/inputs.npz')
aa,bb,vv,ww=[full[k] for k in ['A','B','va','vb']]
gradient=np.gradient(bb);mask_float=ww.astype(float)
for label,x,depth,seed in [('left_image_edge',7.,25.,[2.,0.]),
                            ('right_image_edge',2047.,25.,[2.,0.]),
                            ('near_actual_surface_mask',527.,12.,[2.,-.5])]:
    y=float(np.ceil(np.interp(x,np.arange(len(full['surface_a'])),full['surface_a'])+depth))
    args=dict(center=np.array([x,y]),seed=np.array(seed),r=13,order=2,
              reg=.00005,maxiter=35)
    p1,s1=original(aa,bb,vv,ww,**args)
    p2,s2=cached(aa,bb,vv,ww,target_gradient=gradient,target_mask_float=mask_float,**args)
    assert np.allclose(p1,p2,rtol=0,atol=0,equal_nan=True)
    assert s1.keys()==s2.keys()
    assert all(s1[k]==s2[k] or (np.isnan(s1[k]) and np.isnan(s2[k])) for k in s1)
    report.append(dict(label=label,point=[x,y],radius=13,order=2,direction='forward',
                       coefficients_exact=True,all_statistics_exact=True,
                       returned_finite=bool(np.isfinite(p1).all()),nvalid=int(s1['nvalid'])))
out=dict(only_signature_gradient_and_unchanged_float_mask_cache_differ=True,cases=report,passed=True)
(ROOT/'work/full_width/cached_solver_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
