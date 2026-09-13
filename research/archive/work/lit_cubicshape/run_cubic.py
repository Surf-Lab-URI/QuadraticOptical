"""Fixed cubic-shape tests. Manual references are evaluated after all runs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from concurrent.futures import ThreadPoolExecutor
from extended_flow import load_images
from field_model import LocalField
from cubic_shape import basis,fit_local,CubicField
OUT=Path('work/lit_cubicshape');OUT.mkdir(exist_ok=True)
base=np.load('work/final_ptv13.npz');points=base['points'];initial=base['params'];r=float(base['radius'])
A,B,va,vb,sa,sb,rawA,rawB=load_images(10,.65)
def normalize(raw,valid):
    v=valid.astype(float);sm=gaussian_filter(raw*v,.65)/np.maximum(gaussian_filter(v,.65),1e-5)
    mean=gaussian_filter(sm*v,7)/np.maximum(gaussian_filter(v,7),1e-5);hp=sm-mean
    rms=np.sqrt(gaussian_filter(hp*hp*v,9)/np.maximum(gaussian_filter(v,9),1e-5)+25)
    return np.clip(hp/rms,-3,4)
A=normalize(rawA,va);B=normalize(rawB,vb)
experiments=[('quadratic_continuation',2,16),('cubic16',3,16),('cubic64',3,64)]
reports=[];started=time.time()
for name,order,mult in experiments:
    t=time.time()
    def task(i):return fit_local(A,B,va,vb,points[i],initial[i],r=r,order=order,reg=5e-5,cubic_multiplier=mult,maxiter=35)
    with ThreadPoolExecutor(max_workers=4) as ex:result=list(ex.map(task,range(len(points))))
    params=np.array([a[0] for a in result]);stats={k:np.array([a[1][k] for a in result]) for k in result[0][1]}
    np.savez_compressed(str(OUT/(name+'.npz')),points=points,params=params,radius=r,order=order,reg=5e-5,cubic_multiplier=mult,margin=10,sigma=.65,maskednorm=True,**stats)
    report=dict(name=name,order=order,cubic_multiplier=mult,elapsed=time.time()-t,median_ncc=float(np.median(stats['ncc'])),median_rms=float(np.median(stats['rms'])),nonpositive_local_jacobian=int(np.sum(stats['mindet']<=0)),mean_iterations=float(np.mean(stats['iterations'])))
    reports.append(report);print('FIT',json.dumps(report),flush=True)

# Known cubic warp verification: intensity deformation (not a claim about actual
# fluorescent-particle blob mechanics), local Newton inversion, and fixed seeds.
true=np.array([[1.2,.4,-.2,.25,.15,-.2,1.5,-.75,.5,-1.2],[-.6,.3,.2,.1,-.2,.2,-.6,.9,-1.,1.2]])
center=np.array([40.,40.]);yy,xx=np.indices((81,81));q=np.column_stack([xx.ravel(),yy.ravel()]).astype(float)
inv=q.copy()
for _ in range(20):
    Q,Qx,Qy=basis((inv-center)/13,3);flow=Q@true.T;J=np.stack([Qx@true.T,Qy@true.T],axis=2)/13+np.eye(2)[None]
    err=inv+flow-q;step=np.linalg.solve(J,err[...,None])[...,0];step=np.clip(step,-2,2);inv-=step
Q,_,_=basis((inv-center)/13,3);inverse_error=np.linalg.norm(inv+Q@true.T-q,axis=1);check=np.max(np.abs(q-center),axis=1)<20
synthetic=[]
for seed in range(8):
    rng=np.random.RandomState(seed+811);imp=np.zeros((81,81));imp[rng.randint(1,80,250),rng.randint(1,80,250)]=rng.uniform(.5,2,250)
    ai=gaussian_filter(imp,1.15);ai/=np.std(ai)
    bi=map_coordinates(ai,[inv[:,1],inv[:,0]],order=3,mode='constant',cval=0).reshape(81,81)*1.2+.05
    bi+=rng.normal(0,.005,bi.shape);mask=np.ones(ai.shape,bool)
    init=true[:,:6].copy()
    for name,order,reg,mult in [('quadratic',2,5e-5,16),('cubic_weak_sanity',3,0,16),('cubic16',3,5e-5,16),('cubic64',3,5e-5,64)]:
        pp,st=fit_local(ai,bi,mask,mask,center,init,r=13,order=order,reg=reg,cubic_multiplier=mult,maxiter=100)
        sample=np.array([(x,y) for x in range(-10,11,2) for y in range(-10,11,2)],float);Qt,_,_=basis(sample/13,3);Qp,_,_=basis(sample/13,order)
        endpoint=np.linalg.norm(Qp@pp.T-Qt@true.T,axis=1)
        synthetic.append(dict(seed=seed,name=name,center_error=float(np.linalg.norm(pp[:,0]-true[:,0])),mean_patch_error=float(endpoint.mean()),gradient_error=float(np.linalg.norm((pp[:,1:3]-true[:,1:3])/13)),rms=st['rms']))
synsummary={name:{key:float(np.mean([s[key] for s in synthetic if s['name']==name])) for key in ['center_error','mean_patch_error','gradient_error','rms']} for name in ['quadratic','cubic_weak_sanity','cubic16','cubic64']}
print('SYNTHETIC',json.dumps(synsummary),flush=True)

meta=np.load('work/metadata.npz');manual=meta['p'][:,0].T-1;truth=meta['p'][:,1].T-meta['p'][:,0].T;depth=manual[:,1]-np.interp(manual[:,0],np.arange(501),sa)
baseline=LocalField('work/final_ptv13.npz');db,gb=baseline.evaluate(manual)
metrics=[]
for name,path in [('baseline',Path('work/final_ptv13.npz'))]+[(a[0],OUT/(a[0]+'.npz')) for a in experiments]:
    model=CubicField(path);d,g=model.evaluate(manual);e=np.linalg.norm(d-truth,axis=1);met={}
    for key,sel in [('all',np.ones(len(manual),bool)),('near40',depth<40),('near25',depth<25),('depth40plus',depth>=40)]:
        ee=e[sel];met[key]=dict(n=int(sel.sum()),mean=float(ee.mean()),median=float(np.median(ee)),rmse=float(np.sqrt(np.mean(ee*ee))))
    eps=1e-3;fd=(model.evaluate(manual+[eps,0])[0]-model.evaluate(manual-[eps,0])[0])/(2*eps)
    fm=(model.evaluate(manual+[0,eps])[0]-model.evaluate(manual-[0,eps])[0])/(2*eps)
    # A baseline agreement test independently checks order2 compatibility.
    record=dict(name=name,metrics=met,gradient_fd_max=float(max(np.max(np.abs(fd-g[:,:,0])),np.max(np.abs(fm-g[:,:,1])))),change_from_baseline_median=float(np.median(np.linalg.norm(d-db,axis=1))),gradient_change_from_baseline_median=float(np.median(np.abs(g[:,0,0]-gb[:,0,0]))))
    if name=='baseline':record['baseline_evaluator_agreement']=float(max(np.max(np.abs(d-db)),np.max(np.abs(g-gb))))
    metrics.append(record);np.savez_compressed(str(OUT/(name+'_validation.npz')),points=manual,disp=d,gradient=g,error=e,depth=depth,manual_truth=truth)
    print('VALIDATION',json.dumps(record),flush=True)
out=dict(fits=reports,validation=metrics,synthetic=synsummary,synthetic_raw=synthetic,synthetic_inverse_error=float(inverse_error[check].max()),elapsed_seconds=time.time()-started,new_particle_tracks=False,new_surface_support=False)
(OUT/'cubic_shape_results.json').write_text(json.dumps(out,indent=2))
