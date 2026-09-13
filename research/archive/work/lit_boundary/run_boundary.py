import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,time
from pathlib import Path
sys.path.insert(0,'work');sys.path.insert(0,'work/lit_boundary')
import numpy as np
from scipy.ndimage import gaussian_filter,shift
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from boundary_solver import fit,image_inputs
from field_model import LocalField
OUT=Path('work/lit_boundary');m=np.load('work/metadata.npz');sa=m['surfa'].ravel()-1;sb=m['surfb'].ravel()-1
BASE='data/ExpLCL_1_03-123';rawA=np.array(Image.open(BASE+'_imgA.tif'),float);rawB=np.array(Image.open(BASE+'_imgB.tif'),float)
_f=np.load('work/final_ptv13.npz');f={k:_f[k] for k in _f.files};points=f['points'];seed=f['params'];dep=points[:,1]-np.interp(points[:,0],np.arange(len(sa)),sa);selected=dep<60
_b=np.load('work/final_reverse_field.npz');baseline_reverse={k:_b[k] for k in _b.files}
variants=[dict(name='continuation_control',strip=False,projected=False,margin=10,glare=False),dict(name='surface_strip',strip=True,projected=False,margin=10,glare=False),dict(name='surface_strip_projected',strip=True,projected=True,margin=10,glare=False),dict(name='surface_strip_glare',strip=True,projected=True,margin=4,glare=True),dict(name='surface_strip_curvature_prior',strip=True,projected=False,margin=10,glare=False,curvature_prior=True)]
(OUT/'settings.json').write_text(json.dumps(dict(variants=variants,initialization='All6coefficients from existing final_ptv13; no manual information',refit_grid_source_depth_below=60,radius_polynomial=13,tangent_radius=19,normal_radius=9,maxiter=35,reg=.00005,blend_radius=16,glare_definition='raw<250 and Gaussian(raw,sigma3)<180',source_units='zero-based px; dypositive down',manual_use='Postfit evaluation only'),indent=2))
# Known-displacement smoke test checks parameter scale/sign and strip geometry.
rng=np.random.default_rng(271);test=gaussian_filter(rng.normal(size=(101,101)),1);test/=test.std();dest=shift(test,(-.7,1.4),order=3);v=np.ones_like(test,bool);surface=np.zeros(101)
checks=[]
for st,pr in [(False,False),(True,False),(True,True)]:
 pp,dd=fit(test,dest,v,v,surface,[50.,50.],np.zeros((2,6)),strip=st,projected=pr)
 err=np.linalg.norm(pp[:,0]-[1.4,-.7]);checks.append(dict(strip=st,projected=pr,translation=pp[:,0].tolist(),error=float(err)))
 assert err<.08,(st,pr,err)
(OUT/'smoke_checks.json').write_text(json.dumps(checks,indent=2));print('smoke',checks,flush=True)
for cfg in variants:
 if os.getenv('BOUNDARY_ONLY') and cfg['name']!=os.getenv('BOUNDARY_ONLY'):continue
 name=cfg['name'];t=time.time();A,B,va,vb=image_inputs(rawA,rawB,sa,sb,cfg['margin'],cfg['glare'])
 def task(i):
  if not selected[i]:return seed[i],{k:f[k][i].item() for k in ['ncc','rms','nvalid','cond','iterations','mindet','support_fraction']}
  return fit(A,B,va,vb,sa,points[i],seed[i],strip=cfg['strip'],projected=cfg['projected'],curvature_prior=cfg.get('curvature_prior',False))
 with ThreadPoolExecutor(max_workers=4) as pool:res=list(pool.map(task,range(len(points))))
 params=np.array([v[0] for v in res]);keys=sorted(set().union(*(v[1].keys() for v in res)));diags={k:np.array([v[1].get(k,np.nan) for v in res]) for k in keys}
 np.savez_compressed(OUT/(name+'.npz'),points=points,params=params,radius=13,**diags,refit=selected)
 print(name,'forward',round(time.time()-t,1),'finite',np.isfinite(params).all(axis=(1,2)).sum(),flush=True)
 # Invert the local forward linearization to initialize a completely separate backward image fit.
 reverse_points=points+params[:,:,0]
 def reverse(i):
  if not selected[i]:return baseline_reverse['params'][i],{k:baseline_reverse[k][i].item() for k in ['ncc','rms','nvalid','cond','iterations','mindet','support_fraction']}
  pp=params[i];dest=reverse_points[i]
  if not np.isfinite(pp).all():return np.full((2,6),np.nan),{}
  initial=np.zeros((2,6));initial[:,0]=-pp[:,0]
  try:initial[:,1:3]=(np.linalg.inv(np.eye(2)+pp[:,1:3]/13)-np.eye(2))*13
  except np.linalg.LinAlgError:return np.full((2,6),np.nan),{}
  return fit(B,A,vb,va,sb,dest,initial,strip=bool(cfg['strip'] and selected[i]),projected=cfg['projected'],curvature_prior=cfg.get('curvature_prior',False))
 with ThreadPoolExecutor(max_workers=4) as pool:rev=list(pool.map(reverse,range(len(points))))
 pars=np.array([v[0] for v in rev]);keys=sorted(set().union(*(v[1].keys() for v in rev)));diag={k:np.array([v[1].get(k,np.nan) for v in rev]) for k in keys}
 np.savez_compressed(OUT/(name+'_reverse.npz'),points=reverse_points,params=pars,radius=13,**diag)
 print(name,'reverse done seconds',round(time.time()-t,1),flush=True)
 # Manual targets are read only below, after forward and backward fits are frozen.
 fin=np.load('work/final_results.npz');query=fin['query'];model=LocalField(OUT/(name+'.npz'));back=LocalField(OUT/(name+'_reverse.npz'))
 # Original727grid plus all200manual. Dense support query can be regenerated with LocalField.
 query=query[:927];u,g,ncc=model.evaluate(query,True);bu,bg,bncc=back.evaluate(query+u,True);fb=np.linalg.norm(u+bu,axis=1)
 depth=query[:,1]-np.interp(query[:,0],np.arange(501),sa);targetdepth=(query+u)[:,1]-np.interp((query+u)[:,0],np.arange(501),sb)
 e=np.linalg.norm(u[:200]-fin['manual_truth'],axis=1)
 det=np.linalg.det(np.eye(2)[None]+g);np.savez_compressed(OUT/(name+'_evaluation.npz'),query=query,disp=u,gradient=g,ncc=ncc,fb=fb,back_ncc=bncc,depth=depth,target_depth=targetdepth,manual_error=e,determinant=det)
 summary={}
 for label,sel in [('all',np.ones(200,bool)),('lt20',depth[:200]<20),('lt40',depth[:200]<40),('20to40',(depth[:200]>=20)&(depth[:200]<40)),('40to80',(depth[:200]>=40)&(depth[:200]<80)),('80plus',depth[:200]>=80)]:
  vals=e[sel];summary[label]=dict(n=int(sel.sum()),finite=int(np.isfinite(vals).sum()),mean=float(np.nanmean(vals)),median=float(np.nanmedian(vals)),rmse=float(np.sqrt(np.nanmean(vals*vals))),p90=float(np.nanpercentile(vals,90)),fb_median=float(np.nanmedian(fb[:200][sel])),fb_p90=float(np.nanpercentile(fb[:200][sel],90)),ncc_median=float(np.nanmedian(ncc[:200][sel])))
 summary['grid_diagnostics']=dict(refit_count=int(selected.sum()),nonfinite=int((~np.isfinite(params).all(axis=(1,2))).sum()),negative_jacobian=int((diags['mindet'][selected]<=0).sum()),ncc_median=float(np.nanmedian(diags['ncc'][selected])),support_median=float(np.nanmedian(diags['support_fraction'][selected])),condition_median=float(np.nanmedian(diags['cond'][selected])),forward_backward_grid_p90=float(np.nanpercentile(fb[200:][selected],90)),blended_negative_jacobian=int((det[200:][selected]<=0).sum()))
 summary['closest_two']=[dict(pick=int(i+1),depth=float(depth[i]),predicted=u[i].tolist(),manual_error=float(e[i]),fb=float(fb[i]),ncc=float(ncc[i]),determinant=float(det[i])) for i in np.argsort(depth[:200])[:2]]
 (OUT/(name+'_summary.json')).write_text(json.dumps(summary,indent=2));print(name,'summary',json.dumps(summary),flush=True)
