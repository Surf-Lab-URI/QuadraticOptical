"""Analytic particle textures with independent forward-map inverse rendering.
Tests clean affine, clean quadratic, and moving-interface/variable-glare cases.
All parameters and random seeds are fixed before inspecting synthetic results.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from experiments import solve,Sampler,package_jet
from extended_flow import fit_local
from synthetic_validate import render_texture,displacement,inverse_map,CENTER,G,CURV,TRANS,N

def norm(raw,v):
 vv=v.astype(float);im=gaussian_filter(raw*vv,.65)/np.maximum(gaussian_filter(vv,.65),1e-5)
 mean=gaussian_filter(im*vv,7)/np.maximum(gaussian_filter(vv,7),1e-5);hp=im-mean
 rms=np.sqrt(gaussian_filter(hp*hp*vv,9)/np.maximum(gaussian_filter(vv,9),1e-5)+25)
 return np.clip(hp/rms,-3,4)

def main():
 t=time.time();yy,xx=np.indices((N,N));xy=np.stack([xx,yy],axis=-1).astype(float);r=13;rows=[]
 inv={case:inverse_map(xy,hh)[0] for case,hh in [('affine',np.zeros_like(CURV)),('quadratic',CURV)]}
 for rep in range(4):
  rng=np.random.default_rng(11407+rep);particles=rng.uniform(-25,N+25,size=(int(.04*(N+50)**2),2));widths=rng.uniform(.9,1.5,len(particles));amp=rng.uniform(.5,1.5,len(particles))
  rawA=render_texture(xy,particles,widths,amp)
  for case in ['affine','quadratic','quadratic_boundary_glare','quadratic_particle_brightness']:
   isboundary=case.endswith('glare');hh=np.zeros_like(CURV) if case=='affine' else CURV;rawB=render_texture(inv['affine' if case=='affine' else 'quadratic'],particles,widths,amp)
   if case=='quadratic_particle_brightness':
    changed_amp=amp*np.exp(rng.normal(0,.22,len(amp)))
    rawB=render_texture(inv['quadratic'],particles,widths,changed_amp)
   a=rawA.copy()*45+40;b=rawB.copy()*39.6+45.4
   if isboundary:
    surf=CENTER[1]-18+3*np.sin((np.arange(N)-CENTER[0])/22)
    edge=np.c_[np.arange(N),surf];b_edge=edge+displacement(edge,hh);surfB=np.interp(np.arange(N),b_edge[:,0],b_edge[:,1])
    da=yy-surf[None];db=yy-surfB[None];va=da>=10;vb=db>=10
    a+=150*np.exp(-.5*(da/6)**2)*(1+.4*np.cos(xx/18));b+=210*np.exp(-.5*(db/6)**2)*(1+.55*np.sin(xx/19))
    # Independent, smooth illumination modulation beneath the boundary.
    a*=1+.15*(xx-CENTER[0])/N;b*=1-.18*(xx-CENTER[0])/N
    a[da<0]=20;b[db<0]=20
   else:
    va=np.ones((N,N),bool);vb=va.copy()
   a+=rng.normal(0,.5,a.shape);b+=rng.normal(0,.5,b.shape)
   A=norm(a,va);B=norm(b,vb)
   # Common conventional fit from perturbed translation and zero slopes.
   pa,st=fit_local(A,B,va,vb,CENTER,TRANS+[1.25,-.85],r=r,order=1,reg=.00005,maxiter=60)
   p0,st=fit_local(A,B,va,vb,CENTER,pa[:,0],r=r,order=2,reg=.00005,maxiter=60,seed_affine=pa)
   methods=[('baseline',p0,st)]
   for name,sym,cub,project in [('forward_cubic',False,True,False),('forward_cubic_projected',False,True,True),('symmetric_bilinear',True,False,False),('symmetric_cubic',True,True,False)]:
    p,ss=solve(A,B,va,vb,CENTER,p0,r=r,symmetric=sym,cubic=cub,reg=.00005,maxiter=40,project_photometric=project)
    methods.append((name,p,ss))
   for name,p,ss in methods:
    hhat=np.stack([p[:,3],p[:,4],p[:,5]],axis=1)/r**2
    rows.append(dict(case=case,replicate=rep,method=name,center_error_px=float(np.linalg.norm(p[:,0]-TRANS)),gradient_error=float(np.linalg.norm(p[:,1:3]/r-G)),dudx_error=float(p[0,1]/r-G[0,0]),curvature_error=float(np.linalg.norm(hhat-hh)),ncc=float(ss['ncc']),failed=bool(ss.get('failed',False))))
  print('synthetic',rep+1,'/4 elapsed',round(time.time()-t,1),flush=True)
 summaries=[]
 for case in ['affine','quadratic','quadratic_boundary_glare','quadratic_particle_brightness']:
  for method in ['baseline','forward_cubic','forward_cubic_projected','symmetric_bilinear','symmetric_cubic']:
   rr=[a for a in rows if a['case']==case and a['method']==method];summary=dict(case=case,method=method,n=len(rr))
   for k in ['center_error_px','gradient_error','dudx_error','curvature_error','ncc']:
    v=np.array([a[k] for a in rr]);summary[k+'_mean']=float(v.mean());summary[k+'_max']=float(v.max())
   summary['failed']=sum(a['failed'] for a in rr);summaries.append(summary)
 result=dict(description=__doc__,replicates=4,particle_density=.04,blob_sigma_px=[.9,1.5],r=r,reg=.00005,true_translation=TRANS.tolist(),true_G=G.tolist(),true_quadratic_H=CURV.tolist(),boundary_center_depth_px=18,glare_sigma_px=6,summaries=summaries,individual=rows,seconds=time.time()-t)
 (HERE/'synthetic_extended_metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
