"""Bounded synthetic recovery check for extended_flow.fit_local (read-only import).

Synthetic A is a continuous subpixel Gaussian-particle texture. B is sampled by
inverting a known affine/quadratic A-to-B coordinate map with Newton iterations,
then evaluating the same analytic texture at the inverse coordinates. Thus B is
not made by applying the fitter's own bilinear sampler to A. Particle-image blobs
are deformed with the coordinate map; this checks mathematical registration
recovery, not robustness to actual free-surface optical artifacts.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import hashlib,json,time
from pathlib import Path
import numpy as np
from extended_flow import fit_local

ROOT=Path(__file__).resolve().parent
N=151
CENTER=np.array([75.,75.])
TRANS=np.array([12.4,-3.7])
G=np.array([[.12,-.10],[.07,-.065]])
CURV=np.array([[.004,-.002,.0025],[-.002,.0015,-.003]])
SEED=TRANS+np.array([1.25,-.85])
REG=.00005


def displacement(x,h):
    z=x-CENTER; a=z[...,0];b=z[...,1]
    return TRANS+z@G.T+np.stack([.5*a*a,a*b,.5*b*b],axis=-1)@h.T


def derivative(x,h):
    z=x-CENTER;a=z[...,0];b=z[...,1]
    out=np.broadcast_to(G,x.shape[:-1]+(2,2)).copy()
    out[...,0,0]+=h[0,0]*a+h[0,1]*b
    out[...,0,1]+=h[0,1]*a+h[0,2]*b
    out[...,1,0]+=h[1,0]*a+h[1,1]*b
    out[...,1,1]+=h[1,1]*a+h[1,2]*b
    return out


def inverse_map(dst,h):
    x=CENTER+(dst-CENTER-TRANS)@np.linalg.inv(np.eye(2)+G).T
    for _ in range(16):
        residual=x+displacement(x,h)-dst
        jac=derivative(x,h)+np.eye(2)
        det=jac[...,0,0]*jac[...,1,1]-jac[...,0,1]*jac[...,1,0]
        delta=np.stack([(jac[...,1,1]*residual[...,0]-jac[...,0,1]*residual[...,1])/det,(-jac[...,1,0]*residual[...,0]+jac[...,0,0]*residual[...,1])/det],axis=-1)
        x-=delta
        if np.max(np.abs(delta))<1e-10:break
    err=np.max(np.linalg.norm(x+displacement(x,h)-dst,axis=-1))
    return x,float(err)


def render_texture(x,particles,widths,amplitude):
    result=np.empty(x.shape[:-1]); flat=x.reshape(-1,2);v=result.ravel()
    for i in range(0,len(flat),512):
        delta=flat[i:i+512,None,:]-particles[None,:,:]
        dist2=np.sum(delta*delta,axis=-1)
        v[i:i+512]=np.exp(-.5*dist2/(widths*widths))@amplitude
    return result


def main():
    start=time.time();yy,xx=np.indices((N,N));xy=np.stack([xx,yy],axis=-1).astype(float)
    masks=np.ones((N,N),bool);masks[:3]=False;masks[-3:]=False;masks[:,:3]=False;masks[:,-3:]=False
    inverses={};inverse_errors={}
    for name,h in [('affine',np.zeros_like(CURV)),('quadratic',CURV)]:
        inverses[name],inverse_errors[name]=inverse_map(xy,h)
    rows=[]
    for realization in range(8):
        rng=np.random.default_rng(59103+realization)
        # Include off-image particles so continuous sampling has no artificial frame edge.
        particles=rng.uniform(-25,N+25,size=(int(.04*(N+50)**2),2))
        widths=rng.uniform(.9,1.5,len(particles));amp=rng.uniform(.5,1.5,len(particles))
        rawA=render_texture(xy,particles,widths,amp);mean=float(rawA.mean());sd=float(rawA.std())
        A=(rawA-mean)/sd+rng.normal(0,.01,rawA.shape)
        for case,h in [('affine',np.zeros_like(CURV)),('quadratic',CURV)]:
            rawB=render_texture(inverses[case],particles,widths,amp)
            B=(.88*rawB+.12-mean)/sd+rng.normal(0,.01,rawB.shape)
            for radius in [19,25]:
                affine=None
                for order in [1,2]:
                    p,stats=fit_local(A,B,masks,masks,CENTER,SEED,r=radius,order=order,reg=REG,maxiter=60,seed_affine=affine)
                    if order==1:affine=p.copy()
                    estimated_G=p[:,1:3]/radius
                    q=np.linspace(-radius,radius,13);QX,QY=np.meshgrid(q,q)
                    z=np.stack([QX,QY],axis=-1)
                    basis=np.stack([np.ones_like(QX),QX/radius,QY/radius]+([.5*(QX/radius)**2,QX*QY/radius**2,.5*(QY/radius)**2] if order==2 else []),axis=-1)
                    fitted=basis@p.T;truth=displacement(CENTER+z,h)
                    row=dict(case=case,realization=realization,radius=radius,order=order,reg=REG,translation=p[:,0].tolist(),gradient=estimated_G.tolist(),translation_error_px=float(np.linalg.norm(p[:,0]-TRANS)),gradient_error_frobenius=float(np.linalg.norm(estimated_G-G)),dudx_error=float(estimated_G[0,0]-G[0,0]),dvdx_error=float(estimated_G[1,0]-G[1,0]),window_endpoint_rmse_px=float(np.sqrt(np.mean(np.sum((fitted-truth)**2,axis=-1)))),**stats)
                    if order==2:
                        hhat=p[:,3:6]/radius**2
                        row.update(curvature=hhat.tolist(),curvature_error_frobenius=float(np.linalg.norm(hhat-h)))
                    rows.append(row)
        print('synthetic realization',realization+1,'of 8; elapsed',round(time.time()-start,1),flush=True)
    summaries=[]
    for case in ['affine','quadratic']:
        for radius in [19,25]:
            for order in [1,2]:
                subset=[r for r in rows if r['case']==case and r['radius']==radius and r['order']==order]
                keys=['translation_error_px','gradient_error_frobenius','dudx_error','dvdx_error','window_endpoint_rmse_px','ncc','iterations','mindet']
                summary=dict(case=case,radius=radius,order=order,n=len(subset))
                for key in keys:
                    vals=np.array([r[key] for r in subset]);summary[key+'_mean']=float(vals.mean());summary[key+'_max']=float(vals.max())
                summaries.append(summary)
    data=dict(description=__doc__,seed_translation=SEED.tolist(),true_translation=TRANS.tolist(),true_gradient=G.tolist(),true_quadratic_coefficients_pixel_units=CURV.tolist(),initial_affine_slopes='zero; order2 initialized by image-fitted affine solution',particle_density_per_pixel=.04,gaussian_sigma_range_px=[.9,1.5],noise_sd_relative_to_image_std=.01,frame_B_gain=.88,frame_B_offset=.12,inverse_map_max_error_px=inverse_errors,replicates=8,extended_flow_sha256=hashlib.sha256((ROOT/'extended_flow.py').read_bytes()).hexdigest(),elapsed_seconds=time.time()-start,summaries=summaries,individual_results=rows)
    (ROOT/'synthetic_validation.json').write_text(json.dumps(data,indent=2))
    lines=['# Synthetic mathematical recovery check','', 'Eight independent analytic Gaussian-particle textures per case; forward translation (12.4, −3.7) px, exact affine gradient [[0.12, −0.10], [0.07, −0.065]], mild known curvature for the quadratic case. Seed translation is 1.51 px from truth; initial affine slopes are zero. Quadratic fits start from the image-derived affine result. reg=0.00005. B is sampled by numerically inverting the known coordinate map and evaluating the continuous analytic particle texture, avoiding reuse of the fitter’s image interpolation. B also has 0.88× gain and an offset; both images have 1% image-standard-deviation Gaussian noise.','', '| True map | Radius | Fitted order | Center error, mean px | Gradient matrix error, mean | ∂dₓ/∂x bias | Whole-window EPE RMSE, mean px | NCC mean |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['case']} | {s['radius']} | {s['order']} | {s['translation_error_px_mean']:.4f} | {s['gradient_error_frobenius_mean']:.5f} | {s['dudx_error_mean']:+.5f} | {s['window_endpoint_rmse_px_mean']:.4f} | {s['ncc_mean']:.5f} |")
    lines+=['','Gradient errors are dimensionless displacement derivatives per image pair. Whole-window error is evaluated on a fixed 13×13 grid over the window, independently of the fitting residual. These simulations check the sign, coefficient scaling, numerical recovery, and whether second-order terms help when the brightness-deformation model is correct. They do not include free-surface reflections, missing/appearing particles, unchanged physical particle point-spread functions, out-of-plane motion, or moving liquid masks. They cannot establish uncertainty on the experimental near-surface results.','',f'Worst inverse-map numerical error: {max(inverse_errors.values()):.3g} px. Fit implementation SHA256: {data["extended_flow_sha256"]}.']
    (ROOT/'synthetic_validation.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
