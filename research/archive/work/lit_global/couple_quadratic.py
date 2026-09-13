"""Exploratory image-information-weighted quadratic consensus, not wOFV.

The fixed experiments denoise the 727 already-fitted local polynomials by
requiring agreement inside overlapping liquid supports. No manual values or
locations enter the solve. Photometric information uses an approximate robust
Gauss-Newton Hessian with brightness gain/offset projected out. This is neither
a calibrated covariance nor a complete new image registration optimization.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter,map_coordinates
from extended_flow import load_images
from field_model import LocalField

OUT=Path('work/lit_global');OUT.mkdir(exist_ok=True)
BASE=Path('work/final_ptv13.npz')
base=np.load(str(BASE));data={k:base[k] for k in base.files}
points=data['points'];p0=data['params'];r=float(data['radius']);n=len(points)
A,B,va,vb,sa,sb,rawA,rawB=load_images(10,.65)
def norm(raw,valid):
    vv=valid.astype(float)
    im=gaussian_filter(raw*vv,.65)/np.maximum(gaussian_filter(vv,.65),1e-5)
    mean=gaussian_filter(im*vv,7)/np.maximum(gaussian_filter(vv,7),1e-5)
    hp=im-mean
    rms=np.sqrt(gaussian_filter(hp*hp*vv,9)/np.maximum(gaussian_filter(vv,9),1e-5)+25)
    return np.clip(hp/rms,-3,4)
A=norm(rawA,va);B=norm(rawB,vb)

def qbasis(z):
    z=np.atleast_2d(z);x,y=z[:,0]/r,z[:,1]/r
    return np.column_stack([np.ones(len(z)),x,y,.5*x*x,x*y,.5*y*y])

def patch(i,p):
    center=points[i]
    xx,yy=np.meshgrid(np.arange(int(center[0]-r),int(center[0]+r)+1),np.arange(int(center[1]-r),int(center[1]+r)+1))
    xx=xx.ravel();yy=yy.ravel();ok=(xx>=0)&(xx<501)&(yy>=0)&(yy<501)
    xx,yy=xx[ok],yy[ok];ok=va[yy,xx];xx,yy=xx[ok],yy[ok]
    xy=np.column_stack([xx,yy]);Q=qbasis(xy-center);dest=xy+Q@p.T
    weight=np.exp(-np.sum(((xy-center)/r)**2,axis=1)/(2*.65**2))
    ix=np.clip(np.floor(dest[:,0]).astype(int),0,499);iy=np.clip(np.floor(dest[:,1]).astype(int),0,499)
    fx=dest[:,0]-ix;fy=dest[:,1]-iy
    b00=B[iy,ix];b10=B[iy,ix+1];b01=B[iy+1,ix];b11=B[iy+1,ix+1]
    b=(1-fx)*(1-fy)*b00+fx*(1-fy)*b10+(1-fx)*fy*b01+fx*fy*b11
    gx=(1-fy)*(b10-b00)+fy*(b11-b01)
    gy=(1-fx)*(b01-b00)+fx*(b11-b10)
    valid=(map_coordinates(vb.astype(float),[dest[:,1],dest[:,0]],order=1,mode='constant',cval=0)>.99)
    valid&=(dest[:,0]>=1)&(dest[:,0]<499)&(dest[:,1]>=1)&(dest[:,1]<499)
    w=weight*valid;a=A[yy,xx];M=np.column_stack([b,np.ones(len(b))])
    gb=np.linalg.lstsq(M*np.sqrt(w[:,None]),a*np.sqrt(w),rcond=None)[0]
    gb[0]=np.clip(gb[0],.3,3);gb[1]=np.sum(w*(a-gb[0]*b))/max(w.sum(),1e-10)
    residual=gb[0]*b+gb[1]-a
    wt=w/np.sqrt(1+(residual/.8)**2)
    J=np.column_stack([Q*(gb[0]*gx)[:,None],Q*(gb[0]*gy)[:,None]])
    # Profile out fitted gain/offset as nuisance parameters.
    nuisance=np.linalg.lstsq(M*np.sqrt(wt[:,None]),J*np.sqrt(wt[:,None]),rcond=None)[0]
    Jp=J-M@nuisance
    H=Jp.T@(wt[:,None]*Jp)/max(w.sum(),1e-10)
    am=np.sum(w*a)/max(w.sum(),1e-10);bm=np.sum(w*b)/max(w.sum(),1e-10)
    ncc=np.sum(w*(a-am)*(b-bm))/np.sqrt(np.sum(w*(a-am)**2)*np.sum(w*(b-bm)**2)+1e-12)
    rho=.8**2*(np.sqrt(1+(residual/.8)**2)-1)
    u=Q@p.T
    dx=np.column_stack([np.zeros(len(Q)),np.ones(len(Q)),np.zeros(len(Q)),Q[:,1],Q[:,2],np.zeros(len(Q))])/r
    dy=np.column_stack([np.zeros(len(Q)),np.zeros(len(Q)),np.ones(len(Q)),np.zeros(len(Q)),Q[:,1],Q[:,2]])/r
    jac=np.stack([dx@p.T,dy@p.T],axis=2)
    determinant=np.linalg.det(np.eye(2)[None]+jac)
    return H,dict(ncc=float(ncc),loss=float(np.sum(w*rho)/max(w.sum(),1e-10)),support=float(w.sum()/weight.sum()),mindet=float(determinant.min())),xy,valid,rho,weight

t=time.time();Hs=[];base_stats=[]
for i in range(n):
    h,st,*_=patch(i,p0[i]);Hs.append(h);base_stats.append(st)
Hs=np.array(Hs);scale=float(np.median((Hs[:,0,0]+Hs[:,6,6])/2))
# The small change-to-baseline trust floor controls unsupported parameter modes;
# it does not impose zero velocity or zero affine deformation.
trust_floor=1e-3
H=sparse.block_diag([h/scale+np.eye(12)*trust_floor for h in Hs],format='csr')
rhs=H@p0.ravel()
rows=[];cols=[];vals=[];row=0;pair_count=0
for i,j in sorted(cKDTree(points).query_pairs(11.5)):
    midpoint=(points[i]+points[j])/2
    sample=midpoint+np.array([[0,0],[-4,0],[4,0],[0,-4],[0,4]],float)
    valid=(np.max(np.abs(sample-points[i]),axis=1)<=r)&(np.max(np.abs(sample-points[j]),axis=1)<=r)
    valid&=sample[:,1]>=np.interp(sample[:,0],np.arange(501),sa)+10
    sample=sample[valid]
    if not len(sample):continue
    pair_count+=1
    for qi,qj in zip(qbasis(sample-points[i]),qbasis(sample-points[j])):
        # Mean squared discrepancy per pair; no derivative shrinking.
        for axis in range(2):
            for k in range(6):
                rows.extend([row,row]);cols.extend([12*i+6*axis+k,12*j+6*axis+k]);vals.extend([qi[k]/np.sqrt(len(sample)),-qj[k]/np.sqrt(len(sample))])
            row+=1
C=sparse.csr_matrix((vals,(rows,cols)),shape=(row,12*n));CtC=C.T@C
# Verify the consistency operator has all global quadratic fields in its nullspace.
coeff=np.array([[2,.3,-.2,.001,.003,-.002],[-1,-.1,.25,.002,-.001,.001]])
global_params=[]
for x,y in points:
    q=np.array([1,x,y,.5*x*x,x*y,.5*y*y]);val=coeff@q
    dx=coeff[:,1]+coeff[:,3]*x+coeff[:,4]*y
    dy=coeff[:,2]+coeff[:,4]*x+coeff[:,5]*y
    global_params.append(np.column_stack([val,dx*r,dy*r,coeff[:,3]*r*r,coeff[:,4]*r*r,coeff[:,5]*r*r]))
null_max=float(np.max(np.abs(C@np.array(global_params).ravel())))
assert null_max<1e-9
reports=[];names=[]
for strength in [.01,.1,1.,10.]:
    solved=spsolve(H+strength*CtC,rhs).reshape(n,2,6)
    out={k:v.copy() for k,v in data.items()};out['params']=solved
    st=[patch(i,solved[i])[1] for i in range(n)]
    out['ncc']=np.array([a['ncc'] for a in st]);out['mindet']=np.array([a['mindet'] for a in st]);out['support_fraction']=np.array([a['support'] for a in st])
    tag=str(strength).replace('.','p');name='consensus_'+tag;path=OUT/(name+'.npz')
    out.update(consensus_strength=strength,trust_floor=trust_floor,information_scale=scale,prototype=np.array('image-Hessian-weighted coefficient consensus; not wOFV'))
    np.savez_compressed(str(path),**out);names.append(name)
    shift=np.linalg.norm(solved[:,:,0]-p0[:,:,0],axis=1)
    # Compare actual photometric objectives on common support, not unmatched masks.
    num0=num1=den=0.;loss_of_support=0
    for i in range(n):
        _,_,xy,v0,rho0,wt=patch(i,p0[i]);_,_,xy1,v1,rho1,_=patch(i,solved[i])
        assert np.array_equal(xy,xy1)
        common=v0&v1;num0+=np.sum(wt[common]*rho0[common]);num1+=np.sum(wt[common]*rho1[common]);den+=np.sum(wt[common]);loss_of_support+=int(np.sum(v0&~v1))
    report=dict(name=name,strength=strength,center_change_median=float(np.median(shift)),center_change_p95=float(np.percentile(shift,95)),center_change_max=float(shift.max()),overlap_rms_before=float(np.sqrt(np.mean((C@p0.ravel())**2))),overlap_rms_after=float(np.sqrt(np.mean((C@solved.ravel())**2))),common_support_photo_before=float(num0/den),common_support_photo_after=float(num1/den),lost_valid_samples=loss_of_support,nonpositive_local_determinants=int(np.sum(out['mindet']<=0)))
    reports.append(report);print(json.dumps(report),flush=True)

# Manual references first accessed only after all fixed fitting experiments.
meta=np.load('work/metadata.npz');q=meta['p'][:,0].T-1;truth=meta['p'][:,1].T-meta['p'][:,0].T;depth=q[:,1]-np.interp(q[:,0],np.arange(501),sa)
for name,path in [('baseline',BASE)]+[(s,OUT/(s+'.npz')) for s in names]:
    model=LocalField(str(path));d,g=model.evaluate(q);err=np.linalg.norm(d-truth,axis=1)
    metrics={}
    for key,sel in [('all',np.ones(len(q),bool)),('near40',depth<40),('near25',depth<25),('depth40plus',depth>=40)]:
        e=err[sel];metrics[key]=dict(n=int(sel.sum()),mean=float(e.mean()),median=float(np.median(e)),rmse=float(np.sqrt(np.mean(e*e))))
    eps=1e-3;fd=(model.evaluate(q+[eps,0])[0]-model.evaluate(q-[eps,0])[0])/(2*eps)
    np.savez_compressed(str(OUT/(name+'_validation.npz')),points=q,predicted=d,gradient=g,manual=truth,error=err,depth=depth)
    rec=dict(name=name,metrics=metrics,gradient_fd_max=float(np.max(np.abs(fd-g[:,:,0]))))
    if name=='baseline':reports.insert(0,rec)
    else:next(a for a in reports if a['name']==name).update(rec)
    print('VALIDATION',json.dumps(rec),flush=True)
result=dict(method='fixed photometric-information-weighted consensus of local quadratic fits',paper_reproduction=False,new_particle_tracks=False,new_near_surface_information=False,coupling_rows=row,coupling_pairs=pair_count,global_quadratic_nullspace_max=null_max,trust_floor=trust_floor,elapsed_seconds=time.time()-t,reports=reports)
(OUT/'consensus_results.json').write_text(json.dumps(result,indent=2))
