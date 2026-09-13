import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,time,json
from scipy.ndimage import map_coordinates, gaussian_filter, maximum_filter
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
from extended_flow import load_images

t0=time.time()
_,_,va,vb,sa,sb,rawA,rawB=load_images(10,.5)
A=gaussian_filter(rawA,.5);B=gaussian_filter(rawB,.5)
va=va.astype(float);vb=vb.astype(float)
f=np.load('work/propagate0.npz'); good=(f['ncc']>.7)&np.isfinite(f['params']).all(axis=(1,2))
ip=f['points'][good];pp=f['params'][good];tree=cKDTree(ip)
J=np.eye(2)[None]+pp[:,:,1:]/19
ji=np.linalg.inv(J); bp=ip+pp[:,:,0];btree=cKDTree(bp)

def prior(pt,back=False):
    tt=btree if back else tree
    dist,idx=tt.query(pt,k=12)
    if back: pred=-pp[idx,:,0]+np.einsum('nij,nj->ni',ji[idx]-np.eye(2)[None],pt-bp[idx])
    else: pred=pp[idx,:,0]+np.einsum('nij,nj->ni',pp[idx,:,1:]/19,pt-ip[idx])
    dd=np.linalg.norm(pred[:,None]-pred[None,:],axis=2);weights=np.exp(-dist**2/(2*25**2));med=np.argmin(dd.dot(weights))
    sel=np.linalg.norm(pred-pred[med],axis=1)<4
    return np.average(pred[sel],axis=0,weights=weights[sel])

DY,DX=np.mgrid[-4:5,-4:5]; off=np.c_[DX.ravel(),DY.ravel()]
DY,DX=np.mgrid[-8:9,-8:9]; shifts=np.c_[DX.ravel(),DY.ravel()]
def track(pt,back=False):
    src,dst,vs,vd=(B,A,vb,va) if back else (A,B,va,vb)
    pr=prior(pt,back);co=pt+off;aa=map_coordinates(src,[co[:,1],co[:,0]],order=1)
    sm=map_coordinates(vs,[co[:,1],co[:,0]],order=1)>.99
    def cc(ds):
        coords=co[None,:,:]+ds[:,None,:]
        bb=map_coordinates(dst,[coords[:,:,1],coords[:,:,0]],order=1)
        vm=map_coordinates(vd,[coords[:,:,1],coords[:,:,0]],order=1)>.99
        vm&=sm[None,:];n=vm.sum(axis=1);den=np.maximum(n,1)
        am=(aa[None,:]*vm).sum(axis=1)/den;bm=(bb*vm).sum(axis=1)/den
        az=(aa[None,:]-am[:,None])*vm;bz=(bb-bm[:,None])*vm
        c=(az*bz).sum(axis=1)/np.sqrt((az*az).sum(axis=1)*(bz*bz).sum(axis=1)+1e-10)
        c[n<.85*len(aa)]=-1
        return c
    candidates=pr[None]+shifts;cors=cc(candidates);scores=cors-.005*np.sum(shifts**2,axis=1)
    ii=np.argsort(-scores);sel=[ii[0]]
    for idx in ii[1:]:
        if np.linalg.norm(candidates[idx]-candidates[sel[0]])>2.5:sel.append(idx);break
    fits=[]
    for idx in sel:
        opt=minimize(lambda d:1-cc(d[None,:])[0]+.005*np.sum((d-pr)**2),candidates[idx],method='Nelder-Mead',options={'maxiter':55,'xatol':.035,'fatol':.0002})
        fits.append((float(opt.fun),opt.x,float(cc(opt.x[None])[0])))
    fits.sort(key=lambda z:z[0]);best=fits[0]
    # A genuine distinct runner-up is required; if optimizations merge, use the
    # best coarse candidate outside the winning particle's 2.5px neighborhood.
    outside=np.linalg.norm(candidates-best[1][None],axis=1)>2.5
    alternatives=[float(np.min(1-scores[outside]))]
    alternatives += [fit[0] for fit in fits[1:] if np.linalg.norm(fit[1]-best[1])>2.5]
    gap=max(0.,min(alternatives)-best[0])
    return best[1],best[2],gap,pr

# Automatic particle candidates; manual source locations are never used here.
hp=gaussian_filter(rawA,.6)-gaussian_filter(rawA,2)
background=gaussian_filter(rawA,5)
yy,xx=np.indices(A.shape)
det=(hp==maximum_filter(hp,size=5))&(hp>8)&(xx>=60)&(xx<=390)&(yy>=sa[None]+14)&(yy<=175)&(background<180)
pts=np.c_[xx[det],yy[det]].astype(float)
strength=hp[det]
if len(pts)>1400:
    # Preserve broad spatial coverage with per-tile selection rather than global brightness.
    chosen=[]
    tile=(pts[:,0]//15).astype(int)+40*(pts[:,1]//15).astype(int)
    for k in np.unique(tile):
        ids=np.flatnonzero(tile==k);chosen.extend(ids[np.argsort(-strength[ids])[:7]])
    chosen=np.array(chosen);pts=pts[chosen];strength=strength[chosen]
print('automatic feature count',len(pts),flush=True)
with ThreadPoolExecutor(max_workers=4) as ex: fw=list(ex.map(track,pts))
disp=np.array([q[0] for q in fw]);ncc=np.array([q[1] for q in fw]);gap=np.array([q[2] for q in fw]);pri=np.array([q[3] for q in fw])
print('forward complete',time.time()-t0,flush=True)
dest=pts+disp
with ThreadPoolExecutor(max_workers=4) as ex: bw=list(ex.map(lambda pt:track(pt,True),dest))
bd=np.array([q[0] for q in bw]);bncc=np.array([q[1] for q in bw]);bgap=np.array([q[2] for q in bw]);fb=np.linalg.norm(disp+bd,axis=1)
accepted=(ncc>.68)&(bncc>.68)&(fb<1.0)&(gap>.015)&(bgap>.01)
np.savez('work/ptv_tracks.npz',points=pts,disp=disp,prior=pri,ncc=ncc,ambiguity_gap=gap,back_disp=bd,back_ncc=bncc,back_gap=bgap,fb=fb,accepted=accepted,strength=strength)
print('back complete',time.time()-t0,'accepted',accepted.sum(),flush=True)

def field(query,mask,scale=14,order=2):
    xp=pts[mask];up=disp[mask];tr=cKDTree(xp)
    qual=np.clip((ncc[mask]-.5)/.4,.1,1)*np.clip(gap[mask]/.08,.1,1)*np.exp(-fb[mask]**2/.5)
    pred=[];grad=[];rms=[];n=[];reach=[]
    for pt in query:
        dd,ii=tr.query(pt,k=min(45,len(xp)))
        near=dd<max(scale*2,dd[min(11,len(dd)-1)])
        dd=dd[near];ii=ii[near];q=(xp[ii]-pt)/scale
        X=np.c_[np.ones(len(q)),q[:,0],q[:,1]]
        if order==2:X=np.c_[X,.5*q[:,0]**2,q[:,0]*q[:,1],.5*q[:,1]**2]
        base=np.exp(-.5*(dd/scale)**2)*qual[ii];w=base.copy();Y=up[ii]
        ridge=np.diag([1e-8,1e-5,1e-5]+([.04,.04,.04] if order==2 else []))
        for _ in range(7):
            coef=np.linalg.solve(X.T@(w[:,None]*X)+ridge,X.T@(w[:,None]*Y))
            residual=np.linalg.norm(Y-X@coef,axis=1)
            sc=max(.45,1.4826*np.median(residual))
            w=base/(1+(residual/(2*sc))**4)
        pred.append(coef[0]);grad.append(coef[1:3].T/scale);rms.append(np.sqrt(np.sum(w*residual**2)/sum(w)));n.append(np.sum(w>.1));reach.append(dd.min())
    return np.array(pred),np.array(grad),np.array(rms),np.array(n),np.array(reach)

grid=f['points']
# Only now are manual locations/targets read for an explicitly post hoc evaluation.
m=np.load('work/metadata.npz');manual=m['p'][:,0].T-1;truth=m['p'][:,1].T-m['p'][:,0].T
query=np.r_[grid,manual];depth=manual[:,1]-np.interp(manual[:,0],np.arange(501),sa)
metrics={}
for scale in [10,14,18]:
    pred,grad,rms,count,reach=field(query,accepted,scale)
    name='ptv_quad'+str(scale)
    np.savez('work/'+name+'.npz',points=query,disp=pred,gradient=grad,rms=rms,count=count,nearest_feature=reach,grid_count=len(grid),scale=scale)
    err=np.linalg.norm(pred[len(grid):]-truth,axis=1)
    met={}
    for label,mm in [('all',np.ones(200,bool)),('near40',depth<40),('deep80',depth>=80)]:
        met[label]={'n':int(mm.sum()),'median':float(np.median(err[mm])),'mean':float(np.mean(err[mm])),'p90':float(np.percentile(err[mm],90))}
    metrics[name]=met;print(name,json.dumps(met),flush=True)
    for i in [83,84,93,94,95,141,142]: print('pick',i+1,'pred',pred[len(grid)+i],'truth',truth[i],'err',err[i],flush=True)
with open('work/ptv_metrics.json','w') as out:json.dump(metrics,out,indent=2)
print('total seconds',time.time()-t0,flush=True)
