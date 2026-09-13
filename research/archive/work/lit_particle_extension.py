"""Boundary-visible sparse matching experiment; manual vectors only for reporting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates, label
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from extended_flow import load_images
from field_model import LocalField
from ptv_model import PTVModel

OUT=Path('work/lit_particles'); OUT.mkdir(exist_ok=True)
_,_,_,_,SA,SB,RA,RB=load_images(2,.5)
YY,XX=np.indices(RA.shape)
def prepare(raw,surface):
    labels,_=label(raw>=250)
    areas=np.bincount(labels.ravel())[labels];areas[raw<250]=0
    liquid=YY>=surface[None,:]+2
    visible=liquid&(raw<250)
    smooth=gaussian_filter(raw*visible,.5)/np.maximum(gaussian_filter(visible.astype(float),.5),1e-6)
    hp=gaussian_filter(raw,.6)-gaussian_filter(raw,2)
    bg=gaussian_filter(raw,5)
    det=(hp==maximum_filter(hp,5))&(hp>8)&(XX>=55)&(XX<=445)&liquid&(YY<180)&(bg<210)&(areas<25)
    pts=np.c_[XX[det],YY[det]].astype(float)
    return smooth,visible.astype(float),pts
A,VA,PA=prepare(RA,SA);B,VB,PB=prepare(RB,SB)
F=LocalField('work/final_ptv13.npz');R=LocalField('work/final_reverse_field.npz')
PTV=PTVModel()
DEP=PA[:,1]-np.interp(PA[:,0],np.arange(501),SA)
sel=(PA[:,0]>=60)&(PA[:,0]<=390)&(DEP<40)
PA,DEP=PA[sel],DEP[sel]
M=np.load('work/metadata.npz');MAN=M['p'][:,0].T-1
QUERY=np.r_[PA,MAN]

def fit(point,initial,grad,radius=4,affine=True,back=False):
    src,dst,vs,vd=(B,A,VB,VA) if back else (A,B,VA,VB)
    oy,ox=np.mgrid[-radius:radius+1,-radius:radius+1]
    off=np.c_[ox.ravel(),oy.ravel()]
    co=point+off
    j=np.eye(2)+grad if affine else np.eye(2)
    badj=not np.isfinite(j).all() or np.linalg.det(j)<.2 or np.linalg.norm(j,2)>3
    if badj:j=np.eye(2)
    warped_off=off.dot(j.T)
    aa=map_coordinates(src,[co[:,1],co[:,0]],order=1)
    sm=map_coordinates(vs,[co[:,1],co[:,0]],order=1,mode='constant')>.99
    weights=np.exp(-np.sum(off**2,axis=1)/(2*(radius*.85)**2))*sm
    def correlations(ds):
        pos=point[None,None,:]+ds[:,None,:]+warped_off[None,:,:]
        bb=map_coordinates(dst,[pos[:,:,1],pos[:,:,0]],order=1,mode='nearest')
        vm=map_coordinates(vd,[pos[:,:,1],pos[:,:,0]],order=1,mode='constant')>.99
        ww=vm*weights[None,:]; sw=np.maximum(ww.sum(axis=1),1e-8)
        az=aa[None,:]-(ww*aa[None,:]).sum(axis=1)[:,None]/sw[:,None]
        bz=bb-(ww*bb).sum(axis=1)[:,None]/sw[:,None]
        av=(ww*az**2).sum(axis=1);bv=(ww*bz**2).sum(axis=1)
        cc=(ww*az*bz).sum(axis=1)/np.sqrt(av*bv+1e-12)
        n=(vm&sm[None,:]).sum(axis=1); share=sw/max(weights.sum(),1e-8)
        cc[(n<25)|(share<.65)|(av/sw<20)|(bv/sw<20)]=-1
        return cc,n,share
    if not np.isfinite(initial).all():return np.full(2,np.nan),-1,0,0,0,1
    sy,sx=np.mgrid[-6:7,-6:7];shifts=np.c_[sx.ravel(),sy.ravel()]
    candidates=initial+shifts
    cc,_,_=correlations(candidates);scores=cc-.002*np.sum(shifts**2,axis=1)
    order=np.argsort(-scores);ix=[order[0]]
    for ii in order[1:]:
        if np.linalg.norm(candidates[ii]-candidates[ix[0]])>2.5:ix.append(ii);break
    fits=[]
    for ii in ix:
        opt=minimize(lambda v:1-correlations(v[None])[0][0]+.002*np.sum((v-initial)**2),candidates[ii],method='Nelder-Mead',options={'maxiter':70,'xatol':.02,'fatol':1e-5})
        fits.append((opt.fun,opt.x))
    fits.sort(key=lambda f:f[0]);obj,disp=fits[0]
    other=np.linalg.norm(candidates-disp,axis=1)>2.5
    alt=[np.min(1-scores[other])]
    alt.extend(f[0] for f in fits[1:] if np.linalg.norm(f[1]-disp)>2.5)
    gap=max(0,min(alt)-obj)
    cor,n,share=correlations(disp[None])
    return disp,float(cor[0]),float(gap),int(n[0]),float(share[0]),int(badj)

def run(radius=4,affine=True):
    name=('affine' if affine else 'translation')+str(radius)
    t0=time.time(); ini,g=F.evaluate(QUERY)
    bad=~np.isfinite(ini).all(axis=1)
    if bad.any():
        v=PTV.evaluate(QUERY[bad]);ini[bad]=v['disp'];g[bad]=v['local_polynomial_gradient']
    with ThreadPoolExecutor(max_workers=3) as ex:
        fw=list(ex.map(lambda i:fit(QUERY[i],ini[i],g[i],radius,affine),range(len(QUERY))))
    disp=np.array([v[0] for v in fw]);dest=QUERY+disp
    bini,bg=R.evaluate(dest)
    missing=~np.isfinite(bini).all(axis=1)
    bini[missing]=-disp[missing]
    for i in np.where(missing)[0]:
        try:bg[i]=np.linalg.inv(np.eye(2)+g[i])-np.eye(2)
        except np.linalg.LinAlgError:bg[i]=0
    with ThreadPoolExecutor(max_workers=3) as ex:
        bw=list(ex.map(lambda i:fit(dest[i],bini[i],bg[i],radius,affine,True),range(len(QUERY))))
    bk=np.array([v[0] for v in bw]);fb=np.linalg.norm(disp+bk,axis=1)
    ncc=np.array([v[1] for v in fw]);bncc=np.array([v[1] for v in bw])
    gap=np.array([v[2] for v in fw]);bgap=np.array([v[2] for v in bw])
    depth=QUERY[:,1]-np.interp(QUERY[:,0],np.arange(501),SA)
    target_depth=dest[:,1]-np.interp(dest[:,0],np.arange(501),SB)
    dtree=cKDTree(PB);distance,nearest=dtree.query(dest)
    accepted=(ncc>.75)&(bncc>.75)&(fb<.6)&(gap>.025)&(bgap>.025)&(target_depth>=2)&(distance<1.75)
    # Unique destination feature among automatically detected sources.
    keep=accepted.copy()
    for target in np.unique(nearest[:len(PA)][accepted[:len(PA)]]):
        ids=np.where((nearest[:len(PA)]==target)&accepted[:len(PA)])[0]
        if len(ids)>1:
            best=ids[np.argmax(np.minimum(ncc[ids],bncc[ids])-fb[ids]*.1)]
            keep[ids]=False;keep[best]=True
    vals=dict(points=QUERY,disp=disp,initial=ini,ncc=ncc,back_ncc=bncc,fb=fb,gap=gap,back_gap=bgap,
              accepted=keep,accepted_before_unique=accepted,depth=depth,target_depth=target_depth,
              target_feature_distance=distance,back_disp=bk,auto_count=len(PA),radius=radius,affine=affine,
              forward_usable_pixels=np.array([v[3] for v in fw]),backward_usable_pixels=np.array([v[3] for v in bw]),
              forward_share=np.array([v[4] for v in fw]),backward_share=np.array([v[4] for v in bw]),
              forward_deformation_fallback=np.array([v[5] for v in fw]),backward_deformation_fallback=np.array([v[5] for v in bw]))
    np.savez_compressed(OUT/(name+'.npz'),**vals)
    truth=M['p'][:,1].T-M['p'][:,0].T;err=np.linalg.norm(disp[len(PA):]-truth,axis=1)
    metrics={}
    for label,s in [('manual_all',np.ones(200,bool)),('manual_near40',depth[len(PA):]<40),('manual_accepted',keep[len(PA):])]:
        metrics[label]={'n':int(s.sum()),'mean':float(np.mean(err[s])) if s.any() else None,'median':float(np.median(err[s])) if s.any() else None}
    metrics['auto_bins']={str(lo)+'-'+str(hi):int(np.sum(keep[:len(PA)]&(DEP>=lo)&(DEP<hi))) for lo,hi in [(2,6),(6,10),(10,14),(14,20),(20,40)]}
    metrics['auto_total']=int(keep[:len(PA)].sum());metrics['seconds']=time.time()-t0
    (OUT/(name+'.json')).write_text(json.dumps(metrics,indent=2));print(name,json.dumps(metrics),flush=True)

if __name__=='__main__':
    print('automatic candidates',len(PA),'closer14',int((DEP<14).sum()),flush=True)
    for radius,affine in [(4,False),(4,True),(6,True)]:run(radius,affine)
