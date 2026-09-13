import numpy as np,cv2,json
from pathlib import Path
W=Path('work');D=Path('historical-user-files/Downloads');z=np.load(W/'metadata.npz');base=np.load(W/'baseline_reference_matlab.npz');F=base['flow'];xy=base['xy'];e=base['error'];depth=base['depth']
a=cv2.imread(str(D/'ExpLCL_1_03-123_imgA.tif'),0);b=cv2.imread(str(D/'ExpLCL_1_03-123_imgB.tif'),0)
def prep(I):return np.clip((I.astype(np.float32)-cv2.GaussianBlur(I.astype(np.float32),(0,0),5))*.9+128,0,255).astype(np.uint8)
A=prep(a);B=prep(b)
alg=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);alg.setFinestScale(0);alg.setPatchSize(8);alg.setPatchStride(4);alg.setGradientDescentIterations(50);alg.setVariationalRefinementIterations(10)
G=alg.calc(B,A,None);y,x=np.mgrid[:501,:501].astype(np.float32);xx=x+F[:,:,0];yy=y+F[:,:,1]
gr=cv2.remap(G,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=1000);fb=np.linalg.norm(F+gr,axis=2)
Af=A.astype(np.float32)-128;Bw=cv2.remap(B.astype(np.float32)-128,xx,yy,cv2.INTER_LINEAR)
valid=(y>(z['surfa'].ravel()-1)[None,:]+7)&(yy>np.interp(xx,np.arange(501),z['surfb'].ravel()-1)+7)&(xx>=0)&(xx<500)&(yy>=0)&(yy<500)
weight=valid.astype(np.float32);blur=lambda v: cv2.boxFilter(v,-1,(15,15),normalize=False,borderType=cv2.BORDER_CONSTANT)
n=blur(weight);ma=blur(Af*weight)/np.maximum(n,1);mb=blur(Bw*weight)/np.maximum(n,1)
cov=blur(Af*Bw*weight)/np.maximum(n,1)-ma*mb;va=blur(Af*Af*weight)/np.maximum(n,1)-ma*ma;vb=blur(Bw*Bw*weight)/np.maximum(n,1)-mb*mb
ncc=cov/np.sqrt(np.maximum(va*vb,1));ncc[n<100]=np.nan
sample=lambda im:cv2.remap(im,xy[:,0].reshape(-1,1),xy[:,1].reshape(-1,1),cv2.INTER_LINEAR).ravel()
pfb=sample(fb);pncc=sample(ncc);good=(pfb<1.5)&(pncc>.45)
print('Confidence fixed FB<1.5px, NCC>.45:',good.sum(),'of',len(good),'median',np.median(e[good]),'mean',np.mean(e[good]),'p90',np.percentile(e[good],90),'within1',np.mean(e[good]<1))
for k,sel in [('depth<=40',depth<=40),('depth>40',depth>40)]:
 u=good&sel;print(k,'accepted',u.sum(),'of',sel.sum(),'EPE mean',np.mean(e[u]),'p90',np.percentile(e[u],90))
for i in np.where(e>2)[0]:print('failure',i,'epe',round(float(e[i]),2),'fb',round(float(pfb[i]),2),'ncc',round(float(pncc[i]),2),'pass',good[i])
np.savez_compressed(W/'baseline_confidence_matlab.npz',forward=F,backward=G,fb_error=fb,local_ncc=ncc,valid=valid,manual_fb=pfb,manual_ncc=pncc,manual_accepted=good)
