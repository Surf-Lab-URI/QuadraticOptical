"""Image-only optical-flow baselines. Manual endpoints used exclusively in validation.
Fixed settings sweep is reported in full; no manual displacements initialize flow.
"""
import cv2,numpy as np,json,time
from pathlib import Path
ROOT=Path('work')
DATA=Path('historical-user-files/Downloads'); Z=np.load(ROOT/'metadata.npz')
A=cv2.imread(str(DATA/'ExpLCL_1_03-123_imgA.tif'),0); B=cv2.imread(str(DATA/'ExpLCL_1_03-123_imgB.tif'),0)
sA=Z['surfa'].ravel()-1;sB=Z['surfb'].ravel()-1
p=Z['p'];xy=(p[:,0,:].T-1).astype(np.float32); truth=(p[:,1,:]-p[:,0,:]).T
Y,X=np.mgrid[:A.shape[0],:A.shape[1]]
def prep(I,mode,surf):
 f=I.astype(np.float32)
 if mode=='raw':return I
 hp=f-cv2.GaussianBlur(f,(0,0),5)
 if mode in ['whiten','masked']:
  rms=np.sqrt(cv2.GaussianBlur(hp*hp,(0,0),9)+25)
  hp=hp/rms*35
 else:hp=hp*.9
 if mode=='masked':
  weight=np.clip((Y-surf[None,:]-7)/5,0,1);hp=hp*weight
 return np.clip(hp+128,0,255).astype(np.uint8)
def sample(flow,q=xy):
 return cv2.remap(flow,q[:,0].reshape(-1,1),q[:,1].reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(-1,2)
def metrics(v,q=xy,valid=None):
 e=np.linalg.norm(v-truth,axis=1);dep=q[:,1]-np.interp(q[:,0],np.arange(501),sA)
 if valid is None:valid=np.all(np.isfinite(v),axis=1)
 res={}
 for label,sel in [('all',np.ones(200,bool)),('depth_0_10',dep<=10),('depth_10_20',(dep>10)&(dep<=20)),('depth_20_40',(dep>20)&(dep<=40)),('depth_40_plus',dep>40)]:
  use=sel&valid
  res[label]={'n':int(use.sum()),'median':float(np.median(e[use])) if use.any() else None,'mean':float(np.mean(e[use])) if use.any() else None,'p90':float(np.percentile(e[use],90)) if use.any() else None,'within_1px':float(np.mean(e[use]<=1)) if use.any() else None,'within_2px':float(np.mean(e[use]<=2)) if use.any() else None}
 return res
results=[]; arrays={'xy':xy,'truth':truth,'depth':xy[:,1]-np.interp(xy[:,0],np.arange(501),sA)}
def record(name,flow,sec):
 pred=sample(flow);met=metrics(pred);met0=metrics(sample(flow,xy+1),xy+1)
 arrays[name]=flow;arrays[name+'_pred']=pred
 row={'name':name,'seconds':sec,'matlab_minus1':met,'offset0':met0};results.append(row)
 print(name,round(sec,2),'all',met['all'],'near',met['depth_0_10'],flush=True)
 np.savez_compressed(ROOT/'baseline_flows.npz',**arrays)
 (ROOT/'baseline_metrics.json').write_text(json.dumps(results,indent=2))
for mode in ['raw','highpass','whiten','masked']:
 a=prep(A,mode,sA);b=prep(B,mode,sB)
 for win in [21,41]:
  tic=time.time();f=cv2.calcOpticalFlowFarneback(a,b,None,.5,4,win,7,7,1.5,0);record('farneback_'+mode+'_w'+str(win),f,time.time()-tic)
 for preset in [cv2.DISOPTICAL_FLOW_PRESET_MEDIUM]:
  for patch in [8,16]:
   alg=cv2.DISOpticalFlow_create(preset);alg.setFinestScale(0);alg.setPatchSize(patch);alg.setPatchStride(max(3,patch//2));alg.setGradientDescentIterations(50);alg.setVariationalRefinementIterations(10)
   tic=time.time();f=alg.calc(a,b,None);record('dis_'+mode+'_p'+str(patch),f,time.time()-tic)
 # Uniform 3-pixel LK sampling avoids manual positions in estimation.
 gx,gy=np.meshgrid(np.arange(6,496,3),np.arange(6,496,3));q=np.stack([gx.ravel(),gy.ravel()],axis=1).astype(np.float32).reshape(-1,1,2)
 for win in [15,31]:
  tic=time.time();r,status,err=cv2.calcOpticalFlowPyrLK(a,b,q,None,winSize=(win,win),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,.005),minEigThreshold=1e-5)
  disp=(r-q).reshape(gy.shape+(2,));disp[status.reshape(gy.shape)==0]=np.nan
  mx=((X-6)/3).astype(np.float32);my=((Y-6)/3).astype(np.float32)
  f=cv2.remap(disp,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  record('lk_'+mode+'_w'+str(win),f,time.time()-tic)
print('FINISHED')
