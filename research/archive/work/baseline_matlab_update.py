import numpy as np,cv2,json
from pathlib import Path
W=Path('work');z=np.load(W/'metadata.npz');f=np.load(W/'baseline_flows.npz');xy=(z['p'][:,0,:].T-1).astype(np.float32);truth=(z['p'][:,1,:]-z['p'][:,0,:]).T;surf=z['surfa'].ravel()-1;depth=xy[:,1]-np.interp(xy[:,0],np.arange(501),surf)
def sample(flow,q):return cv2.remap(flow,q[:,0].reshape(-1,1),q[:,1].reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(-1,2)
rows=[]
for name in f.files:
 if f[name].shape!=(501,501,2):continue
 pred=sample(f[name],xy);e=np.linalg.norm(pred-truth,axis=1);row={'method':name,'metrics':{}}
 for label,sel in [('all',np.ones(200,bool)),('0_20',depth<=20),('20_40',(depth>20)&(depth<=40)),('40_80',(depth>40)&(depth<=80)),('80_plus',depth>80)]:
  valid=sel&np.isfinite(e);a=e[valid];row['metrics'][label]={'n':int(valid.sum()),'median_epe':float(np.median(a)) if len(a) else None,'mean_epe':float(np.mean(a)) if len(a) else None,'rmse_epe':float(np.sqrt(np.mean(a*a))) if len(a) else None,'p90_epe':float(np.percentile(a,90)) if len(a) else None,'fraction_under1':float(np.mean(a<1)) if len(a) else None,'fraction_under2':float(np.mean(a<2)) if len(a) else None}
 rows.append(row)
 if name=='dis_highpass_p8':np.savez_compressed(W/'baseline_reference_matlab.npz',flow=f[name],xy=xy,pred=pred,truth=truth,depth=depth,error=e)
(W/'baseline_metrics_matlab.json').write_text(json.dumps(rows,indent=2));print([a for a in rows if a['method']=='dis_highpass_p8']);print('depth',depth.min(),np.percentile(depth,10))
# Regenerate selected DIS confidence on exactly same coordinate convention.
s=(W/'baseline_confidence.py').read_text().replace("baseline_reference.npz","baseline_reference_matlab.npz").replace("baseline_confidence.npz","baseline_confidence_matlab.npz").replace("z['surfa'].ravel()[None,:]+7","(z['surfa'].ravel()-1)[None,:]+7").replace("z['surfb'].ravel())+7","z['surfb'].ravel()-1)+7")
(W/'baseline_confidence_matlab.py').write_text(s)
