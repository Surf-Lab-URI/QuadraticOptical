from pathlib import Path
import numpy as np,json
W=Path('work');names=['spline_converged_l0p0001','spline_converged_l0p001','spline_converged_l0p01'];zs=[np.load(W/(n+'.npz')) for n in names];p=zs[0]['points'];par=np.array([z['params'] for z in zs]);m=np.load(W/'metadata.npz');depth=p[:,1]-np.interp(p[:,0],np.arange(501),m['surfa'].ravel()-1)
# These are deterministic sensitivity ranges, not confidence intervals.
spread_uv=np.max(np.linalg.norm(par[:,:,:,0]-par[1:2,:,:,0],axis=2),axis=0);range_dudx=np.ptp(par[:,:,0,1],axis=0);range_dvdx=np.ptp(par[:,:,1,1],axis=0);det=(1+par[:,:,0,1])*(1+par[:,:,1,2])-par[:,:,0,2]*par[:,:,1,1]
np.savez_compressed(W/'spline_sensitivity.npz',points=p,depth=depth,displacement_models=par[:,:,:,0],params_models=par,lam=np.array([.0001,.001,.01]),max_displacement_deviation_from_middle=spread_uv,dudx_range=range_dudx,dvdx_range=range_dvdx,mapping_determinant_models=det)
for label,use in [('all',np.ones(len(p),bool)),('near<40',depth<40)]:print(label,'N',use.sum(),'flow sensitivity p50,p90,max',np.percentile(spread_uv[use],[50,90,100]),'dudx_range p50,p90,max',np.percentile(range_dudx[use],[50,90,100]))
