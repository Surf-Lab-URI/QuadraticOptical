"""Independent round-trip and consistency QA of exported numeric deliverables."""
import csv,json,time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parent.parent
O=ROOT/'outputs';W=ROOT/'work'
need=['manual_comparison.csv','predicted_grid.csv','velocity_results.mat','velocity_results.npz','validation_summary.json']
start=time.time()
while not all((O/n).exists() for n in need):
 if time.time()-start>90:raise RuntimeError('Numeric exports not yet complete: '+', '.join(n for n in need if not (O/n).exists()))
 time.sleep(1)
z=np.load(W/'final_results.npz');out=np.load(O/'velocity_results.npz');m=loadmat(O/'velocity_results.mat');summ=json.loads((O/'validation_summary.json').read_text());orig=np.load(W/'metadata.npz')
checks=[]
def check(label,passed,details=None):
 checks.append({'check':label,'passed':bool(passed),'details':details})
 if not passed:print('FAIL',label,details,flush=True)
def same(label,a,b,atol=1e-12):
 a=np.asarray(a);b=np.asarray(b);ok=a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=0,equal_nan=True)
 dif=float(np.nanmax(abs(a-b))) if a.shape==b.shape and np.issubdtype(a.dtype,np.number) and np.issubdtype(b.dtype,np.number) and a.size and np.any(np.isfinite(a)&np.isfinite(b)) else None
 check(label,ok,dict(shape_a=list(a.shape),shape_b=list(b.shape),max_abs_difference=dif))
N=int(z['grid_count']);sl=slice(200,200+N);ds=slice(200+N,None);q=z['query'];d=z['disp'];g=z['gradient'];shape=z['dense_x'].shape
for k in z.files:same('NPZ:'+k,out[k],z[k])
check('NPZ unknown calibration marked NaN',np.isnan(out['frame_interval_seconds']) and np.isnan(out['length_per_pixel']))
matmap={'manual_xy_zero_based':q[:200],'manual_xy_matlab':q[:200]+1,'manual_displacement_px_per_pair':z['manual_truth'],'predicted_at_manual_px_per_pair':d[:200],'grid_xy_zero_based':q[sl],'grid_displacement_px_per_pair':d[sl],'grid_displacement_gradient':g[sl],'dense_x_px':z['dense_x'],'dense_y_px':z['dense_y'],'dense_dx_px_per_pair':d[ds,0].reshape(shape),'dense_dy_down_px_per_pair':d[ds,1].reshape(shape),'dense_d_dx_dx':g[ds,0,0].reshape(shape),'dense_accepted':z['accepted'][ds].reshape(shape),'dense_gradient_accepted':z['gradient_accepted'][ds].reshape(shape),'grid_midpoint_xy':q[sl]+d[sl]/2}
for k,value in matmap.items():same('MAT:'+k,m[k],value)
for k,key,sel in [('manual_disagreement_px','manual_error',slice(None)),('manual_prediction_accepted','accepted',slice(0,200)),('grid_accepted','accepted',sl),('grid_gradient_accepted','gradient_accepted',sl),('grid_forward_backward_error_px','fb',sl),('grid_method_spread_px','method_spread',sl),('grid_gradient_sensitivity','gradient_spread',sl),('surface_a_y_zero_based','surface_a',slice(None)),('surface_b_y_zero_based','surface_b',slice(None))]:same('MAT:'+k,m[k].ravel(),z[key][sel])
true_mid=g[sl]@np.linalg.inv(np.eye(2)+g[sl]/2);same('MAT:midpoint gradient chain rule',m['grid_midpoint_displacement_gradient'],true_mid)
check('MAT unknown calibration marked NaN',np.isnan(m['frame_interval_seconds']).all() and np.isnan(m['length_per_pixel']).all())
same('MAT manual MATLAB coordinates match input file',m['manual_xy_matlab'],orig['p'][:,0].T)
same('MAT manual displacement matches original pairs',m['manual_displacement_px_per_pair'],orig['p'][:,1].T-orig['p'][:,0].T)
for name,sel,is_manual in [('manual_comparison.csv',slice(0,200),True),('predicted_grid.csv',sl,False)]:
 with (O/name).open() as stream:reader=csv.DictReader(stream);rows=list(reader)
 check('CSV:'+name+' row count',len(rows)==(200 if is_manual else N))
 arr={k:np.array([float(row[k]) for row in rows]) for k in rows[0]}
 same('CSV:'+name+' row IDs',arr['pick' if is_manual else 'node'],np.arange(1,len(rows)+1))
 for col,value in [('x_px',q[sel,0]),('y_px',q[sel,1]),('dx_px_per_pair',d[sel,0]),('dy_down_px_per_pair',d[sel,1]),('d_dx_dx',g[sel,0,0]),('d_dx_dy',g[sel,0,1]),('d_dy_dx',g[sel,1,0]),('d_dy_dy',g[sel,1,1])]:same('CSV:'+name+':'+col,arr[col],value)
 for k in ['ncc','fb','method_spread','gradient_spread','depth','target_depth','nearest_feature','feature_count25','determinant','alternatives_available','local_valid_share','reverse_valid_share','accepted','gradient_accepted']:same('CSV:'+name+':'+k,arr[k],z[k][sel])
 if is_manual:
  same('CSV manual_dx',arr['manual_dx'],z['manual_truth'][:,0]);same('CSV manual_dy',arr['manual_dy_down'],z['manual_truth'][:,1]);same('CSV manual endpoint error',arr['endpoint_disagreement_px'],np.linalg.norm(d[:200]-z['manual_truth'],axis=1))
e=np.linalg.norm(d[:200]-z['manual_truth'],axis=1);dep=z['depth'][:200];ok=z['accepted'][:200]
for label,mask in [('all',np.ones(200,bool)),('near40',dep<40),('depth40_80',(dep>=40)&(dep<80)),('depth80_plus',dep>=80)]:
 for suffix,sel2 in [('',mask),('_screened',mask&ok)]:
  row=summ['metrics'][label+suffix];v=e[sel2]
  calc=dict(n=len(v),median=float(np.median(v)),mean=float(np.mean(v)),rmse=float(np.sqrt(np.mean(v**2))),p90=float(np.percentile(v,90)),within1=float(np.mean(v<1)),within2=float(np.mean(v<2)))
  for k,value in calc.items():check('JSON:'+label+suffix+':'+k,abs(row[k]-value)<1e-12)
check('Accepted manual count',int(ok.sum())==185)
check('Accepted near-40 manual count',int((ok&(dep<40)).sum())==16)
check('Accepted grid count',int(z['accepted'][sl].sum())==536)
check('Accepted gradient grid count',int(z['gradient_accepted'][sl].sum())==513)
check('Gradient screen is a subset of velocity screen',np.all(~z['gradient_accepted']|z['accepted']))
check('All accepted values finite',np.isfinite(d[z['accepted']]).all() and np.isfinite(g[z['gradient_accepted']]).all())
check('All accepted estimates have three available alternatives',np.all(z['alternatives_available'][z['accepted']]))
check('Alternative-availability flag matches all three stored fields',np.array_equal(z['alternatives_available'],np.all(np.isfinite(z['alternative_displacements']),axis=(0,2)) & np.all(np.isfinite(z['alternative_gradients']),axis=(0,2,3))))
check('Accepted local contribution shares meet threshold',np.all(z['local_valid_share'][z['accepted']]>=.95) and np.all(z['reverse_valid_share'][z['accepted']]>=.95))
check('No accepted velocity in excluded A band',np.all(z['depth'][z['accepted']]>=12))
check('No accepted velocity in excluded B band',np.all(z['target_depth'][z['accepted']]>=10))
check('No accepted gradient above 20px source depth',np.all(z['depth'][z['gradient_accepted']]>=20))
check('All accepted gradients satisfy method sensitivity threshold',np.all(z['gradient_spread'][z['gradient_accepted']]<=.08))
result=dict(checks=len(checks),passed=sum(c['passed'] for c in checks),failed=sum(not c['passed'] for c in checks),rows_manual=200,rows_grid=N,dense_shape=list(shape),source_frame_interval_unknown=True,details=checks)
(W/'numeric_qa.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2),flush=True)
