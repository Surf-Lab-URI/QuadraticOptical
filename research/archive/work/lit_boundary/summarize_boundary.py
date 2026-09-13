import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,csv
from pathlib import Path
sys.path.insert(0,'work')
import numpy as np
from field_model import LocalField
from scipy.spatial import cKDTree
O=Path('work/lit_boundary');names=['continuation_control','surface_strip','surface_strip_projected','surface_strip_glare','surface_strip_curvature_prior'];fin=np.load('work/final_results.npz')
# Freeze unmodified deep reverse fits to original baseline, matching runner.
# Reevaluate composed FB after this isolation step; manual errors unaffected.
_b=np.load('work/final_reverse_field.npz');backbase={k:_b[k] for k in _b.files}
for name in names:
    path=O/(name+'_reverse.npz');a=dict(np.load(path));fw=dict(np.load(O/(name+'.npz')));unmodified=~fw['refit']
    for key in ['params','ncc','rms','nvalid','cond','iterations','mindet','support_fraction']:
        a[key][unmodified]=backbase[key][unmodified]
    a['points'][unmodified]=backbase['points'][unmodified]
    np.savez_compressed(path,**a)
    v=dict(np.load(O/(name+'_evaluation.npz')));mod=LocalField(path);bu,_,bncc=mod.evaluate(v['query']+v['disp'],True)
    v['fb']=np.linalg.norm(v['disp']+bu,axis=1);v['back_ncc']=bncc;np.savez_compressed(O/(name+'_evaluation.npz'),**v)
    summ=json.loads((O/(name+'_summary.json')).read_text());depth=v['depth']
    for lab,sel in [('all',np.ones(200,bool)),('lt20',depth[:200]<20),('lt40',depth[:200]<40),('20to40',(depth[:200]>=20)&(depth[:200]<40)),('40to80',(depth[:200]>=40)&(depth[:200]<80)),('80plus',depth[:200]>=80)]:
        summ[lab]['fb_median']=float(np.nanmedian(v['fb'][:200][sel]));summ[lab]['fb_p90']=float(np.nanpercentile(v['fb'][:200][sel],90))
    for c in summ['closest_two']:c['fb']=float(v['fb'][c['pick']-1])
    summ['deep_reverse_frozen_to_baseline']=True
    summ['grid_diagnostics']['forward_backward_grid_p90']=float(np.nanpercentile(v['fb'][200:][fw['refit']],90))
    (O/(name+'_summary.json')).write_text(json.dumps(summ,indent=2))

allvals={n:dict(np.load(O/(n+'_evaluation.npz'))) for n in names};out={}
for name,v in allvals.items():
 model=LocalField(O/(name+'.npz'));back=LocalField(O/(name+'_reverse.npz'));q=v['query'];g=v['gradient'];u=v['disp'];depth=v['depth'];rr={}
 for label,mod,pts in [('forward',model,q),('reverse',back,q+u)]:
  nonfold=[];support=[];joint=[];cond=[];eff=[]
  data=mod.data
  for pt in pts:
   if not np.isfinite(pt).all():nonfold.append(0.);support.append(0.);joint.append(0.);cond.append(np.nan);eff.append(np.nan);continue
   ii=np.asarray(mod.tree.query_ball_point(pt,mod.R));d=np.linalg.norm(pt-mod.points[ii],axis=1)/mod.R;w=(1-d)**4*(1+4*d)
   if not len(ii) or w.sum()<1e-12:nonfold.append(0.);support.append(0.);joint.append(0.);cond.append(np.nan);eff.append(np.nan);continue
   w/=w.sum();jac=data['mindet'][ii]>.05;overlap=data['support_fraction'][ii]>.85
   nonfold.append(np.dot(w,jac));support.append(np.dot(w,overlap));joint.append(np.dot(w,jac&overlap));cond.append(np.dot(w,data['cond'][ii]));eff.append(np.dot(w,data['effective_n'][ii]))
  rr[label+'_nonfolding_share']=np.asarray(nonfold);rr[label+'_support_share']=np.asarray(support);rr[label+'_valid_share']=np.asarray(joint);rr[label+'_condition']=np.asarray(cond);rr[label+'_effective_n']=np.asarray(eff)
 common=(depth>=12)&(depth<=150)&(v['target_depth']>=10)&fin['support'][:927]&(v['ncc']>=.6)&(v['fb']<=1)&(v['determinant']>.2)&np.isfinite(u).all(axis=1)
 local=common&(rr['forward_valid_share']>=.95)&(rr['reverse_valid_share']>=.95)
 # Sensitivity to all three independent historical variants plus the unchanged baseline.
 oldalts=np.concatenate([fin['disp'][None,:927],fin['alternative_displacements'][:,:927]],axis=0)
 s=np.max(np.linalg.norm(oldalts-u[None],axis=2),axis=0)
 sensitivityfinite=np.isfinite(oldalts).all(axis=(0,2))
 screened=local&sensitivityfinite&(s<=1.5)
 grad_s=np.max(np.abs(fin['alternative_gradients'][:,:927,0,0]-g[None,:,0,0]),axis=0)
 gradok=screened&(depth>=20)&(fin['nearest_feature'][:927]<=10)&(grad_s<=.08)
 eps=1e-4;test=q[:200];fd=(model.evaluate(test+[eps,0])[0]-model.evaluate(test-[eps,0])[0])/(2*eps);fdmax=float(np.nanmax(abs(fd-g[:200,:,0])))
 rr.update(screened=screened,gradient_screened=gradok,common_image_geometry=common,local_quality_pass=local,old_method_spread=s,old_gradient_spread=grad_s)
 np.savez_compressed(O/(name+'_checks.npz'),**rr)
 summary=json.loads((O/(name+'_summary.json')).read_text());summary['derivative_fd_max']=fdmax
 summary['coverage']={}
 for label,ii in [('manual',np.arange(200)),('grid',np.arange(200,927))]:
  summary['coverage'][label]={}
  for band,bs in [('all',np.ones(927,bool)),('lt20',depth<20),('lt40',depth<40),('20to40',(depth>=20)&(depth<40))]:
   ids=ii[bs[ii]];acc=ids[screened[ids]];go=ids[gradok[ids]];props=dict(total=len(ids),image_geometry_pass=int(common[ids].sum()),local_quality_pass=int(local[ids].sum()),screened=len(acc),gradient_screened=len(go),min_screened_depth=float(np.min(depth[acc])) if len(acc) else None)
   if label=='manual':props['screened_mean_manual_error']=float(np.mean(v['manual_error'][acc])) if len(acc) else None
   summary['coverage'][label][band]=props
 summary['sensitivity']={}
 for band,bs in [('lt40',depth<40),('all',np.ones(927,bool))]:
  ids=np.arange(200,927)[bs[200:]];summary['sensitivity'][band]=dict(displacement_change_median=float(np.nanmedian(np.linalg.norm(u[ids]-fin['disp'][ids],axis=1))),horizontal_gradient_change_median=float(np.nanmedian(np.abs(g[ids,0,0]-fin['gradient'][ids,0,0]))),horizontal_gradient_change_p90=float(np.nanpercentile(np.abs(g[ids,0,0]-fin['gradient'][ids,0,0]),90)),min_local_nonfolding_share=float(np.nanmin(rr['forward_nonfolding_share'][ids])))
 out[name]=summary
(O/'comparison.json').write_text(json.dumps(out,indent=2))
with (O/'comparison.csv').open('w') as f:
 wr=csv.writer(f);wr.writerow(['variant','mean_epe_all','mean_epe_lt40','mean_epe_lt20','fbmedian_lt40','fbp90_lt40','grid_screened','grid_lt40_screened','manual_screened','manual_lt40_screened','grid_gradient_screened'])
 for name,s in out.items():wr.writerow([name,s['all']['mean'],s['lt40']['mean'],s['lt20']['mean'],s['lt40']['fb_median'],s['lt40']['fb_p90'],s['coverage']['grid']['all']['screened'],s['coverage']['grid']['lt40']['screened'],s['coverage']['manual']['all']['screened'],s['coverage']['manual']['lt40']['screened'],s['coverage']['grid']['all']['gradient_screened']])
print((O/'comparison.csv').read_text(),flush=True)
print('finite-difference maxima',{n:s['derivative_fd_max'] for n,s in out.items()})
