import numpy as np,json
from scipy.spatial import cKDTree
from pathlib import Path
z=np.load('work/final_results.npz');audit=json.loads(Path('work/coverage_audit.json').read_text());q=z['query'];valids={}
for label,path,query in [('forward','work/final_ptv13.npz',q),('reverse','work/final_reverse_field.npz',q+z['disp'])]:
 f=np.load(path);t=cKDTree(f['points']);shares=[];support=[]
 for pt in query:
  if not np.isfinite(pt).all():shares.append(0.);support.append(0.);continue
  ii=t.query_ball_point(pt,16)
  if not ii:shares.append(0.);support.append(0.);continue
  dd=np.linalg.norm(pt-f['points'][ii],axis=1)/16;w=(1-dd)**4*(1+4*dd)
  if w.sum()<1e-12:shares.append(0.);support.append(0.);continue
  shares.append(np.dot(w,f['mindet'][ii]>.05)/w.sum());support.append(np.dot(w,f['support_fraction'][ii])/w.sum())
 valids[label+'_nonfolding_share']=np.array(shares);valids[label+'_support_mean']=np.array(support)
orig=z['accepted'];extra=np.zeros(len(q),bool);extra[audit['provisional_indices']['window_only_additional']]=True
nofold=extra&(valids['forward_nonfolding_share']>=.95)&(valids['reverse_nonfolding_share']>=.95)
stats={}
for name,ids in [('manual',np.arange(200)),('grid',np.arange(200,927)),('dense',np.arange(927,len(q)))]:
 stats[name]={}
 for band,condition in [('lt20',z['depth']<20),('20to40',(z['depth']>=20)&(z['depth']<40)),('all',np.ones(len(q),bool))]:
  ii=ids[condition[ids]];sel=ii[nofold[ii]];old=ii[orig[ii]]
  stats[name][band]={'original':len(old),'additional':len(sel),'combined':len(sel)+len(old),'min_extra_depth':float(np.min(z['depth'][sel])) if len(sel) else None,'additional_query_indices':sel.tolist()}
  if name=='manual' and len(sel):stats[name][band]['posthoc_extra_mean_epe']=float(np.mean(z['manual_error'][sel]));stats[name][band]['picks']=(sel+1).tolist()
audit['local_nonfolding_provisional']=stats
audit['provisional_indices']['support_only_additional']=np.flatnonzero(nofold).tolist()
Path('work/coverage_audit.json').write_text(json.dumps(audit,indent=2))
np.savez_compressed('work/coverage_local_diagnostics.npz',**valids,support_only_additional=nofold,window_only_additional=extra)
print(json.dumps({k:{b:{kk:vv for kk,vv in v.items() if kk!='additional_query_indices'} for b,v in val.items()} for k,val in stats.items()},indent=2))
