import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json
from pathlib import Path
sys.path.insert(0,'work');import lit_particle_extension as m
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('work/lit_particles');names=['translation4','affine4','affine6'];data={n:dict(np.load(O/(n+'.npz'))) for n in names};n=int(data['affine4']['auto_count']);a=data['affine4'];checks=[];details={}
truth=m.M['p'][:,1].T-m.M['p'][:,0].T
for name,f in data.items():
 assert n==int(f['auto_count']) and np.array_equal(f['points'],np.r_[m.PA,m.MAN]);assert np.allclose(f['depth'],f['points'][:,1]-np.interp(f['points'][:,0],np.arange(501),m.SA))
 dd,nn=cKDTree(m.PB).query(f['points']+f['disp'])
 assert np.allclose(dd,f['target_feature_distance']);assert np.allclose(f['fb'],np.linalg.norm(f['disp']+f['back_disp'],axis=1))
 crit=dict(ncc=f['ncc']>.75,back_ncc=f['back_ncc']>.75,fb=f['fb']<.6,gap=f['gap']>.025,back_gap=f['back_gap']>.025,target_depth=f['target_depth']>=2,target_feature_distance=f['target_feature_distance']<1.75)
 raw=np.logical_and.reduce(list(crit.values()));assert np.array_equal(raw,f['accepted_before_unique'])
 keep=raw.copy()
 for target in np.unique(nn[:n][raw[:n]]):
  ids=np.where((nn[:n]==target)&raw[:n])[0]
  if len(ids)>1:
   best=ids[np.argmax(np.minimum(f['ncc'][ids],f['back_ncc'][ids])-.1*f['fb'][ids])];keep[ids]=False;keep[best]=True
 assert np.array_equal(keep,f['accepted']);assert len(np.unique(nn[:n][keep[:n]]))==sum(keep[:n])
 assert np.array_equal(keep[n:],raw[n:])
 idx=np.where(keep[:n]&(f['depth'][:n]<14))[0]
 fallback=(f['forward_deformation_fallback'][:n]>0)|(f['backward_deformation_fallback'][:n]>0)
 checks.append(dict(variant=name,auto_count=n,manual_count=len(f['points'])-n,accepted_auto=int(sum(keep[:n])),accepted_manual=int(sum(keep[n:])),before_unique_auto=int(sum(raw[:n])),duplicate_destinations_removed=int(sum(raw[:n])-sum(keep[:n])),accepted_below14=len(idx),all_below14_indices=idx.tolist(),accepted_auto_affine_fallback=int(sum(keep[:n]&fallback))))
 details[name]={}
 for i in np.where(a['accepted'][:n]&(a['depth'][:n]<14))[0]:
  d={key:f[key][i].tolist() for key in ['points','disp','initial','depth','target_depth','ncc','back_ncc','fb','gap','back_gap','forward_usable_pixels','backward_usable_pixels','forward_share','backward_share','target_feature_distance','forward_deformation_fallback','backward_deformation_fallback','accepted']}
  d['target_xy']=(f['points'][i]+f['disp'][i]).tolist();d['target_detected_feature_xy']=m.PB[nn[i]].tolist();d['failed_criteria']=[k for k,c in crit.items() if not c[i]];d['initial_correction_px']=float(np.linalg.norm(f['disp'][i]-f['initial'][i]));details[name][str(i)]=d
inter=np.logical_and.reduce([d['accepted'][:n] for d in data.values()]);aff=data['affine4']['accepted'][:n]&data['affine6']['accepted'][:n]
shallow=np.where(inter&(a['depth'][:n]<14))[0]
spread={str(i):dict(affine4_vs_affine6=float(np.linalg.norm(data['affine4']['disp'][i]-data['affine6']['disp'][i])),affine4_vs_translation4=float(np.linalg.norm(data['affine4']['disp'][i]-data['translation4']['disp'][i])),max_pairwise=float(max(np.linalg.norm(data[s]['disp'][i]-data[t]['disp'][i]) for s in names for t in names))) for i in np.where(a['accepted'][:n]&(a['depth'][:n]<14))[0]}
res=dict(checks=checks,automatic_below14=int(sum(a['depth'][:n]<14)),affine_intersection=int(sum(aff)),all_three_intersection=int(sum(inter)),below14_affine_intersection=int(sum(aff&(a['depth'][:n]<14))),below14_all_three_intersection=int(sum(inter&(a['depth'][:n]<14))),stable_shallow_source_indices=shallow.tolist(),details=details,spread=spread,surface_a_depression_apex=dict(x=int(np.argmax(m.SA)),y=float(np.max(m.SA))))
(O/'audit.json').write_text(json.dumps(res,indent=2));print(json.dumps({k:v for k,v in res.items() if k not in ['details']},indent=2))
# Scientific diagnostic contact sheet of original pixels. No enhancement or reconstruction.
indices=np.where(a['accepted'][:n]&(a['depth'][:n]<14))[0];fig,axs=plt.subplots(3,2,figsize=(12,10.5));margin=16
for row,i in enumerate(indices):
 for col,(raw,surf,point,title) in enumerate([(m.RA,m.SA,a['points'][i],'A source'),(m.RB,m.SB,a['points'][i]+a['disp'][i],'B destination, affine4')]):
  ax=axs[row,col];x,y=point;x0=max(0,int(x)-margin);x1=min(raw.shape[1],int(x)+margin+1);y0=max(0,int(y)-margin);y1=min(raw.shape[0],int(y)+margin+1)
  ax.imshow(raw[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=255,extent=(x0-.5,x1-.5,y1-.5,y0-.5),interpolation='nearest')
  xs=np.arange(x0,x1);ax.plot(xs,surf[xs],color='#f8cf47',lw=1);ax.plot(x,y,'+',color='#ef5e62',ms=12,mew=1.4);ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
  ax.set_title(f'{title}: ({x:.2f}, {y:.2f})\nsource ({a["points"][i,0]:.0f},{a["points"][i,1]:.0f}), depth {a["depth"][i]:.2f} px; '+('stable across 3' if inter[i] else 'fails other configurations'),fontsize=10);ax.set_xlabel('x [px]');ax.set_ylabel('y down [px]')
fig.suptitle('Raw pixel crops for the three shallow affine4 candidates\nRed cross: queried center; yellow line: supplied surface',fontsize=12);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(O/'audit_raw_crops.png',dpi=160);plt.close(fig)
