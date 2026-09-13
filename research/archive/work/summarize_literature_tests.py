import os
os.environ['MPLCONFIGDIR']=os.path.abspath('work/mplcache')
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O=Path('outputs')
read=lambda p:json.loads(Path(p).read_text())
reg=read('work/lit_registration/real_metrics.json')
boundary=read('work/lit_boundary/comparison.json')
shape=read('work/lit_cubicshape/cubic_shape_results.json')
glob=read('work/lit_global/consensus_results.json')
rows=[]
def add(name,family,m,near='near40',notes=''):
 a=m['all'];n=m[near]
 rows.append(dict(method=name,family=family,n_all=a['n'],n_near=n['n'],mean_all_px=a['mean'],
                  mean_near_px=n['mean'],median_all_px=a['median'],median_near_px=n['median'],
                  rmse_all_px=a.get('rms',a.get('rmse')),rmse_near_px=n.get('rms',n.get('rmse')),notes=notes))
add('Original quadratic','reference',reg['baseline'])
add('Extra iterations only','control',reg['forward_refit'])
add('Quadratic + cubic image interpolation','sampling',reg['forward_cubic'],notes='Same quadratic deformation; consistent cubic image values and image derivatives.')
add('Cubic interpolation + projected photometry','sampling',read('work/lit_registration/projected_metrics.json'))
add('Symmetric midpoint + bilinear interpolation','symmetric',reg['symmetric_bilinear'])
add('Symmetric midpoint + cubic interpolation','symmetric',reg['symmetric_cubic'])
for k,title in [('surface_strip','Surface-following strips'),('surface_strip_projected','Strips + projected photometry'),('surface_strip_glare','Strips + glare masking'),('surface_strip_curvature_prior','Strips + curvature prior')]:
 add(title,'boundary',boundary[k],near='lt40')
for v in shape['validation']:
 if v['name'] in ['cubic16','cubic64']:add('Cubic deformation, ridge '+v['name'][5:],'shape_order',v['metrics'],notes='True third-order spatial warp; bilinear image interpolation.')
for v in glob['reports']:
 if v['name']!='baseline':add(v['name'],'global_consensus',v['metrics'],notes='Inspired consensus prototype, not published wOFV implementation.')
combined=Path('work/lit_cubicshape/combined_results.json')
if combined.exists():
 # The separate combined experiment is added by the final integration script
 # after its schema and reference metrics have been independently checked.
 pass
with (O/'optical_flow_method_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
(O/'optical_flow_method_comparison.json').write_text(json.dumps({'units':'pixel displacement endpoint disagreement per image pair','manual_targets_used_in_fitting':False,'independent_test_set':False,'rows':rows},indent=2))

chosen=[rows[0],rows[2],rows[10],rows[6],rows[8],rows[5],rows[12]]
labels=['Original quadratic','Cubic image interpolation','Cubic spatial deformation','Surface-following windows',
        'Windows + glare masking','Symmetric deformation (cubic)','Weakest global coupling']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(12,6.1),sharey=True)
fig.subplots_adjust(left=.28,right=.96,bottom=.21,top=.77,wspace=.2)
ink='#17263b';muted='#536170';blue='#168faa'
fig.text(.065,.94,'Which literature leads improved this image pair?',fontsize=18,weight='bold',color=ink)
fig.text(.065,.887,'Same manual positions for every method · lower endpoint disagreement is better',fontsize=11,color=muted)
ys=np.arange(len(chosen))
for a,key,title in zip(ax,['mean_all_px','mean_near_px'],['All 200 manual picks','26 picks within 40 px of surface']):
 baseline=chosen[0][key]
 for y,r in zip(ys,chosen):
  col=blue if y==1 else ('#3d526b' if y==0 else '#8a96a3')
  a.plot([0,r[key]],[y,y],color=col,lw=2.2,alpha=.5)
  a.scatter([r[key]],[y],s=65,color=col,zorder=3)
  a.text(r[key]+.014,y,'{:.3f}'.format(r[key]),va='center',fontsize=10,color=col)
 a.axvline(baseline,color='#526278',ls='--',lw=1,alpha=.55)
 a.set_title(title,fontsize=12,loc='left',pad=15,weight='bold',color=ink)
 a.set_xlabel('Mean disagreement (pixels)')
 a.set_xlim(0,.77 if key=='mean_all_px' else 1.14)
 a.set_yticks(ys);a.set_yticklabels(labels);a.grid(axis='x',alpha=.14)
 ax[0].set_ylim(len(chosen)-.5,-.5)
fig.text(.065,.10,'Cubic interpolation improves mean error by 3.8% overall and 4.4% near the surface; it retains quadratic deformation.',fontsize=10.5,color=ink)
fig.text(.065,.05,'These are exploratory comparisons on one previously examined pair. No improved surface-gradient accuracy is established.',fontsize=10,color=muted)
for ext in ['png','svg']:fig.savefig(O/('optical_flow_improvement_summary.'+ext),dpi=220,facecolor='white')
plt.close(fig)
print('Saved comparison table with',len(rows),'fixed continuous-field configurations and summary figure.')
