"""Display-only quiver revision, using existing immutable scientific results."""
import export_results as e
import numpy as np
import matplotlib.pyplot as plt
r={};raw={};s={}
for pair in [80,100]:
    r[pair]=e.read(e.HERE/str(pair)/'results.npz')
    with np.load(e.HERE/str(pair)/'inputs.npz') as z:raw[pair]=z['rawA'];s[pair]=z['surface_a']
    fig,ax=plt.subplots(figsize=(16,3.7))
    e.quiver(ax,r[pair],raw[pair],s[pair],pair)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.53),ncol=3,frameon=False,fontsize=8)
    fig.text(.12,.03,'Arrows are thinned for clarity; estimates failing conservative screens are omitted. Units assume DX in metres/pixel and DT in seconds.',fontsize=9)
    fig.subplots_adjust(bottom=.30,top=.76);e.save(fig,e.OUT/('pair_'+str(pair))/'quiver')
fig,axes=plt.subplots(2,1,figsize=(16,4.6))
for pair,ax in zip([80,100],axes):e.quiver(ax,r[pair],raw[pair],s[pair],pair)
axes[0].set_xlabel('')
fig.text(.12,.050,'Image-only estimates. Cyan = inferred local surface; orange = 1 cm below it. Both panels use the same arrow scale.',fontsize=8)
fig.text(.12,.023,'Physical units assume DX in metres/pixel and DT in seconds. Height is relative to the median surface ordinate in each image.',fontsize=8)
fig.subplots_adjust(bottom=.16,top=.87,hspace=.65);e.save(fig,e.OUT/'quiver_comparison')
fig,axes=plt.subplots(2,1,figsize=(16,3.4))
for pair,ax in zip([80,100],axes):e.quiver(ax,r[pair],raw[pair],s[pair],pair,gain=20,min_depth_cm=.5)
axes[0].set_xlabel('')
fig.text(.12,.035,'Supplementary view of the same estimates: larger arrow gain reveals slower motion. Cyan = 0.5 cm local depth; orange = 1 cm. Physical units are assumed.',fontsize=9)
fig.subplots_adjust(bottom=.18,top=.83,hspace=.65);e.save(fig,e.OUT/'quiver_deeper_half')
print('Quiver display revisions complete')
