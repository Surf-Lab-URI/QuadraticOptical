"""Display the already accepted deeper-water vectors at a visible uniform gain."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from export_pairs import read, ROOT, CYAN, ORANGE, INK

for pair in (80,100):
    out=ROOT/'outputs'/('pair_'+str(pair));z=read(ROOT/'work/pairs'/str(pair)/'results.npz')
    q=z['query_full'];d=z['disp'];c=z['classical_displacement_native_at_grid']
    ok=z['accepted'].astype(bool);cv=z['classical_available_at_grid'].astype(bool)
    ix=np.rint((q[:,0]-7)/8).astype(int);iy=np.rint((q[:,1]-7)/8).astype(int)
    sampled=(ix%4==0)&(iy%4==0)&(q[:,1]>=500)
    gain=40.;ends=np.r_[q[sampled&ok]+gain*d[sampled&ok],q[sampled&cv]+gain*c[sampled&cv]]
    limits=(min(0,float(ends[:,0].min())-8),max(2047,float(ends[:,0].max())+8),
            max(2047,float(ends[:,1].max())+8),min(500,float(ends[:,1].min())-8))
    fig,ax=plt.subplots(figsize=(11.5,10.5));fig.subplots_adjust(top=.84,bottom=.17,left=.09,right=.96)
    fig.suptitle('Pair {} | deeper-water velocity detail'.format(pair),x=.09,y=.98,ha='left',fontsize=20,fontweight='bold',color=INK)
    fig.text(.09,.927,'Same accepted field; sources at image y ≥ 500. Arrows enlarged 40×, sampled every 32 pixels.',fontsize=10.5)
    ax.set_facecolor('#14232d')
    def quiv(mask,v,color,width,zorder):
        return ax.quiver(q[mask,0],q[mask,1],v[mask,0],v[mask,1],angles='xy',scale_units='xy',scale=1/gain,
                         color=color,width=width,headwidth=3.6,headlength=4.5,headaxislength=4.1,minlength=.05,zorder=zorder)
    a=quiv(sampled&ok,d,CYAN,.0016,3);quiv(sampled&cv,c,ORANGE,.00075,4)
    ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='Image x (pixels)',ylabel='Image y (pixels, downward)');ax.set_aspect('equal')
    ax.quiverkey(a,.77,1.045,1,'1 pixel / pair',coordinates='axes',labelpos='E',fontproperties={'size':10},labelcolor=INK)
    handles=[Line2D([],[],color=CYAN,lw=3,label='Conservative hybrid'),Line2D([],[],color=ORANGE,lw=1.5,label='Supplied native PIV')]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.087))
    fig.text(.09,.065,'Arrow lengths use the same 40× display gain for both methods. This view supplements the full-image and near-surface plots.',fontsize=9.2)
    fig.text(.09,.027,'Only the display scale changes. All displacements and acceptance masks are identical to the exported full-image results.',fontsize=9.2)
    for ext in ('png','svg'):fig.savefig(out/('hybrid_piv_deep_water.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    (out/'deep_quiver_display.json').write_text(json.dumps(dict(pair=pair,gain=gain,source_y_min=500,sampling_pixels=32,
        hybrid_arrows=int((sampled&ok).sum()),classical_arrows=int((sampled&cv).sum()),display_limits=limits,
        numerical_values_changed=False),indent=2))
    print('Wrote deeper-water view for',pair,flush=True)
