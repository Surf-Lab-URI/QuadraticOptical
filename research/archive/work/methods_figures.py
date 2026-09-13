import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'methods_figures';OUT.mkdir(exist_ok=True)
z=np.load(ROOT/'work/full_width/inputs.npz');z={k:z[k] for k in z.files}
t=np.load(ROOT/'work/full_width/ptv_tracks.npz');t={k:t[k] for k in t.files}
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.titlesize':11})
lo,hi=365,715;top,bottom=340,505
fig,axs=plt.subplots(2,1,figsize=(10,9.2))
for ax,fr in zip(axs,'AB'):
    raw=z['raw'+fr];sf=z['full_surface_'+fr.lower()];v=z['v'+fr.lower()]
    ax.imshow(raw,cmap='gray',vmin=0,vmax=210,extent=(-.5,2047.5,839.5,259.5),interpolation='nearest')
    xx=np.arange(2048);first=v.argmax(axis=0)+260
    ax.fill_between(xx,top,sf,color='#98a3b3',alpha=.22)
    ax.fill_between(xx,sf,first,color='#b672d9',alpha=.38)
    ax.plot(xx,sf,color='#ffcf49',lw=1.5)
    ax.plot(xx,first,color='#e6a1ff',lw=1.1)
    ax.plot(xx,sf+14,color='#80efbd',lw=1,ls='--')
    ax.set(xlim=(lo,hi),ylim=(bottom,top),ylabel='Full-image y (px)',title='Frame '+fr+': supplied geometry and actual fitting boundary')
    ax.set_aspect('equal')
axs[-1].set_xlabel('Full-image x (px)')
handles=[Line2D([],[],color='#ffcf49',label='Geometric surface trace'),Line2D([],[],color='#e6a1ff',label='First valid fitting row'),Line2D([],[],color='#80efbd',ls='--',label='14 px comparison margin'),Patch(facecolor='#b672d9',alpha=.5,label='Excluded surface / glare band')]
fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.005))
fig.subplots_adjust(top=.97,bottom=.11,hspace=.24,left=.08,right=.98)
fig.savefig(OUT/'surface_masks.png',dpi=220);plt.close(fig)

fig,axs=plt.subplots(2,1,figsize=(10,9.2))
raw=z['rawA'];dog=gaussian_filter(raw,.6)-gaussian_filter(raw,2)
p=t['points']+z['origin0'];sel=(p[:,0]>=lo)&(p[:,0]<=hi)&(p[:,1]>=top)&(p[:,1]<=bottom)
for ax in axs:
    ax.set(xlim=(lo,hi),ylim=(bottom,top),ylabel='Full-image y (px)')
    ax.set_aspect('equal')
axs[0].imshow(dog,cmap='gray',vmin=-10,vmax=50,extent=(-.5,2047.5,839.5,259.5),interpolation='nearest')
axs[0].plot(p[sel,0],p[sel,1],'o',mfc='none',mec='#ffcf49',ms=4,mew=.7)
axs[0].set_title('Particle-like peaks in the difference-of-Gaussians response')
axs[1].imshow(raw,cmap='gray',vmin=0,vmax=210,extent=(-.5,2047.5,839.5,259.5),interpolation='nearest')
ok=sel&t['accepted'];bad=sel&~t['accepted'];d=t['disp']
axs[1].quiver(p[ok,0],p[ok,1],d[ok,0],d[ok,1],angles='xy',scale_units='xy',scale=1,color='#43d9f3',width=.0026,headwidth=3.5,headlength=4.5)
axs[1].plot(p[bad,0],p[bad,1],'x',color='#ed9950',ms=4,mew=.8)
axs[1].set_title('Accepted automatic A-to-B tracks (cyan); rejected candidates (orange crosses)')
axs[1].set_xlabel('Full-image x (px)')
for ax in axs:
    xx=np.arange(2048);sf=z['full_surface_a'];ax.plot(xx,sf,color='#ffcf49',lw=1.2)
    ax.plot(xx,sf+14,color='#80efbd',lw=1,ls='--')
fig.subplots_adjust(top=.97,bottom=.07,hspace=.22,left=.08,right=.98)
fig.savefig(OUT/'particle_detection.png',dpi=220);plt.close(fig)
print('Created two methods figures from the saved input arrays and track records.')
