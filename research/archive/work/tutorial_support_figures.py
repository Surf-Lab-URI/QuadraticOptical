import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT=Path('outputs/hybrid_tutorial_figures');OUT.mkdir(exist_ok=True)
NAVY='#173A50';TEAL='#087C83';ORANGE='#D98432';PURPLE='#9A59B5'
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'font.family':'DejaVu Sans','pdf.fonttype':42})
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(7.1,3.2),gridspec_kw={'width_ratios':[1,1.45]})
ax=axs[0]
for x,y,val in [(0,0,20),(1,0,60),(0,1,40),(1,1,80)]:
 ax.scatter(x,y,s=90,c=NAVY);ax.text(x,y+(-.14 if y==0 else .19),str(val),ha='center',color=NAVY,fontweight='bold')
ax.plot([0,1,1,0,0],[0,0,1,1,0],color='#a7adb4')
ax.scatter(.25,.6,s=85,color=TEAL,zorder=4);ax.text(.30,.56,'Sample here\n(0.25, 0.60)',ha='left',fontsize=9,color=TEAL)
ax.set(xlim=(-.25,1.25),ylim=(1.4,-.35),xticks=[0,1],yticks=[0,1],xlabel='x (pixels)',ylabel='y (pixels)',title='Four neighboring pixels')
ax.set_aspect('equal');ax.spines[['top','right']].set_visible(False) if False else None
for s in ['top','right']:ax.spines[s].set_visible(False)
ax=axs[1];ax.axis('off');ax.set_title('A weighted average gives subpixel intensity')
rows=[['20','0.75 × 0.40 = 0.30','6'],['60','0.25 × 0.40 = 0.10','6'],['40','0.75 × 0.60 = 0.45','18'],['80','0.25 × 0.60 = 0.15','12']]
tbl=ax.table(cellText=rows,colLabels=['Intensity','Weight','Product'],cellLoc='center',loc='center',colWidths=[.23,.48,.29]);tbl.auto_set_font_size(False);tbl.set_fontsize(9);tbl.scale(1,1.6)
for (r,c),cell in tbl.get_celld().items():
 cell.set_edgecolor('#d4d9df')
 if r==0:cell.set_facecolor('#eaf1f3')
ax.text(.5,.10,'Interpolated intensity = 6 + 6 + 18 + 12 = 42',ha='center',va='center',fontsize=9,color=TEAL,fontweight='bold',transform=ax.transAxes)
fig.subplots_adjust(wspace=.4,bottom=.18,top=.88);save(fig,'bilinear_sampling')

e=np.linspace(-4,4,600);delta=.8;rho=delta**2*(np.sqrt(1+(e/delta)**2)-1);w=1/np.sqrt(1+(e/delta)**2)
fig,axs=plt.subplots(1,2,figsize=(7.1,3.15))
axs[0].plot(e,.5*e**2,color=ORANGE,label='Squared error');axs[0].plot(e,rho,color=TEAL,label='Implemented robust loss',lw=2)
axs[0].set(xlabel='Intensity residual e',ylabel='Penalty',title='Large mismatches have less influence');axs[0].legend(fontsize=8.5,frameon=False)
axs[1].plot(e,w,color=TEAL,lw=2);axs[1].axhline(1,color='#a8adb4',ls=':',lw=1);axs[1].set(xlabel='Intensity residual e',ylabel='Relative residual weight',ylim=(0,1.08),title='Downweighted does not mean deleted')
for ax in axs:
 ax.grid(alpha=.16)
 for s in ['top','right']:ax.spines[s].set_visible(False)
fig.tight_layout();save(fig,'robust_loss')

x=np.linspace(0,16,500);centers=np.array([0.,16.]);delta_x=x[:,None]-centers[None];t=np.abs(delta_x)/16
w=np.maximum(1-t,0)**4*(1+4*t);dw=-20/16**2*(1-t)**3*delta_x
values=np.array([4.,5.]);d=(w*values).sum(1)/w.sum(1);g=(dw*(values[None]-d[:,None])).sum(1)/w.sum(1)
fig,axs=plt.subplots(3,1,figsize=(7.1,5.0),sharex=True)
axs[0].plot(x,w[:,0]/w.sum(1),color=TEAL,label='Left patch share');axs[0].plot(x,w[:,1]/w.sum(1),color=PURPLE,label='Right patch share');axs[0].set(ylabel='Weight share');axs[0].legend(frameon=False,ncol=2,fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.01))
axs[1].axhline(4,color=TEAL,ls='--',label='Left patch: 4 pixels');axs[1].axhline(5,color=PURPLE,ls='--',label='Right patch: 5 pixels');axs[1].plot(x,d,color=NAVY,lw=2,label='Combined displacement');axs[1].set(ylabel='Displacement\n(pixels)');axs[1].legend(frameon=False,ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.30))
axs[2].axhline(0,color=ORANGE,ls='--',label='Each separate patch slope');axs[2].plot(x,g,color=NAVY,lw=2,label='Slope of the combined field');axs[2].set(xlabel='Horizontal location x (pixels)',ylabel='Slope\n(pixels per pixel)');axs[2].legend(frameon=False,ncol=2,fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.01))
for ax in axs:
 ax.grid(alpha=.16);ax.set_xlim(0,16)
 for s in ['top','right']:ax.spines[s].set_visible(False)
fig.suptitle('Two overlapping patches: a constructed example',y=.99,fontsize=11)
fig.subplots_adjust(hspace=.72,bottom=.12,top=.87,left=.15,right=.98);save(fig,'field_blending')
print('Created bilinear_sampling, robust_loss, field_blending in PNG and vector PDF.')
