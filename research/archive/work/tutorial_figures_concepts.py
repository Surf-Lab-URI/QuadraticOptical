import os, json
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.path import Path as MplPath
from pathlib import Path

OUT=Path('outputs/hybrid_tutorial_figures')
OUT.mkdir(parents=True,exist_ok=True)
NAVY='#173A50'; TEAL='#087C83'; ORANGE='#D98432'; MAGENTA='#9A59B5'; GRAY='#82919A'; PALE='#E7EDF0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'text.color':NAVY,'axes.labelcolor':NAVY,'axes.edgecolor':GRAY,'xtick.color':NAVY,'ytick.color':NAVY,'pdf.fonttype':42,'ps.fonttype':42})
manifest={}
def save(fig,name,description,equations):
    fig.savefig(OUT/(name+'.png'),dpi=220,facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    plt.close(fig)
    manifest[name]={'png':str(OUT/(name+'.png')),'pdf':str(OUT/(name+'.pdf')),'description':description,'equations':equations,'status':'Synthetic teaching illustration; not experimental measurements.'}
def panel(ax,letter,title):
    ax.set_title(letter+'  '+title,loc='left',pad=8,fontweight='bold')
def arrow(ax,a,b,color=TEAL,lw=2,style='-|>',**kw):
    p=FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=12,linewidth=lw,color=color,**kw); ax.add_patch(p); return p

# 1. Coordinate basics and pixel displacement.
fig=plt.figure(figsize=(7.1,5.3)); gs=gridspec.GridSpec(2,2,height_ratios=[1,.66],hspace=.49,wspace=.29)
Y,X=np.mgrid[0:48,0:48]
pts=np.array([[20,30],[9,17],[33,18],[29,38],[12,39],[37,8]],float)
for j,(delta,name,c) in enumerate([(np.array([0.,0.]),'Frame A: earlier',ORANGE),(np.array([6.,-2.]),'Frame B: later',TEAL)]):
    ax=fig.add_subplot(gs[0,j]); im=np.full(X.shape,10.,float)
    for k,p in enumerate(pts+delta): im+= (185 if k else 230)*np.exp(-((X-p[0])**2+(Y-p[1])**2)/(2*1.15**2))
    ax.imshow(im,cmap='gray',vmin=0,vmax=255,origin='upper',extent=(-.5,47.5,47.5,-.5));
    q=pts[0]+delta; ax.add_patch(Circle(q,3.5,fill=False,lw=1.4,color=c));
    ax.annotate('(%d, %d)'%tuple(q),xy=q,xytext=(3,7),color='white',fontsize=10,arrowprops={'arrowstyle':'-','color':c,'lw':1.3})
    ax.set_xticks([0,20,40]);ax.set_yticks([0,20,40]);ax.set_xlabel('Horizontal pixel coordinate x →');ax.set_ylabel('y increases downward');panel(ax,chr(97+j),name)
ax=fig.add_subplot(gs[1,0]);ax.set_xlim(17,30);ax.set_ylim(33,25);ax.set_xticks([20,23,26]);ax.set_yticks([28,30]);ax.grid(color=PALE)
ax.plot(20,30,'o',color=ORANGE,ms=6);ax.plot(26,28,'o',color=TEAL,ms=6);arrow(ax,(20,30),(26,28))
ax.plot([20,26,26],[30,30,28],color=GRAY,ls='--',lw=1)
ax.text(23,31.2,'6 pixels right',ha='center',fontsize=9);ax.text(27,29.2,'2 pixels\nup',va='center',fontsize=9)
ax.set_xlabel('x (pixels)');ax.set_ylabel('y (pixels)');panel(ax,'c','Subtract the two positions')
ax=fig.add_subplot(gs[1,1]);ax.axis('off')
ax.text(0,1.05,'Convert displacement to velocity',va='top',fontweight='bold',fontsize=9)
ax.text(0,.76,r'$\mathbf{d}=(6,-2)$ pixels per pair',fontsize=11)
ax.text(0,.46,r'$u=6a/\Delta t,\qquad v_{\rm up}=2a/\Delta t$',fontsize=11)
ax.text(0,.15,'a = physical length per pixel\nΔt = time between frames\nu = rightward velocity\nvᵤₚ = upward velocity',fontsize=9,linespacing=1.45,va='center')
fig.text(.5,.987,'Synthetic image pair and coordinates',ha='center',va='top',fontsize=9)
fig.subplots_adjust(left=.08,right=.98,top=.92,bottom=.10)
save(fig,'basics_displacement','Synthetic images illustrate one source particle at A=(20,30) and its target at B=(26,28), in pixel coordinates increasing right and down. The displacement is (6,-2) pixels per pair. Conversion to velocity requires a spatial scale and the time interval; upward physical velocity reverses the y sign.',[r'd=(26-20,28-30)=(6,-2) pixels/pair',r'u=a d_x/Delta t; v_up=-a d_y/Delta t'])

# 2. Forward coordinate maps deform an entire neighborhood.
fig,axs=plt.subplots(1,3,figsize=(7.1,3.4)); rng=np.random.RandomState(18); p=rng.uniform(-.85,.85,(13,2))
t=np.array([.22,-.10]); M=np.array([[.98,-.28],[.16,.78]])
def trans(q,kind):
    q=np.asarray(q); out=q+t if kind==0 else q.dot(M.T)+t
    if kind==2:
        xx=q[...,0];yy=q[...,1];out=out+np.stack([.22*yy**2+.12*xx*yy,.20*xx**2-.08*xx*yy],axis=-1)
    return out
ss=np.linspace(-1,1,101)
for k,ax in enumerate(axs):
    color=[TEAL,ORANGE,MAGENTA][k]
    for v in np.linspace(-1,1,7):
        for q in [np.stack([ss,ss*0+v],axis=1),np.stack([ss*0+v,ss],axis=1)]:
            ax.plot(q[:,0],q[:,1],color=PALE,lw=.8,zorder=0)
            z=trans(q,k);ax.plot(z[:,0],z[:,1],color=color,lw=.7,alpha=.68,zorder=1)
    theta=np.linspace(0,2*np.pi,70)
    for xy in p:
        ax.add_patch(Circle(xy,.048,fill=False,edgecolor=GRAY,lw=.6,zorder=2))
        ring=xy+.05*np.stack([np.cos(theta),np.sin(theta)],axis=1); z=trans(ring,k)
        ax.fill(z[:,0],z[:,1],color=color,zorder=3)
    ax.set_aspect('equal');ax.set_xlim(-1.2,1.62);ax.set_ylim(1.27,-1.27);ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(False)
    panel(ax,chr(97+k),['Translation','Affine deformation','Quadratic deformation'][k])
    ax.text(.5,-.10,['All points move equally.\nThe grid keeps its shape.','Displacement varies linearly.\nStraight grid lines stay straight.','The rate of change can vary.\nGrid lines can bend.'][k],transform=ax.transAxes,ha='center',va='top',fontsize=9,linespacing=1.4)
fig.text(.05,.94,'Gray: original neighborhood. Color: mapped neighborhood. Synthetic example.',fontsize=9)
fig.subplots_adjust(left=.025,right=.99,top=.77,bottom=.26,wspace=.11)
save(fig,'deformation_models','Synthetic map of the same grid and particle-like blobs. The translation uses t=(0.22,-0.10). The affine map has matrix [[0.98,-0.28],[0.16,0.78]], combining linear deformations. The quadratic map adds (0.22 y²+0.12 xy, 0.20 x²-0.08 xy) to the affine map. Curved grid lines demonstrate spatially varying deformation. No individual particle spin is inferred.',[r'T_translation(q)=q+t',r'T_affine(q)=M q+t',r'T_quadratic(q)=M q+t+(0.22 y^2+0.12xy,0.20x^2-0.08xy)'])

# 3. Pattern ambiguity and the penalized tracking objective.
fig=plt.figure(figsize=(7.1,5.0));gs=gridspec.GridSpec(2,2,height_ratios=[.50,1],hspace=.70,wspace=.36)
xx=np.linspace(0,80,600); yy=np.linspace(-3,3,42);XX,YY=np.meshgrid(xx,yy)
for k in range(2):
    ax=fig.add_subplot(gs[0,k]);
    positions=[8,12,15,40,48,66,73] if k==0 else [8,12,15,34,38,41,60,64,67]
    im=np.zeros_like(XX)
    for n,b in enumerate(positions):im+=np.exp(-((XX-b)**2/(2*.70**2)+(YY-((n%3)-1)*.9)**2/(2*.5**2)))
    ax.imshow(im,cmap='gray',aspect='auto',extent=(0,80,3,-3));
    for q,c in [(6,TEAL)]+([(32,ORANGE)] if k else []):ax.add_patch(Rectangle((q,-2.6),12,5.2,fill=False,color=c,lw=1.5))
    ax.set_xticks([]);ax.set_yticks([]);panel(ax,chr(97+k),['Distinctive group','Repeated groups'][k])
    ax.text(.0,-.28,['One neighborhood has a distinctive pattern.','Similar patterns offer competing matches.'][k],transform=ax.transAxes,fontsize=9,va='top')
    ax=fig.add_subplot(gs[1,k]);d=np.linspace(-2,16,700);prior=6
    if k==0: corr=.15+.78*np.exp(-.5*((d-6)/1.5)**2)+.35*np.exp(-.5*((d-11)/.95)**2)
    else:corr=.14+.78*np.exp(-.5*((d-4)/.85)**2)+.78*np.exp(-.5*((d-8)/.85)**2)
    cost=1-corr+.005*(d-prior)**2
    ax.plot(d,cost,color=TEAL,lw=2)
    if k==0:
        bestx=d[np.argmin(cost)];altwindow=(d>9)&(d<13);altx=d[altwindow][np.argmin(cost[altwindow])];y1=np.interp(bestx,d,cost);y2=np.interp(altx,d,cost)
    else:bestx=4.;altx=8.;y1=np.interp(bestx,d,cost);y2=np.interp(altx,d,cost)
    ax.plot(bestx,y1,'o',color=TEAL,ms=5);ax.plot(altx,y2,'o',color=ORANGE,ms=5)
    ax.axvline(prior,color=GRAY,ls=':',lw=1)
    if k==0:
        ax.hlines([y1,y2],6.0,14,color=GRAY,ls='--',lw=.8);arrow(ax,(13,y1),(13,y2),color=ORANGE,lw=1.5,style='<->');ax.text(12.3,(y1+y2)/2,'Large\ngap',ha='right',va='center',fontsize=9)
    else:ax.annotate('Nearly tied alternatives',xy=(8,y2),xytext=(8,0.65),ha='center',fontsize=9,arrowprops={'arrowstyle':'->','color':ORANGE},color=NAVY)
    ax.set_xlim(-2,16);ax.set_ylim(0,1.50);ax.set_xlabel('Trial horizontal displacement (pixels)');ax.set_ylabel('Illustrative cost J (smaller is better)');ax.set_xticks([0,4,8,12,16]);ax.set_yticks([0,.5,1]);ax.grid(axis='y',color=PALE)
fig.text(.5,.065,r'$J=1-\mathrm{NCC}+0.005\,\|\mathbf{d}-\mathbf{d}_0\|^2$',ha='center',fontsize=11)
fig.text(.5,.014,'NCC = normalized cross-correlation; dotted line = starting prediction d₀. Synthetic illustration.',ha='center',fontsize=9)
fig.subplots_adjust(left=.095,right=.97,top=.93,bottom=.22)
save(fig,'correlation_ambiguity','Synthetic bright-pattern strips and illustrative trial-displacement cost curves explain why a distinctive pattern is easier to match than repeated particle groups. The tracker compares minima of its penalized objective, not simply peak raw correlation. The second cost curve has two almost tied alternatives. Curves illustrate the mathematical cost form but are not measured costs from these experiments and the image strips are not input to the plotted curves.',[r'J_track(d)=1-NCC(d)+0.005 ||d-d0||^2',r'gap=J_alternative-J_best'])

# 4. Reverse composition and perturbation agreement.
fig=plt.figure(figsize=(7.1,6.0)); gs=gridspec.GridSpec(2,2,height_ratios=[.84,1],hspace=.42,wspace=.29)
ax=fig.add_subplot(gs[0,:]);ax.set_xlim(-.2,8.8);ax.set_ylim(-1.8,2.5);ax.axis('off');panel(ax,'a','Forward, then reverse from the predicted endpoint')
a=np.array([1.,.35]); b=np.array([7.,.35]); back=np.array([1.55,-.25]);
ax.plot(*a,'o',color=ORANGE,ms=7);ax.plot(*b,'o',color=TEAL,ms=7);ax.plot(*back,'o',color=MAGENTA,ms=6)
ax.add_patch(FancyArrowPatch(path=MplPath([a,(4,2.2),b],[MplPath.MOVETO,MplPath.CURVE3,MplPath.CURVE3]),arrowstyle='-|>',mutation_scale=12,lw=2,color=TEAL))
ax.add_patch(FancyArrowPatch(path=MplPath([b,(4,-1.0),back],[MplPath.MOVETO,MplPath.CURVE3,MplPath.CURVE3]),arrowstyle='-|>',mutation_scale=12,lw=2,color=MAGENTA))
ax.text(4,1.66,r'Forward displacement $\mathbf{d}_F(\mathbf{x})$',ha='center',fontsize=10,color=TEAL)
ax.text(4,-1.06,r'Reverse displacement $\mathbf{d}_R(\mathbf{x}+\mathbf{d}_F(\mathbf{x}))$',ha='center',fontsize=10,color=MAGENTA)
ax.text(.98,.68,'A: start x',ha='center',fontsize=9)
ax.text(7.10,1.0,'B: predicted endpoint',ha='center',fontsize=9)
arrow(ax,a,back,ORANGE,style='<->',lw=1.5)
ax.text(.1,-.55,'Closure\nerror',fontsize=9,color=ORANGE,ha='left')
ax.text(4,-1.68,r'$e_{\rm FB}=\|\mathbf{d}_F(\mathbf{x})+\mathbf{d}_R(\mathbf{x}+\mathbf{d}_F(\mathbf{x}))\|$',ha='center',fontsize=11)
ax=fig.add_subplot(gs[1,0]);ax.set_xlim(-24,24);ax.set_ylim(24,-24);ax.set_aspect('equal');panel(ax,'b','Change the window or mask')
sx=np.linspace(-24,24,150);sy=-18+3*np.exp(-sx**2/120);ax.fill_between(sx,-24,sy+10,color=MAGENTA,alpha=.17);ax.plot(sx,sy,color=NAVY,lw=1,label='Surface')
ax.plot(sx,sy+10,color=MAGENTA,lw=1.5);ax.plot(sx,sy+14,color=MAGENTA,lw=1.5,ls='--');
ax.add_patch(Rectangle((-13.5,-13.5),27,27,fill=False,edgecolor=TEAL,lw=1.7));ax.add_patch(Rectangle((-19.5,-19.5),39,39,fill=False,edgecolor=ORANGE,lw=1.5,ls='--'));ax.plot(0,0,'o',color=NAVY,ms=4)
ax.set_xticks([]);ax.set_yticks([])
ax.text(0,23,'27 × 27 and 39 × 39 windows',ha='center',fontsize=9)
ax.text(-23,-22,'Synthetic surface and masks',fontsize=9)
ax.text(.5,-.07,'Solid mask: 10-pixel margin\nDashed mask: 14-pixel margin',transform=ax.transAxes,ha='center',va='top',fontsize=9,linespacing=1.35)
ax=fig.add_subplot(gs[1,1]);ax.set_xlim(-.5,6.8);ax.set_ylim(3.5,-2.0);ax.set_aspect('equal');panel(ax,'c','Compare the predictions')
ax.plot(0,0,'o',color=NAVY,ms=5)
ends=[(4.9,-.45),(4.5,-.12),(5.25,-.18),(4.85,-.8)]; colors=[TEAL,ORANGE,MAGENTA,GRAY];labels=['Primary quadratic','Affine map','Larger window','Larger mask margin']
for end,c,l in zip(ends,colors,labels):arrow(ax,(0,0),end,c,lw=1.5);ax.plot(*end,'o',color=c,ms=4,label=l)
# No numeric distance circle: endpoints are a schematic comparison.
ax.text(0,1.0,'All three comparisons must be\nwithin 1.5 pixels of the primary.',fontsize=9,linespacing=1.4)
ax.legend(loc='lower left',bbox_to_anchor=(-.03,-.06),frameon=False,handlelength=.8,labelspacing=.25)
ax.set_xticks([]);ax.set_yticks([])
for sp in ax.spines.values():sp.set_visible(False)
fig.text(.5,.014,'Synthetic illustration. Agreement does not prove a physically correct match.',ha='center',fontsize=9)
fig.subplots_adjust(left=.055,right=.98,top=.93,bottom=.15)
save(fig,'consistency_checks','Synthetic diagram of composed forward-backward matching. The reverse field is evaluated at the predicted B endpoint, not at the original A coordinate. The lower panels illustrate changing nominal 27×27 to 39×39 windows, increasing the geometric exclusion margin from 10 to 14 pixels, and comparing affine versus quadratic deformation. Actual windows are clipped by masks and image limits; actual image availability can deepen the effective mask. All three comparison displacements must stay within 1.5 pixels of the primary prediction.',[r'e_FB=||d_F(x)+d_R(x+d_F(x))||',r'max_m ||d_m(x)-d_primary(x)|| <= 1.5 pixels'])

# 5. A velocity is not its derivative; paths matter.
fig=plt.figure(figsize=(7.1,5.7));gs=gridspec.GridSpec(2,2,height_ratios=[.75,1],hspace=.60,wspace=.35)
q=np.array([2,6,10,14,18]); center=10
for k in range(2):
    ax=fig.add_subplot(gs[0,k]);u=5+(.3*(q-center) if k else 0*q)
    for y in [0,1,2]:ax.quiver(q,np.full(len(q),y),u,np.zeros(len(q)),angles='xy',scale_units='xy',scale=1.5,color=[TEAL,ORANGE][k],width=.011,headwidth=3.5)
    ax.axvline(center,color=GRAY,ls='--',lw=1);ax.set_xlim(0,26);ax.set_ylim(2.6,-.6);ax.set_xticks([0,10,20]);ax.set_yticks([]);ax.set_xlabel('Horizontal position x (mm)');panel(ax,chr(97+k),['Uniform velocity','Increasing velocity'][k]);
    ax.text(.5,-.43,[r'$u=5\ \mathrm{mm/s};\quad \partial_x u=0$',r'$u(10)=5\ \mathrm{mm/s};\quad \partial_x u=0.3\ \mathrm{s}^{-1}$'][k],transform=ax.transAxes,ha='center',fontsize=10)
ax=fig.add_subplot(gs[1,0]);x=np.linspace(0,20,100);ax.plot(x,0*x+5,color=TEAL,lw=2,label='Uniform field');ax.plot(x,5+.3*(x-10),color=ORANGE,lw=2,label='Increasing field');ax.plot(10,5,'o',color=NAVY,ms=5);ax.annotate('Same velocity here',xy=(10,5),xytext=(.5,8),fontsize=9,arrowprops={'arrowstyle':'->','color':GRAY});ax.set_xlabel('x (mm), at a fixed height');ax.set_ylabel('Horizontal velocity u (mm/s)');ax.set_xlim(0,20);ax.set_ylim(1,9);ax.set_xticks([0,10,20]);ax.grid(color=PALE);ax.legend(loc='lower right',frameon=False);panel(ax,'c','The gradient is the slope')
ax=fig.add_subplot(gs[1,1]);x=np.linspace(0,20,200);s=2+3*np.exp(-((x-11)/4.3)**2);h=3.;ax.plot(x,s,color=NAVY,lw=1.7,label='Curved surface');ax.fill_between(x,-1,s,color=PALE);ax.plot(x,s+h,color=MAGENTA,lw=2,label='Constant vertical depth');ax.axhline(8,color=TEAL,lw=2,ls='--',label='Fixed image height');ax.set_xlim(0,20);ax.set_ylim(12,-1);ax.set_yticks([0,5,10]);ax.set_xticks([0,10,20]);ax.set_xlabel('x (schematic coordinates)');ax.set_ylabel('y increases downward');panel(ax,'d','Choose a direction');ax.legend(loc='lower left',frameon=False,fontsize=9,handlelength=1.2,labelspacing=.25)
fig.text(.5,.017,r'Along $y=s(x)+h$: $\frac{d}{dx}u(x,s(x)+h)=\partial_xu+s^{\prime}(x)\,\partial_yu$.',ha='center',fontsize=10)
fig.text(.5,.987,'Synthetic velocities and surface geometry',ha='center',va='top',fontsize=9)
fig.subplots_adjust(left=.09,right=.98,top=.91,bottom=.16)
save(fig,'velocity_gradient','Synthetic velocity fields with zero vertical velocity have the same speed of 5 mm/s at x=10 mm, but different horizontal slopes: 0 and 0.3 s^-1. The quiver arrow lengths share a common display scale; velocities are illustrative and do not use experimental calibration. The right lower panel distinguishes differentiation at fixed image y from differentiation along a curve at a fixed vertical depth below the surface. The latter adds s-prime times the vertical derivative.',[r'u_1(x)=5 mm/s',r'u_2(x)=5 mm/s+(0.3 s^-1)(x-10 mm)',r'd/dx u(x,s(x)+h)=partial_x u+s_prime(x) partial_y u'])
(OUT/'concept_figure_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v['png'] for k,v in manifest.items()},indent=2))
