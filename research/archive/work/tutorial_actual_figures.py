"""Teaching figures from frozen delivered results; no new velocity fit is performed."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']=os.path.abspath('work/mplcache')
from pathlib import Path
import json, copy
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'outputs/hybrid_tutorial_figures';O.mkdir(exist_ok=True)
def read(name):
    with np.load(ROOT/'work/full_width'/name) as f:return {k:f[k] for k in f.files}
z=read('inputs.npz'); t=read('ptv_tracks.npz'); f=read('main.npz'); results=read('results.npz')
A=z['A'];B=z['B'];va=z['va'];vb=z['vb'];raw=z['rawA'];origin=z['origin0'];xs=np.arange(2048)
INK='#16394d';CYAN='#12cbea';GOLD='#ffbe45';RED='#f27975';PURPLE='#c28ae1';GREEN='#80edbd'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
    fig.savefig(O/(name+'.png'),dpi=270,bbox_inches='tight',pad_inches=.06)
    fig.savefig(O/(name+'.pdf'),bbox_inches='tight',pad_inches=.06)
    plt.close(fig)
def imglobal(ax,img,cmap='gray',vmin=0,vmax=210):
    ax.imshow(img,cmap=cmap,vmin=vmin,vmax=vmax,extent=(-.5,2047.5,839.5,259.5),interpolation='nearest',origin='upper')
def lines(ax,shade=False):
    sf=z['full_surface_a'];first=va.argmax(axis=0)+origin[1]
    ax.plot(xs,sf,color=GOLD,lw=1.0)
    if shade:ax.fill_between(xs,sf,first,color=PURPLE,alpha=.48,lw=0)
    ax.plot(xs,sf+14,color=GREEN,lw=.85,ls='--')
def roi(ax,lo=365,hi=555,top=350,bottom=465):
    ax.set_xlim(lo,hi);ax.set_ylim(bottom,top);ax.set_aspect('equal');ax.set_xlabel('x (full-image pixels)');ax.set_ylabel('y (pixels, downward)')
# 1. Actual detector and track decisions. Saved records keep exact baseline provenance.
lo,hi,top,bottom=365,555,350,465
bg=gaussian_filter(raw,5);dog=gaussian_filter(raw,.6)-gaussian_filter(raw,2)
p=t['points']+origin
sel=(p[:,0]>=lo)&(p[:,0]<=hi)&(p[:,1]>=top)&(p[:,1]<=bottom)
ok=sel&t['accepted'];bad=sel&~t['accepted']
fig,axs=plt.subplots(2,2,figsize=(7.15,6.2));fig.subplots_adjust(left=.08,right=.98,top=.91,bottom=.245,wspace=.24,hspace=.50)
fig.suptitle('Finding particle-like image structures, then testing their motion',x=.08,y=.985,ha='left',fontsize=12,color=INK,fontweight='bold')
imglist=[raw,bg,dog,raw];titles=['(a) Raw frame A','(b) Broad background: Gaussian σ = 5 px','(c) Small bright structures: DoG response','(d) Saved A → B tracking decisions']
for ax,img,title in zip(axs.ravel(),imglist,titles):
    imglobal(ax,img,'gray',-10 if img is dog else 0,50 if img is dog else 210);roi(ax,lo,hi,top,bottom);ax.set_title(title,loc='left',pad=7)
    lines(ax,shade=img is raw)
# Highlight background threshold pixels without calling them classified reflections.
high=np.ma.array(np.ones_like(bg),mask=bg<180)
axs[0,1].imshow(high,cmap=matplotlib.colors.ListedColormap([RED]),alpha=.42,extent=(-.5,2047.5,839.5,259.5),interpolation='nearest',origin='upper')
axs[1,0].scatter(p[sel,0],p[sel,1],s=20,facecolors='none',edgecolors=GOLD,lw=.65)
axs[1,1].quiver(p[ok,0],p[ok,1],t['disp'][ok,0],t['disp'][ok,1],angles='xy',scale_units='xy',scale=1,color=CYAN,width=.0035,headwidth=3.5,headlength=4.3,minlength=.25)
axs[1,1].scatter(p[bad,0],p[bad,1],s=21,c=RED,marker='x',lw=.8)
handles=[Line2D([],[],color=GOLD,label='Supplied surface'),Line2D([],[],color=GREEN,ls='--',label='Detection depth ≥ 14 px'),Patch(facecolor=PURPLE,alpha=.5,label='Excluded fitting band'),Patch(facecolor=RED,alpha=.45,label='Background ≥ 180'),Line2D([],[],marker='o',color='none',markeredgecolor=GOLD,markerfacecolor='none',label='Candidate peak'),Line2D([],[],color=CYAN,label='Accepted automatic track'),Line2D([],[],marker='x',color=RED,ls='none',label='Rejected track')]
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.53,.015),columnspacing=1.2,handlelength=1.9)
save(fig,'actual_detection')
# 2. Reconstruct frozen local maps. Compare common valid pixels only. This is a
# coefficient-ablation illustration, not an independently optimized translation fit.
def sample_case(i):
    c=f['points'][i];r=float(f['radius']);pp=f['params'][i];rr=np.arange(-int(r),int(r)+1);dx,dy=np.meshgrid(rr,rr);shape=dx.shape
    coords=np.c_[dx.ravel()+c[0],dy.ravel()+c[1]];qx=dx.ravel()/r;qy=dy.ravel()/r;Q=np.c_[np.ones(len(qx)),qx,qy,.5*qx*qx,qx*qy,.5*qy*qy]
    aa=A[coords[:,1].astype(int),coords[:,0].astype(int)];sm=va[coords[:,1].astype(int),coords[:,0].astype(int)]
    wt=np.exp(-(qx*qx+qy*qy)/(2*.65**2));samples=[];valid=[]
    for nterms in (1,3,6):
        dest=coords+Q[:,:nterms].dot(pp[:,:nterms].T)
        bb=map_coordinates(B,[dest[:,1],dest[:,0]],order=1,mode='nearest')
        vm=sm&(map_coordinates(vb.astype(float),[dest[:,1],dest[:,0]],order=1,mode='constant',cval=0)>.99)
        vm&=(dest[:,0]>=1)&(dest[:,0]<B.shape[1]-2)&(dest[:,1]>=1)&(dest[:,1]<B.shape[0]-2)
        samples.append(bb);valid.append(vm)
    common=valid[0]&valid[1]&valid[2];weight=wt*common;stats=[];images=[]
    for bb in samples:
        M=np.c_[bb,np.ones(len(bb))];gb=np.linalg.lstsq(M*weight[:,None]**.5,aa*weight**.5,rcond=None)[0];gb[0]=np.clip(gb[0],.3,3);gb[1]=np.sum(weight*(aa-gb[0]*bb))/weight.sum()
        adjusted=gb[0]*bb+gb[1];am=np.sum(weight*aa)/weight.sum();bm=np.sum(weight*bb)/weight.sum()
        ncc=np.sum(weight*(aa-am)*(bb-bm))/np.sqrt(np.sum(weight*(aa-am)**2)*np.sum(weight*(bb-bm)**2)+1e-12)
        rms=np.sqrt(np.sum(weight*(adjusted-aa)**2)/weight.sum())
        stats.append(dict(ncc=float(ncc),rms=float(rms),gain=float(gb[0]),offset=float(gb[1])));images.append(adjusted)
    return dict(i=int(i),center_full=(c+origin).tolist(),params=pp.tolist(),shape=shape,aa=aa,weight=weight,common=common,valid=valid,images=images,stats=stats,dx=dx,dy=dy)
q=results['query_full'][200:200+int(results['grid_count'])];depth=results['depth'][200:200+len(q)];passed=results['accepted'][200:200+len(q)]
candidate=np.where(passed&(q[:,0]>=375)&(q[:,0]<=640)&(depth>=18)&(depth<=65)&f['frozen_original'])[0]
cases=[]
for i in candidate:
    c=sample_case(i)
    if c['common'].sum()>=500 and c['stats'][2]['ncc']>.75:cases.append(c)
# Require visible curvature benefit after affine ablation, then rank combined gain.
ranked=sorted(cases,key=lambda c: c['stats'][2]['ncc']-c['stats'][0]['ncc']+.8*(c['stats'][2]['ncc']-c['stats'][1]['ncc']),reverse=True)
case=next((c for c in ranked if c['stats'][2]['ncc']-c['stats'][1]['ncc']>.025),ranked[0]);i=case['i'];j=200+i
print('SELECTED',i,case['center_full'],case['stats'],flush=True)
shape=case['shape'];mask=~case['common'].reshape(shape);source=np.ma.array(case['aa'].reshape(shape),mask=mask)
fig,axs=plt.subplots(2,3,figsize=(7.15,6.15));fig.subplots_adjust(left=.08,right=.95,top=.80,bottom=.17,wspace=.25,hspace=.65)
fig.suptitle('What a deforming image match changes in a real neighborhood',x=.08,y=.986,ha='left',fontsize=12,color=INK,fontweight='bold')
fig.text(.08,.925,'Saved center (x, y) = ({:.0f}, {:.0f}) px; depth = {:.2f} px; window = 27 × 27 px'.format(*case['center_full'],depth[i]),fontsize=9,color=INK)
cm=copy.copy(plt.get_cmap('gray'));cm.set_bad('#c9cdd2');crm=copy.copy(plt.get_cmap('RdBu_r'));crm.set_bad('#c9cdd2')
commonims=[source,np.ma.array(case['images'][0].reshape(shape),mask=mask),np.ma.array(case['images'][2].reshape(shape),mask=mask)]
for ax,img,title in zip(axs[0],commonims,['(a) Source A','(b) B: translation term only','(c) B: full quadratic map']):
    im=ax.imshow(img,cmap=cm,vmin=-1,vmax=3,extent=(-13.5,13.5,13.5,-13.5),interpolation='nearest');ax.set_title(title,pad=6,fontsize=9);ax.set_xlabel('Offset x (px)')
weight=np.ma.array(case['weight'].reshape(shape),mask=mask);wm=axs[1,0].imshow(weight,cmap='YlGnBu',vmin=0,vmax=1,extent=(-13.5,13.5,13.5,-13.5),interpolation='nearest');axs[1,0].set_title('(d) Weights on common pixels',fontsize=9,pad=6)
for ax,n,title in [(axs[1,1],0,'(e) Translation residual'),(axs[1,2],2,'(f) Quadratic residual')]:
    res=np.ma.array((case['images'][n]-case['aa']).reshape(shape),mask=mask);rim=ax.imshow(res,cmap=crm,vmin=-1.5,vmax=1.5,extent=(-13.5,13.5,13.5,-13.5),interpolation='nearest');ax.set_title(title,fontsize=9,pad=6)
for ax in axs.ravel():ax.set_xticks([-10,0,10]);ax.set_yticks([-10,0,10]);ax.set_aspect('equal')
for ax in axs[:,0]:ax.set_ylabel('Offset y (px, down)')
for ax in axs[1]:ax.set_xlabel('Offset x (px)')
fig.text(.08,.883,'Common samples: {}. Weighted NCC {:.3f} → {:.3f}; weighted RMS {:.3f} → {:.3f}.'.format(int(case['common'].sum()),case['stats'][0]['ncc'],case['stats'][2]['ncc'],case['stats'][0]['rms'],case['stats'][2]['rms']),fontsize=9,color=INK)
fig.text(.08,.055,'Same saved translation; higher coefficients removed in (b). Spatial parameters are never refitted.',fontsize=8.5,color=INK)
fig.text(.08,.020,'Residual = adjusted B − A: blue −1.5, white 0, red +1.5. Gray would mark excluded pixels.',fontsize=8.5,color=INK)
save(fig,'actual_registration')
# 3. Manual and model velocities at exactly the same coordinates.
manual=results['query_full'][:200];truth=results['manual_truth'];pred=results['disp'][:200];mok=results['accepted'][:200]
masked=np.ma.array(raw,mask=~va);gc=copy.copy(plt.get_cmap('gray'));gc.set_bad('#181e29')
fig,axs=plt.subplots(2,1,figsize=(7.15,7.0));fig.subplots_adjust(left=.095,right=.98,top=.91,bottom=.17,hspace=.32)
fig.suptitle('Checking the field against existing manual particle matches',x=.095,y=.985,ha='left',fontsize=12,color=INK,fontweight='bold')
for ax in axs:
    imglobal(ax,masked,gc,25,180);roi(ax,355,710,342,525);lines(ax,True)
    ax.set_xticks([375,450,525,600,675]);ax.set_yticks([350,400,450,500])
axs[0].quiver(manual[:,0],manual[:,1],truth[:,0],truth[:,1],color=GOLD,angles='xy',scale_units='xy',scale=1,width=.0028,headwidth=3.5,headlength=4.2,minlength=.15)
axs[0].set_title('(a) Manual A → B displacements: all 200 supplied matches',loc='left',pad=7)
axs[1].quiver(manual[mok,0],manual[mok,1],pred[mok,0],pred[mok,1],color=CYAN,angles='xy',scale_units='xy',scale=1,width=.0028,headwidth=3.5,headlength=4.2,minlength=.15)
axs[1].scatter(manual[~mok,0],manual[~mok,1],c=RED,marker='x',s=23,lw=.9)
axs[1].set_title('(b) Blended quadratic field: 185 accepted at those 200 locations',loc='left',pad=7)
fig.text(.095,.075,'Arrows use true A → B displacement scale (pixels); crosses = 15 withheld model estimates.',fontsize=9,color=INK)
fig.text(.095,.036,'Mean endpoint disagreement: 0.497 px for all 200; 0.452 px for the 185 accepted locations.',fontsize=9,color=INK)
save(fig,'actual_validation')
# 4. Coverage map summarizes all source grid decisions at legible scale; avoid
# shrinking 10k arrows into a seven-inch figure.
N=int(results['grid_count']);sl=slice(200,200+N);gq=results['query_full'][sl];gok=results['accepted'][sl];gradok=results['gradient_accepted'][sl]
fig,axs=plt.subplots(2,1,figsize=(7.15,4.5));fig.subplots_adjust(left=.08,right=.98,top=.85,bottom=.23,hspace=.68)
fig.suptitle('Where the saved evidence supports reporting a value',x=.08,y=.985,ha='left',fontsize=12,color=INK,fontweight='bold')
for ax,flag,title in zip(axs,[gok,gradok],['(a) Displacement: 10,098 of 10,946 grid points accepted','(b) Horizontal derivative: 9,838 of 10,946 grid points accepted']):
    ax.set_facecolor('#e8ebef');ax.scatter(gq[flag,0],gq[flag,1],s=.7,c='#117f8a',lw=0,rasterized=True);ax.scatter(gq[~flag,0],gq[~flag,1],s=2.1,c='#c75861',marker='x',lw=.3,rasterized=True)
    ax.plot(xs,z['full_surface_a'],color='#835899',lw=1);ax.fill_between(xs,z['full_surface_a'],va.argmax(axis=0)+origin[1],color=PURPLE,alpha=.5,lw=0)
    ax.add_patch(Rectangle((manual[:,0].min(),manual[:,1].min()),np.ptp(manual[:,0]),np.ptp(manual[:,1]),fill=False,edgecolor='#ee9c27',lw=1))
    ax.set(xlim=(0,2047),ylim=(752,332),xlabel='Full-image x (px)',ylabel='y (px, down)');ax.set_aspect('equal');ax.set_title(title,loc='left',pad=7,fontsize=9.5);ax.set_yticks([400,600]);ax.set_xticks([0,500,1000,1500,2000])
handles=[Line2D([],[],color='#117f8a',marker='.',ls='none',label='Accepted'),Line2D([],[],color='#c75861',marker='x',ls='none',label='Withheld'),Patch(edgecolor='#ee9c27',facecolor='none',label='Manual-pick extent')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.06),ncol=3,frameon=False)
fig.text(.08,.025,'Sampling every 8 px is not a claim of 8 px spatial resolution. The same saved results are shown.',fontsize=8.5,color=INK)
save(fig,'actual_coverage')
# Exact audit numbers for use in the document; all scalar stats trace to frozen files.
selected_meta={k:case[k] for k in ['i','center_full','params','stats']}
selected_meta.update(depth=float(depth[i]),common_valid_count=int(case['common'].sum()),nominal_count=int(np.prod(shape)),saved_local_ncc=float(f['ncc'][i]),saved_local_rms=float(f['rms'][i]),local_center_displacement=f['params'][i,:,0].tolist(),local_center_gradient=(f['params'][i,:,1:3]/float(f['radius'])).tolist(),local_hessian=(f['params'][i,:,3:]/float(f['radius'])**2).tolist(),blended_displacement=results['disp'][j].tolist(),blended_gradient=results['gradient'][j].tolist(),blended_ncc=float(results['ncc'][j]),blended_fb=float(results['fb'][j]),blended_method_spread=float(results['method_spread'][j]),blended_gradient_spread=float(results['gradient_spread'][j]),vector_accepted=bool(results['accepted'][j]),horizontal_gradient_accepted=bool(results['gradient_accepted'][j]))
# DoG peak rules at saved source records within chosen teaching view.
ix=t['points'][sel,0].astype(int);iy=t['points'][sel,1].astype(int)
det_meta=dict(bounds_full=[lo,hi,top,bottom],saved_candidate_count=int(sel.sum()),saved_accepted_count=int(ok.sum()),saved_rejected_count=int(bad.sum()),candidate_peak_min=float(dog[iy,ix].min()),candidate_background_max=float(bg[iy,ix].max()),candidate_min_depth=float(t['source_depth'][sel].min()),all_recomputed_DoG_maxima=bool(np.all(dog[iy,ix]==maximum_filter(dog,5)[iy,ix])))
meta=dict(detection=det_meta,registration=selected_meta,manual=dict(count=200,accepted=int(mok.sum()),all_mean_EPE=float(results['manual_error'].mean()),accepted_mean_EPE=float(results['manual_error'][mok].mean())),figures=['actual_detection','actual_registration','actual_validation','actual_coverage'])
(ROOT/'work/tutorial_actual_figure_numbers.json').write_text(json.dumps(meta,indent=2))
notes='''# Actual-data tutorial figures\n\nAll figures are regenerated from frozen saved inputs, model parameters, track records and final results. No image or velocity has been refitted or edited. Coordinates are zero-based full-image pixels, with y downward. The prepared image origin is (0,260). PNG and vector PDF versions are in outputs/hybrid_tutorial_figures.\n\n## actual_detection\nFour panels show the actual raw A image, Gaussian background sigma=5, difference of Gaussians sigma=.6 minus sigma=2, and saved automatic tracking decisions. Gaussian filtering is on floating-point raw intensities with SciPy default reflected boundaries, exactly the implemented detection convention. Circle positions are preserved candidate records; they are not newly generated matches. Red background-overlay pixels have C>=180, which is a broad-background rejection rule rather than reflection classification. Gold is supplied geometry, green dashed is geometry+14, purple is geometry-to-first-valid-pixel mask exclusion. The ROI deliberately contains the original-image overlap, so restored raw image values can be visible behind the purple band. Sources below the 14-pixel depth line can still be rejected for NCC, reverse closure, or ambiguity. B particle centers were not detected separately. Tracking arrows use actual displacement scale. Endpoints belong to frame B but are shown over A for context; B has its own mask, so an endpoint over the A exclusion overlay is not itself a B-validity failure. Counts and filter extrema are in the JSON.\n\n## actual_registration\nThe selected example uses a frozen original local quadratic map at an accepted full-width grid location near the first depression. Source samples occupy a nominal 27x27 integer window. B is read by bilinear interpolation at source+displacement. Translation uses only the saved constant coefficient, affine uses its first three coefficients, and quadratic uses all six coefficients per component. All three sample sets are intersected with the A mask, interpolated B mask>.99, and strict B solver-domain bounds; the identical common sample set and Gaussian weights are used to compare NCC and RMS. For display and RMS, each sampling case receives its own weighted least-squares intensity gain (clipped .3..3) and re-estimated offset, matching the solver's photometric procedure but restricted to the common set. This is photometric adjustment only; no spatial fitting occurs.\n\nThis is an illustrative coefficient-ablation example selected for visible deformation benefit. The translation and affine alternatives are not independent optimized fits, so their comparison cannot establish the gain of quadratic optimization over optimally refitted simpler models. The grayscale image display range is -1 to 3 normalized intensity units; the weight panel runs from pale 0 to dark-blue 1. Gray cells are outside the common mask (none are excluded in this selected example). The NCC/RMS comparison numbers need not equal the saved local-fit diagnostic because common intersection changes its samples. The saved local map's center displacement and slopes are not the final quiver values: final values blend all overlapping maps. The field's composed reverse error, model sensitivity, and gradient are therefore explicitly reported separately in the JSON. Curvature components are dxx,dxy,dyy in px/(px^2). Do not imply that particle spin is measured.\n\n## actual_validation\nTwo panels show all 200 manual picks and the 185 conservatively accepted final blended estimates at those exact locations. The 15 rejected model locations are crosses. Arrows remain actual displacements in pixels per pair, not rescaled vectors. Manual references did not fit the model, but this same pair informed earlier method comparisons and is not an untouched test set. The error quoted for all 200 includes raw model predictions at withheld locations; the accepted-only number is distinct.\n\n## actual_coverage\nThe two rows show the actual final vector and horizontal-horizontal derivative masks at 10,946 grid sources, not additional fit results. Colored dots avoid shrinking 10k arrows to illegibility. The orange rectangle only bounds the 200 existing manual source locations; it does not imply a filled validation region or additional labels. The sampling interval is not effective spatial resolution.\n'''
(ROOT/'work/tutorial_actual_figure_notes.md').write_text(notes)
print(json.dumps(meta,indent=2),flush=True)
