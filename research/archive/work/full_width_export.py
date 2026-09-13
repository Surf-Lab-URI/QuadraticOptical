"""Create screened full-width figures and scientific data products."""
import os
os.environ['MPLCONFIGDIR']=os.path.abspath('work/mplcache')
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,csv,copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import savemat
W=Path('work/full_width');O=Path('outputs');P='full_width_conservative_'
with np.load(W/'results.npz') as stored:
    z={k:stored[k] for k in stored.files}
inp=np.load(W/'inputs.npz');s=json.loads((W/'summary.json').read_text())
q=z['query_full'];d=z['disp'];g=z['gradient'];ok=z['accepted'];gok=z['gradient_accepted'];N=int(z['grid_count'])
gs=slice(200,200+N);ds=slice(200+N,None);sa=z['full_surface_a'];xs=np.arange(2048);origin=z['origin0']
DX=float(z['DX']);DT=float(z['DT']);truth=z['manual_truth'];manual=q[:200]
INK='#17263b';MUTED='#506176';CYAN='#24d2ed';ORANGE='#ffb04a';PURPLE='#aa7da7';REJECT='#d9a1ab'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.labelsize':10,
                     'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
A=np.ma.array(inp['rawA'],mask=~inp['va']);cmap=copy.copy(plt.get_cmap('gray'));cmap.set_bad('#10131a')
first_valid_y=np.argmax(inp['va'],axis=0)+origin[1]
def save(fig,name):
    fig.savefig(O/(P+name+'.png'),dpi=240,bbox_inches='tight')
    fig.savefig(O/(P+name+'.svg'),bbox_inches='tight');plt.close(fig)
def bg(ax,limits=(0,2047,755,330),image=True):
    if image:
        ax.imshow(A,cmap=cmap,vmin=35,vmax=180,origin='upper',extent=(-.5,2047.5,839.5,259.5),alpha=.7)
        ax.set_facecolor('#10131a')
    else:ax.set_facecolor('#e6e9ee')
    ax.plot(xs,sa,color='#f5ddee' if image else '#80527e',lw=1)
    ax.fill_between(xs,sa,first_valid_y,color=PURPLE,alpha=.45,lw=0)
    ax.set_xlim(limits[:2]);ax.set_ylim(limits[2:]);ax.set_aspect('equal')
    ax.set_xlabel('Full-image x (px)');ax.set_ylabel('Full-image y (px, downward)')
def quiv(ax,pts,vec,width=.0007,color=CYAN):
    return ax.quiver(pts[:,0],pts[:,1],vec[:,0],vec[:,1],angles='xy',scale_units='xy',scale=1,color=color,
                     width=width,headwidth=3.6,headlength=4.5,headaxislength=4.1,minlength=.12,pivot='tail')
def field(ax,lo=0,hi=2048,width=.0007,dots=1.0):
    select=(q[gs,0]>=lo)&(q[gs,0]<hi);a=ok[gs]&select;r=(~ok[gs])&select
    ax.scatter(q[gs][a,0],q[gs][a,1],s=dots,c=CYAN,alpha=.60,lw=0)
    quiv(ax,q[gs][a],d[gs][a],width=width)
    ax.scatter(q[gs][r,0],q[gs][r,1],s=dots*1.8,c=REJECT,marker='x',alpha=.50,lw=.4)

fig,ax=plt.subplots(figsize=(16,5.4));fig.subplots_adjust(top=.82,bottom=.24,left=.065,right=.98)
fig.suptitle('Conservative field across the full image width',x=.065,y=.99,ha='left',fontsize=20,fontweight='bold',color=INK)
fig.text(.065,.89,'Original quadratic model · same confidence thresholds · A → B arrows at true displacement scale',fontsize=11,color=MUTED)
bg(ax);field(ax)
fig.text(.065,.145,'{:,} / {:,} grid locations pass. Cyan dots mark accepted locations with tiny motion; crosses mark rejected estimates.'.format(s['accepted_grid'],N),fontsize=10.5,color=INK)
fig.text(.065,.091,'Full width ≈11.57 cm; depth limit ≈2.00 cm if the supplied DX is metres/pixel. Missing mask pixels and image edges remain excluded.',fontsize=10,color=MUTED)
fig.text(.065,.041,'Only the first depression has manual reference matches. The other regions are checked by image consistency, not additional manual labels.',fontsize=10,color=MUTED)
save(fig,'quiver')

# Overlapping windows keep each depression visible within a readable panel.
fig,axs=plt.subplots(2,2,figsize=(13.5,11.5));fig.subplots_adjust(top=.89,bottom=.13,wspace=.15,hspace=.26)
fig.suptitle('Full-width field · overlapping detailed views',x=.075,y=.99,ha='left',fontsize=20,fontweight='bold',color=INK)
fig.text(.075,.942,'Same accepted vectors, shown at true displacement scale; window overlap provides context only.',fontsize=10.5,color=MUTED)
for ax,(lo,hi) in zip(axs.ravel(),[(0,650),(450,1150),(950,1650),(1450,2048)]):
    selected=ok[gs]&(q[gs,0]>=lo)&(q[gs,0]<hi);endpoints=q[gs][selected]+d[gs][selected]
    left=max(0,min(lo-8,float(endpoints[:,0].min())-5));right=min(2047,max(hi+8,float(endpoints[:,0].max())+5))
    bg(ax,(left,right,755,330));field(ax,lo,hi,width=.0025,dots=2)
    ax.set_title('Sources x = {}–{} px'.format(lo,hi-1),loc='left',fontweight='bold',pad=10)
fig.text(.075,.068,'Purple: excluded surface band, including unavailable glare pixels. Cyan dots mark tiny motions; no vector magnitude is enlarged.',color=MUTED,fontsize=10)
save(fig,'quiver_details')

# Horizontal derivative and its sensitivity share exactly the same image axes.
xx=z['dense_x_full'];yy=z['dense_y_full'];shape=xx.shape
xe,ye=np.meshgrid(np.r_[xx[0]-2,xx[0,-1]+2],np.r_[yy[:,0]-2,yy[-1,0]+2])
gm=g[ds,0,0].reshape(shape);valid=gok[ds].reshape(shape);sensitivity=z['gradient_spread'][ds].reshape(shape)
fig,axs=plt.subplots(2,1,figsize=(16,8.7));fig.subplots_adjust(top=.88,bottom=.14,left=.065,right=.95,hspace=.34)
fig.suptitle('Horizontal gradient across the full image width',x=.065,y=.99,ha='left',fontsize=20,fontweight='bold',color=INK)
fig.text(.065,.941,'∂dₓ/∂x at fixed image height · derivative of the same smoothly blended displacement field',fontsize=11,color=MUTED)
for ax in axs:bg(ax,image=False)
im=axs[0].pcolormesh(xe,ye,np.ma.array(gm,mask=~valid),cmap='RdBu_r',vmin=-.2,vmax=.2,shading='flat',rasterized=True)
axs[0].set_title('Supported horizontal displacement gradient',loc='left',fontweight='bold',pad=9)
cb=fig.colorbar(im,ax=axs[0],shrink=.85,pad=.015,extend='both');cb.set_label('∂dₓ/∂x (px/px)')
im=axs[1].pcolormesh(xe,ye,np.ma.array(sensitivity,mask=~valid),cmap='magma',vmin=0,vmax=.08,shading='flat',rasterized=True)
axs[1].set_title('Maximum change across the three comparison fits',loc='left',fontweight='bold',pad=9)
cb=fig.colorbar(im,ax=axs[1],shrink=.85,pad=.015);cb.set_label('Sensitivity (px/px)')
fig.text(.065,.083,'{:,} grid locations pass the stricter horizontal-gradient screen. Gray areas are withheld. Sensitivity is not a confidence interval.'.format(s['gradient_grid']),fontsize=10,color=MUTED)
fig.text(.065,.038,'If DT = 0.01 s, multiply the upper values by 100 for the finite-time velocity-gradient proxy in s⁻¹.',fontsize=10,color=MUTED)
save(fig,'horizontal_gradient')

fig,axs=plt.subplots(1,2,figsize=(14,5.3));fig.subplots_adjust(top=.79,bottom=.22,wspace=.13)
fig.suptitle('Existing manual reference · first depression',x=.08,y=.99,ha='left',fontsize=19,fontweight='bold',color=INK)
fig.text(.08,.925,'Identical source locations in both panels · A → B displacement in pixels',fontsize=11,color=MUTED)
for ax in axs:bg(ax,(359,699,524,341))
axs[0].set_title('Manual · all 200 picks',loc='left',fontweight='bold',pad=12)
quiv(axs[0],manual,truth,width=.003,color=ORANGE)
axs[1].set_title('Model · {} / 200 pass'.format(int(ok[:200].sum())),loc='left',fontweight='bold',pad=12)
quiv(axs[1],manual[ok[:200]],d[:200][ok[:200]],width=.003)
axs[1].scatter(manual[~ok[:200],0],manual[~ok[:200],1],s=30,marker='x',c=REJECT);axs[1].set_ylabel('')
fig.text(.08,.135,'Mean disagreement: {:.3f} px across all 200 picks; {:.3f} px across the 26 picks within 40 px of the trace.'.format(s['manual_metrics']['all']['mean'],s['manual_metrics']['near40']['mean']),color=INK,fontsize=10.5)
fig.text(.08,.071,'These matches did not fit the model. This pair informed earlier method comparisons and is not an untouched validation set.',color=MUTED,fontsize=10)
save(fig,'manual_comparison')

diags=['depth','target_depth','ncc','fb','method_spread','gradient_spread','nearest_feature','feature_count25','determinant',
       'alternatives_available','local_valid_share','reverse_valid_share','support','source_visible','target_visible','evidence_pass','accepted','gradient_accepted']
du=d[gs].copy();du[~ok[gs]]=np.nan;gd=g[gs,0,0].copy();gd[~gok[gs]]=np.nan
denseu=d[ds].reshape(shape+(2,)).copy();denseu[~ok[ds].reshape(shape)]=np.nan;denseg=gm.copy();denseg[~valid]=np.nan
def table(name,indices,ismanual=False):
    with (O/(P+name+'.csv')).open('w',newline='') as f:
        wr=csv.writer(f);header=['pick' if ismanual else 'node','x_full_zero_based_px','y_full_zero_based_px','dx_raw_px_per_pair','dy_down_raw_px_per_pair',
                                'd_dx_dx_raw','d_dx_dy_raw','d_dy_dx_raw','d_dy_dy_raw']+diags
        header+=['dx_conservative_px_per_pair','dy_down_conservative_px_per_pair','d_dx_dx_conservative','U_conservative_assumed_mps','V_up_conservative_assumed_mps','dU_dX_conservative_assumed_per_s']
        if ismanual:header+=['manual_dx_px','manual_dy_down_px','endpoint_error_px','change_from_previous_prediction_px']
        else:header+=['previous_patch_frozen','original_patch_frozen']
        wr.writerow(header)
        for row,j in enumerate(indices):
            u=d[j] if ok[j] else np.array([np.nan,np.nan]);gg=g[j,0,0] if gok[j] else np.nan
            vals=[row+1,*q[j],*d[j],*g[j].ravel()]+[int(z[k][j]) if z[k].dtype.kind in 'biu' else float(z[k][j]) for k in diags]
            vals += [*u,gg,u[0]*DX/DT,-u[1]*DX/DT,gg/DT]
            if ismanual:vals += [*truth[j],float(z['manual_error'][j]),float(z['previous_manual_change'][j])]
            else:vals += [int(z['previous_grid_flag'][row]),int(z['original_grid_flag'][row])]
            wr.writerow(vals)
table('grid',range(200,200+N));table('manual',range(200),True)
tr=np.load(W/'ptv_tracks.npz')
data=dict(grid_xy_full_zero_based=q[gs],grid_xy_full_matlab_one_based=q[gs]+1,
    grid_displacement_raw_px_per_pair=d[gs],grid_displacement_gradient_raw=g[gs],grid_displacement_conservative_px_per_pair=du,
    grid_d_dx_dx_conservative=gd,grid_velocity_conservative_assumed_mps=du*DX/DT*np.array([1,-1]),grid_dU_dX_conservative_assumed_per_s=gd/DT,
    grid_accepted=ok[gs].astype(np.uint8),grid_horizontal_gradient_accepted=gok[gs].astype(np.uint8),
    grid_previous_patch_frozen=z['previous_grid_flag'].astype(np.uint8),grid_original_patch_frozen=z['original_grid_flag'].astype(np.uint8),
    manual_xy_full_zero_based=manual,manual_xy_full_matlab_one_based=manual+1,manual_displacement_px_per_pair=truth,
    manual_velocity_assumed_mps=truth*DX/DT*np.array([1,-1]),model_at_manual_raw_px_per_pair=d[:200],manual_model_accepted=ok[:200].astype(np.uint8),
    manual_error_px=z['manual_error'],manual_prediction_change_px=z['previous_manual_change'],
    dense_x_full_zero_based_px=xx,dense_y_full_zero_based_px=yy,dense_displacement_conservative_px_per_pair=denseu,
    dense_d_dx_dx_conservative=denseg,dense_accepted=ok[ds].reshape(shape).astype(np.uint8),dense_horizontal_gradient_accepted=valid.astype(np.uint8),
    dense_velocity_conservative_assumed_mps=denseu*DX/DT*np.array([1,-1]),dense_dU_dX_conservative_assumed_per_s=denseg/DT,
    surface_x_full_zero_based=xs,surface_a_y_full_zero_based=sa,surface_b_y_full_zero_based=z['full_surface_b'],
    automatic_track_xy_full_zero_based=tr['points']+origin,automatic_displacement_px_per_pair=tr['disp'],automatic_track_accepted=tr['accepted'].astype(np.uint8),
    DX_from_supplied_MAT=DX,DT_from_supplied_MAT=DT,physical_units_confirmed=np.uint8(0),
    units_note='Pixels are primary. Assumed SI arrays require DX in metres/pixel and DT in seconds, absent from sourceunitlabels. Uright,Vup; rawdy is down.',
    gradient_note='RawG differentiates A-to-B displacement with respect to A imagecoordinates. Only horizontal-horizontal entry has derivative sensitivityscreen. DividebyDT for sourceposition derivative ofintervalaveragedmaterialmotion; not independentlyresolved instantaneousEulerian gradient.',
    model_note='Original maskedquadratic bilinearfit radius13,reg0.00005,automaticPTVprior,C2blend radius16,originalevidence thresholds. Previous2184patches frozen; added source/targetvisiblepixel and imageboundsdomaincheck.',
    sampling_note='Fittinggrid8px and denseevaluation4px are sample spacings, not spatialresolution. Useconservativearrays withNaNat rejectedlocations.')
for k in diags:data['grid_'+k]=z[k][gs]
savemat(O/(P+'field.mat'),data,do_compression=True);np.savez_compressed(O/(P+'field.npz'),**data)
(O/(P+'summary.json')).write_text(json.dumps(s,indent=2));print('Exported full-width figures, data and tables.',flush=True)
