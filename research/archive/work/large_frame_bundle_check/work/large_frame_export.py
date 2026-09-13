"""Export the larger-frame conservative analysis; never change fitted results."""
import os
os.environ['MPLCONFIGDIR']=os.path.abspath('work/mplcache')
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.io import savemat
from PIL import Image

W=Path('work/large_frame');O=Path('outputs');prefix='large_frame_conservative_'
z=np.load(W/'results.npz');inp=np.load(W/'inputs.npz');s=json.loads((W/'summary.json').read_text())
q=z['query_full'];d=z['disp'];g=z['gradient'];ok=z['accepted'];gok=z['gradient_accepted']
N=int(z['grid_count']);gs=slice(200,200+N);ds=slice(200+N,None)
truth=z['manual_truth'];manual=q[:200];sa=z['full_surface_a'];xs=np.arange(len(sa));origin=z['origin0']
DX=float(z['DX']);DT=float(z['DT']);scale=DX/DT
INK='#17263b';MUTED='#506176';CYAN='#1fc8ed';ORANGE='#ffb04a';PURPLE='#aa7da7';REJECT='#d49ba5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.labelsize':10,
 'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
A=inp['rawA'];h,w=A.shape

def save(fig,name):
    fig.savefig(O/(prefix+name+'.png'),dpi=240,bbox_inches='tight')
    fig.savefig(O/(prefix+name+'.svg'),bbox_inches='tight')
    plt.close(fig)

def background(ax,limits=(319,719,755,335),image=True):
    if image:
        ax.imshow(A,cmap='gray',vmin=35,vmax=180,origin='upper',
                  extent=(origin[0]-.5,origin[0]+w-.5,origin[1]+h-.5,origin[1]-.5))
    else:ax.set_facecolor('#e6e9ee')
    ax.fill_between(xs,sa,sa+10,color=PURPLE,alpha=.42,lw=0)
    ax.plot(xs,sa,color='#f8e0f3' if image else '#785078',lw=1.2)
    ax.set_xlim(limits[:2]);ax.set_ylim(limits[2:]);ax.set_aspect('equal')
    ax.set_xlabel('Full-image x (px)');ax.set_ylabel('Full-image y (px, downward)')

def quiver(ax,p,v,color=CYAN,width=.0027,alpha=1):
    return ax.quiver(p[:,0],p[:,1],v[:,0],v[:,1],color=color,angles='xy',scale_units='xy',scale=1,
        width=width,headwidth=3.6,headlength=4.5,headaxislength=4.1,pivot='tail',alpha=alpha,minlength=.15)

# Requested comparison on the larger field of view. Every accepted 8 px grid node is shown.
fig,axs=plt.subplots(1,2,figsize=(12.2,8.0));fig.subplots_adjust(top=.85,bottom=.20,wspace=.17)
fig.suptitle('Capillary depression · conservative field extension',x=.085,y=.99,ha='left',fontsize=19,fontweight='bold',color=INK)
fig.text(.085,.937,'Same quadratic model and quality thresholds as the original analysis',color=MUTED,fontsize=11)
for ax in axs:
    background(ax,(315,745,755,335))
    ax.images[0].set_alpha(.65);ax.set_facecolor('#10131a')
axs[0].set_title('Manual particle matches · 200',loc='left',fontweight='bold',pad=12)
quiver(axs[0],manual,truth,ORANGE,width=.0032)
axs[1].set_title('Predicted field · {:,} accepted vectors'.format(s['accepted_grid']),loc='left',fontweight='bold',pad=12)
axs[1].scatter(q[gs][ok[gs],0],q[gs][ok[gs],1],s=2.5,c=CYAN,alpha=.75,lw=0)
pred=quiver(axs[1],q[gs][ok[gs]],d[gs][ok[gs]])
axs[1].scatter(q[gs][~ok[gs],0],q[gs][~ok[gs],1],s=4,c=REJECT,marker='x',alpha=.6,lw=.5)
for ax in axs:
    ax.text(.04,.035,'A → B · true displacement scale',transform=ax.transAxes,color='white',fontsize=9,
            bbox=dict(facecolor='#102032',alpha=.7,edgecolor='none',pad=4))
axs[1].set_ylabel('')
fig.text(.085,.137,'Mean disagreement at all manual picks: 0.497 px. Manual matches cover the upper portion only.',color=INK,fontsize=10.5)
fig.text(.085,.091,'Cyan dots mark accepted locations even where motion is tiny. Crosses are rejected; purple excludes glare.',color=MUTED,fontsize=10)
fig.text(.085,.045,'Depth limit: 354 px below the traced surface (≈2 cm if the supplied DX is metres/pixel).',color=MUTED,fontsize=10)
save(fig,'quiver')

# Zoomed manual validation; strict screened predictions, no flagged vectors promoted for display.
fig,axs=plt.subplots(1,2,figsize=(14,5.3));fig.subplots_adjust(top=.79,bottom=.22,wspace=.13)
fig.suptitle('Manual comparison near the depression',x=.08,y=.99,ha='left',fontsize=19,fontweight='bold',color=INK)
fig.text(.08,.925,'Identical source locations in both panels · A → B displacement in pixels',color=MUTED,fontsize=11)
for ax in axs:background(ax,(359,699,524,341))
axs[0].set_title('Manual · all 200 picks',loc='left',fontweight='bold',pad=12);quiver(axs[0],manual,truth,ORANGE,width=.003)
axs[1].set_title('Model · 185 / 200 pass the screen',loc='left',fontweight='bold',pad=12)
quiver(axs[1],manual[ok[:200]],d[:200][ok[:200]],width=.003)
axs[1].scatter(manual[~ok[:200],0],manual[~ok[:200],1],marker='x',s=30,color=REJECT,lw=1)
axs[1].set_ylabel('')
fig.text(.08,.135,'All 200: mean 0.497 px, median 0.393 px. Within 40 px of the trace: mean 0.665 px (26 picks).',color=INK,fontsize=10.5)
fig.text(.08,.073,'Crosses indicate screened-out model locations. All raw predictions and diagnostics remain in the data files.',color=MUTED,fontsize=10)
save(fig,'manual_comparison')

# Full-frame overview retains the provided image; only the analyzed first depression is outlined.
fullpath=W/'source_imgA.tif'
if not fullpath.exists():fullpath=Path('data/ExpLCL_1_03_123_imgA.tif')
full=np.asarray(Image.open(str(fullpath)))
fig,ax=plt.subplots(figsize=(14,5.3));fig.subplots_adjust(top=.85,bottom=.21)
ax.imshow(full,cmap='gray',vmin=35,vmax=180,origin='upper');ax.plot(xs,sa,color='#f8e0f3',lw=1)
ax.add_patch(Rectangle((315,335),430,420,fill=False,lw=2,color=CYAN))
ax.set(xlim=(0,2047),ylim=(855,285),xlabel='Full-image x (px)',ylabel='Full-image y (px, downward)')
ax.set_title('Larger frame A · region analyzed around the original depression',loc='left',fontweight='bold',fontsize=16,pad=15)
fig.text(.125,.095,'Cyan rectangle: plotted region. The supplied larger TIFF is already masked above its glare boundary.',color=MUTED,fontsize=10)
fig.text(.125,.045,'Inside the original crop, the earlier supplied image pixels were retained to preserve the original near-surface estimates.',color=MUTED,fontsize=10)
save(fig,'overview')

# The G00 screen applies only to the horizontal derivative of horizontal displacement.
xx=z['dense_x_full'];yy=z['dense_y_full'];shape=xx.shape
xe,ye=np.meshgrid(np.r_[xx[0]-2,xx[0,-1]+2],np.r_[yy[:,0]-2,yy[-1,0]+2])
gm=g[ds,0,0].reshape(shape);valid=gok[ds].reshape(shape);sens=z['gradient_spread'][ds].reshape(shape)
fig,axs=plt.subplots(1,2,figsize=(12.2,8.1));fig.subplots_adjust(top=.85,bottom=.18,wspace=.31)
fig.suptitle('Horizontal gradient of the conservative field',x=.08,y=.99,ha='left',fontsize=19,fontweight='bold',color=INK)
fig.text(.08,.935,'∂dₓ/∂x at fixed image height · analytic derivative of the same blended displacement field',color=MUTED,fontsize=10.5)
for ax in axs:background(ax,image=False)
im=axs[0].pcolormesh(xe,ye,np.ma.array(gm,mask=~valid),cmap='RdBu_r',vmin=-.2,vmax=.2,shading='flat',rasterized=True)
axs[0].set_title('Screened displacement gradient',loc='left',fontweight='bold',pad=12)
cb=fig.colorbar(im,ax=axs[0],shrink=.8,pad=.025,extend='both');cb.set_label('∂dₓ/∂x (px/px)')
im=axs[1].pcolormesh(xe,ye,np.ma.array(sens,mask=~valid),cmap='magma',vmin=0,vmax=.08,shading='flat',rasterized=True)
axs[1].set_title('Sensitivity across three variants',loc='left',fontweight='bold',pad=12);axs[1].set_ylabel('')
cb=fig.colorbar(im,ax=axs[1],shrink=.8,pad=.025);cb.set_label('Maximum gradient change (px/px)')
fig.text(.08,.115,'1,986 grid nodes pass the stricter ∂dₓ/∂x screen. Gray areas are withheld; sensitivity is not a confidence interval.',color=MUTED,fontsize=10)
fig.text(.08,.065,'If DT = 0.01 s, multiply the left-hand values by 100 for the finite-time velocity gradient in s⁻¹.',color=MUTED,fontsize=10)
save(fig,'horizontal_gradient')

# Validation separates evidence at manual locations from added, unlabelled depth coverage.
e=z['manual_error'];dep=z['depth'][:200]
fig,axs=plt.subplots(1,2,figsize=(12.2,4.9));fig.subplots_adjust(top=.84,bottom=.23,wspace=.30)
fig.suptitle('Manual validation and added coverage',x=.08,y=.99,ha='left',fontsize=18,fontweight='bold',color=INK)
axs[0].scatter(dep[ok[:200]],e[ok[:200]],s=22,c='#088bad',alpha=.8,label='Passes screen')
axs[0].scatter(dep[~ok[:200]],e[~ok[:200]],s=30,c='#b56a7b',marker='x',label='Flagged')
axs[0].axhline(1,lw=1,ls='--',color='#a1aab4');axs[0].set(xlabel='Depth below A trace (px)',ylabel='Endpoint disagreement (px)',xlim=(0,150),ylim=(0,max(3.5,float(e.max())*1.08)));axs[0].legend(fontsize=9);axs[0].grid(alpha=.15)
edges=[12,20,40,80,150,250,354.001];labels=['12–20','20–40','40–80','80–150','150–250','250–354']
total=[];accepted=[]
for lo,hi in zip(edges[:-1],edges[1:]):
    a=(z['depth'][gs]>=lo)&(z['depth'][gs]<hi);total.append(int(a.sum()));accepted.append(int((a&ok[gs]).sum()))
ix=np.arange(len(total));axs[1].bar(ix,total,color='#dce3e9',label='Evaluated');axs[1].bar(ix,accepted,color='#168cab',label='Accepted')
for i,v in enumerate(accepted):axs[1].text(i,v+9,str(v),ha='center',fontsize=9,color=INK)
axs[1].set_xticks(ix);axs[1].set_xticklabels(labels,fontsize=9);axs[1].set(xlabel='Depth band below A trace (px)',ylabel='Grid locations',ylim=(0,760));axs[1].legend(fontsize=9)
fig.text(.08,.12,'The added deeper field passes image consistency checks, but has no additional manual validation.',color=MUTED,fontsize=10)
fig.text(.08,.061,'The same image pair informed earlier method comparisons; these 200 picks are not an untouched test set.',color=MUTED,fontsize=10)
save(fig,'validation')

diag=['depth','target_depth','ncc','fb','method_spread','gradient_spread','nearest_feature','feature_count25','determinant',
      'alternatives_available','local_valid_share','reverse_valid_share','support','accepted','gradient_accepted']
def write_table(path,indices,ismanual=False):
    with path.open('w',newline='') as f:
        wr=csv.writer(f)
        header=['pick' if ismanual else 'node','x_full_zero_based_px','y_full_zero_based_px','dx_raw_px_per_pair','dy_down_raw_px_per_pair',
                'd_dx_dx_raw','d_dx_dy_raw','d_dy_dx_raw','d_dy_dy_raw']+diag
        header+=['dx_conservative_px_per_pair','dy_down_conservative_px_per_pair','d_dx_dx_conservative',
                 'U_conservative_assumed_mps','V_up_conservative_assumed_mps','dU_dX_conservative_assumed_per_s']
        if ismanual:header+=['manual_dx_px','manual_dy_down_px','endpoint_error_px','change_from_original_prediction_px']
        else:header+=['original_patch_frozen']
        wr.writerow(header)
        for row,j in enumerate(indices):
            u=d[j] if ok[j] else np.array([np.nan,np.nan]);gg=g[j,0,0] if gok[j] else np.nan
            vals=[row+1,*q[j],*d[j],*g[j].ravel()]+[int(z[k][j]) if z[k].dtype.kind in 'biu' else float(z[k][j]) for k in diag]
            vals += [*u,gg,u[0]*scale,-u[1]*scale,gg/DT]
            if ismanual:vals += [*truth[j],float(e[j]),float(z['manual_change'][j])]
            else:vals += [int(z['original_grid_flag'][row])]
            wr.writerow(vals)
write_table(O/(prefix+'grid.csv'),range(200,200+N))
write_table(O/(prefix+'manual.csv'),range(200),True)

du=d[gs].copy();du[~ok[gs]]=np.nan
gd=g[gs,0,0].copy();gd[~gok[gs]]=np.nan
dense_u=d[ds].reshape(shape+(2,)).copy();dense_u[~ok[ds].reshape(shape)]=np.nan
dense_g=gm.copy();dense_g[~valid]=np.nan
tr=np.load(W/'ptv_tracks.npz')
mat=dict(
 grid_xy_full_zero_based=q[gs],grid_xy_full_matlab_one_based=q[gs]+1,
 grid_displacement_raw_px_per_pair=d[gs],grid_displacement_gradient_raw=g[gs],
 grid_displacement_conservative_px_per_pair=du,grid_d_dx_dx_conservative=gd,
 grid_accepted=ok[gs].astype(np.uint8),grid_horizontal_gradient_accepted=gok[gs].astype(np.uint8),
 grid_velocity_conservative_assumed_mps=du*scale*np.array([1,-1]),
 grid_dU_dX_conservative_assumed_per_s=gd/DT,
 grid_original_patch_frozen=z['original_grid_flag'].astype(np.uint8),
 manual_xy_full_zero_based=manual,manual_xy_full_matlab_one_based=manual+1,
 manual_displacement_px_per_pair=truth,manual_velocity_assumed_mps=truth*scale*np.array([1,-1]),
 model_at_manual_raw_px_per_pair=d[:200],manual_model_accepted=ok[:200].astype(np.uint8),
 manual_error_px=e,manual_prediction_change_px=z['manual_change'],
 supplied_PIV_at_manual_px_per_pair=z['supplied_PIV_at_manual'],supplied_PIV_error_px=z['supplied_PIV_error'],
 dense_x_full_zero_based_px=xx,dense_y_full_zero_based_px=yy,
 dense_displacement_conservative_px_per_pair=dense_u,dense_d_dx_dx_conservative=dense_g,
 dense_accepted=ok[ds].reshape(shape).astype(np.uint8),dense_horizontal_gradient_accepted=valid.astype(np.uint8),
 dense_velocity_conservative_assumed_mps=dense_u*scale*np.array([1,-1]),
 dense_dU_dX_conservative_assumed_per_s=dense_g/DT,
 surface_x_full_zero_based=np.arange(2048),surface_a_y_full_zero_based=sa,surface_b_y_full_zero_based=z['full_surface_b'],
 automatic_track_xy_full_zero_based=tr['points']+origin,automatic_displacement_px_per_pair=tr['disp'],
 automatic_track_accepted=tr['accepted'].astype(np.uint8),
 DX_from_supplied_MAT=DX,DT_from_supplied_MAT=DT,physical_units_confirmed=np.uint8(0),
 units_note='Pixels are primary. Assumed physical fields require DX in metres/pixel and DT in seconds; labels were absent in supplied MAT. U right and V up; raw dy is down.',
 gradient_note='G[i,j] differentiates displacement component i with respect to A image coordinate j. Only G[0,0] has the gradient sensitivity screen. Finite-time displacement gradient/DT is a velocity proxy at A, not a directly measured instantaneous Eulerian derivative.',
 coordinate_note='Coordinates refer to larger images. Add1 for MATLAB indexing. Original crop origin in full zero-based image is [299,339].',
 method_note='Original masked quadratic fit with bilinear sampling, radius13, robust reg0.00005, automatic PTV seed, C2 blend radius16; old727 patch fits frozen; original image-only acceptance thresholds.',
 data_note='Use conservative arrays (rejected values are NaN) for plotting. Raw estimates are included for diagnostics; dense4px sampling is not4px spatial resolution.')
for k in diag:mat['grid_'+k]=z[k][gs]
savemat(O/(prefix+'field.mat'),mat,do_compression=True)
np.savez_compressed(O/(prefix+'field.npz'),**mat)
(O/(prefix+'summary.json')).write_text(json.dumps(s,indent=2))
print('Exported five figure sets, two CSV tables, screened MATLAB/NumPy data and summary.',flush=True)
