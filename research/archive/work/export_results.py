import os
os.environ['MPLCONFIGDIR']=os.path.abspath('work/mplcache')
import numpy as np,json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import savemat
from PIL import Image
from pathlib import Path
O=Path('outputs');O.mkdir(exist_ok=True);z=np.load('work/final_results.npz');summary=json.load(open('work/final_summary.json'));meta=np.load('work/metadata.npz')
A=np.asarray(Image.open('data/ExpLCL_1_03-123_imgA.tif'))
q=z['query'];d=z['disp'];g=z['gradient'];ok=z['accepted'];gok=z['gradient_accepted'];N=int(z['grid_count']);manual=q[:200];truth=z['manual_truth'];sa=z['surface_a'];sb=z['surface_b'];xs=np.arange(501);sl=slice(200,200+N);ds=slice(200+N,None)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.labelsize':10,'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
ORANGE='#ed9a36';CYAN='#25c4dc';GRAY='#b4bac5';INK='#17263b';PURPLE='#946ca6'

def background(ax,lim=(60,390,180,0),image=True):
 if image:ax.imshow(A,cmap='gray',vmin=35,vmax=180,origin='upper',extent=(-.5,500.5,500.5,-.5))
 ax.fill_between(xs,sa,sa+10,color=PURPLE,alpha=.42,lw=0)
 ax.plot(xs,sa,color='#f4d9ed' if image else '#79557c',lw=1.2)
 ax.set_xlim(lim[0],lim[1]);ax.set_ylim(lim[2],lim[3]);ax.set_aspect('equal')
 ax.set_xlabel('Horizontal image coordinate x (px)');ax.set_ylabel('Image y (px, downward)')
 ax.set_xticks([60,120,180,240,300,360]);ax.tick_params(colors=INK)

def quiv(ax,pts,vec,color,width=.003,alpha=1):
 return ax.quiver(pts[:,0],pts[:,1],vec[:,0],vec[:,1],angles='xy',scale_units='xy',scale=1,color=color,width=width,headwidth=3.8,headlength=4.5,headaxislength=4.1,pivot='tail',alpha=alpha,minlength=.15)

def save(fig,name):
 fig.savefig(O/(name+'.png'),dpi=250,bbox_inches='tight');fig.savefig(O/(name+'.svg'),bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(14,5.2));fig.subplots_adjust(top=.80,bottom=.20,wspace=.12)
fig.suptitle('Particle motion around the capillary depression',x=.08,y=.99,ha='left',fontsize=19,fontweight='bold',color=INK)
fig.text(.08,.925,'A → B displacement  |  pixels per image pair  |  arrows shown at true displacement scale',color='#465365',fontsize=11)
for a in ax:background(a)
ax[0].set_title('Manual picks · 200 vectors',loc='left',fontweight='bold',pad=12);v=quiv(ax[0],manual,truth,ORANGE)
ax[1].set_title('Hybrid quadratic optical flow · same 200 positions',loc='left',fontweight='bold',pad=12)
quiv(ax[1],manual[~ok[:200]],d[:200][~ok[:200]],GRAY,width=.003,alpha=.85);quiv(ax[1],manual[ok[:200]],d[:200][ok[:200]],CYAN)
ax[1].set_ylabel('')
fig.text(.08,.12,'Mean disagreement: {:.3f} px overall; {:.3f} px for the 26 picks within 40 px of the surface.'.format(summary['metrics']['all']['mean'],summary['metrics']['near40']['mean']),color=INK,fontsize=11)
fig.text(.08,.065,'Purple band: first 10 px below the trace (glare). Gray predicted arrows fail the provisional image-only screen.',color='#536170',fontsize=10)
save(fig,'quiver_comparison')

fig,ax=plt.subplots(figsize=(12,6.7));fig.subplots_adjust(top=.88,bottom=.16)
background(ax,(60,380,170,15));ax.set_title('Manual and predicted displacements at identical locations',loc='left',fontweight='bold',fontsize=16,pad=15)
quiv(ax,manual,truth,ORANGE,width=.0039,alpha=.95);quiv(ax,manual[ok[:200]],d[:200][ok[:200]],CYAN,width=.0022);quiv(ax,manual[~ok[:200]],d[:200][~ok[:200]],GRAY,width=.0022)
leg=ax.legend(handles=[Line2D([0],[0],color=ORANGE,lw=3,label='Manual'),Line2D([0],[0],color=CYAN,lw=3,label='Model, passes screen'),Line2D([0],[0],color=GRAY,lw=3,label='Model, flagged')],loc='upper right',facecolor='#15202e',framealpha=.9)
plt.setp(leg.get_texts(),color='white')
fig.text(.125,.065,'True displacement scale. Coordinates are zero-based image pixels; positive vertical motion points downward.',fontsize=10,color='#536170')
save(fig,'quiver_overlay')

fig,ax=plt.subplots(figsize=(12,7));fig.subplots_adjust(top=.88,bottom=.16);background(ax)
a=ok[sl];pts=q[sl];uv=d[sl];ax.scatter(pts[~a,0],pts[~a,1],s=6,c='#cf8895',marker='x',alpha=.65,lw=.6)
v=quiv(ax,pts[a],uv[a],CYAN,width=.0024)
ax.set_title('Predicted field with image-based support checks',loc='left',fontweight='bold',fontsize=16,pad=15)
fig.text(.125,.07,f'{a.sum()} / {len(a)} grid nodes pass; crosses flag rejected estimates. Arrows show displacement at true scale.',fontsize=10,color='#536170')
save(fig,'predicted_field')

# The analytic derivative differentiates the same smoothly blended displacement field.
shape=z['dense_x'].shape;xx=z['dense_x'];yy=z['dense_y'];xe,ye=np.meshgrid(np.r_[xx[0]-1,xx[0,-1]+1],np.r_[yy[:,0]-1,yy[-1,0]+1]);gm=g[ds,0,0].reshape(shape);gs=z['gradient_spread'][ds].reshape(shape);valid=gok[ds].reshape(shape)
fig,axs=plt.subplots(1,2,figsize=(14,5.2));fig.subplots_adjust(top=.80,bottom=.19,wspace=.26)
fig.suptitle('Horizontal gradient and sensitivity',x=.08,y=.99,ha='left',fontsize=19,fontweight='bold',color=INK)
fig.text(.08,.925,'∂dₓ/∂x at fixed image height  |  divide by the frame interval Δt for a velocity gradient in s⁻¹',fontsize=11,color='#465365')
for a in axs:
 a.set_facecolor('#e5e8eb');background(a,image=False)
masked=np.ma.array(gm,mask=~valid);im=axs[0].pcolormesh(xe,ye,masked,cmap='RdBu_r',vmin=-.25,vmax=.25,shading='flat',rasterized=True)
axs[0].set_title('Quadratic model · supported locations',loc='left',fontweight='bold',pad=12)
cb=fig.colorbar(im,ax=axs[0],shrink=.8,pad=.02);cb.set_label('Displacement gradient (px/px)')
im=axs[1].pcolormesh(xe,ye,np.ma.array(gs,mask=~valid),cmap='magma',vmin=0,vmax=.08,shading='flat',rasterized=True)
axs[1].set_title('Maximum change across three variants',loc='left',fontweight='bold',pad=12);axs[1].set_ylabel('');cb=fig.colorbar(im,ax=axs[1],shrink=.8,pad=.02);cb.set_label('Gradient sensitivity (px/px)')
fig.text(.08,.105,'Variants: affine fit, larger quadratic window, and a 14 px surface exclusion. Sensitivity is not a confidence interval.',fontsize=10,color='#536170')
fig.text(.08,.05,'Gray regions lack support or exceed the sensitivity threshold. No gradient is reported within 20 px of the surface.',fontsize=10,color='#536170')
save(fig,'horizontal_gradient')

fig,axs=plt.subplots(1,2,figsize=(12,4.8));fig.subplots_adjust(top=.84,bottom=.18,wspace=.30)
fig.suptitle('Comparison with all 200 manual particle matches',x=.08,y=.99,ha='left',fontsize=17,fontweight='bold',color=INK)
e=z['manual_error'];dep=z['depth'][:200]
axs[0].scatter(dep[ok[:200]],e[ok[:200]],s=19,color='#168daa',alpha=.8,label='Passes screen');axs[0].scatter(dep[~ok[:200]],e[~ok[:200]],s=35,color='#ab4960',marker='x',label='Flagged')
axs[0].axhline(1,color='#8c99a4',ls='--',lw=1);axs[0].axvspan(0,16.71,color=PURPLE,alpha=.12);axs[0].set(xlabel='Depth below A surface (px)',ylabel='Endpoint disagreement (px)',xlim=(0,150),ylim=(0,max(3,e.max()*1.1)));axs[0].legend(fontsize=9);axs[0].grid(alpha=.15)
base=np.load('work/baseline_reference_matlab.npz')['error'];ncc=np.load('work/baseline_ncc.npz');from scipy.interpolate import LinearNDInterpolator
ncce=np.linalg.norm(LinearNDInterpolator(ncc['points'],ncc['disp'])(manual)-truth,axis=1)
for err,label,col in [(ncce,'Translation-only NCC','#798693'),(base,'High-pass DIS','#aa7451'),(e,'Hybrid quadratic','#168daa')]:
 vals=np.sort(err);axs[1].plot(vals,np.arange(1,201)/200,label=label,color=col,lw=2)
axs[1].set(xlabel='Endpoint disagreement (px)',ylabel='Fraction of 200 picks',xlim=(0,5),ylim=(0,1.02));axs[1].legend(fontsize=9,loc='lower right');axs[1].grid(alpha=.15)
fig.text(.08,.06,'Manual endpoints did not enter model fitting. Method choices were compared on this pair; this is not an untouched test set.',fontsize=10,color='#536170')
save(fig,'validation')

# Numeric deliverables: keep finite raw estimates and explicit masks, rather than silently dropping points.
keys=['ncc','fb','method_spread','gradient_spread','depth','target_depth','nearest_feature','feature_count25','determinant','alternatives_available','local_valid_share','reverse_valid_share','accepted','gradient_accepted']
def writecsv(path,indices,ismanual=False):
 with open(path,'w',newline='') as f:
  wr=csv.writer(f);header=['pick' if ismanual else 'node','x_px','y_px','dx_px_per_pair','dy_down_px_per_pair','d_dx_dx','d_dx_dy','d_dy_dx','d_dy_dy']+keys
  if ismanual:header+=['manual_dx','manual_dy_down','endpoint_disagreement_px']
  wr.writerow(header)
  for row,i in enumerate(indices):
   vals=[row+1,*q[i],*d[i],*g[i].ravel()]+[int(z[k][i]) if z[k].dtype.kind in 'biu' else float(z[k][i]) for k in keys]
   if ismanual:vals += [*truth[i],float(e[i])]
   wr.writerow(vals)
writecsv(O/'manual_comparison.csv',range(200),True);writecsv(O/'predicted_grid.csv',range(200,200+N))
mat={'manual_xy_zero_based':manual,'manual_xy_matlab':manual+1,'manual_displacement_px_per_pair':truth,'predicted_at_manual_px_per_pair':d[:200],'manual_disagreement_px':e,'manual_prediction_accepted':ok[:200].astype(np.uint8),'grid_xy_zero_based':q[sl],'grid_displacement_px_per_pair':d[sl],'grid_displacement_gradient':g[sl],'grid_accepted':ok[sl].astype(np.uint8),'grid_gradient_accepted':gok[sl].astype(np.uint8),'grid_forward_backward_error_px':z['fb'][sl],'grid_method_spread_px':z['method_spread'][sl],'grid_gradient_sensitivity':z['gradient_spread'][sl],'dense_x_px':xx,'dense_y_px':yy,'dense_dx_px_per_pair':d[ds,0].reshape(shape),'dense_dy_down_px_per_pair':d[ds,1].reshape(shape),'dense_d_dx_dx':gm,'dense_accepted':ok[ds].reshape(shape).astype(np.uint8),'dense_gradient_accepted':valid.astype(np.uint8),'surface_a_y_zero_based':sa,'surface_b_y_zero_based':sb,'frame_interval_seconds':np.nan,'length_per_pixel':np.nan,'units_note':'A to B, pixels per image pair; x right, y down; derivatives of finite-time displacement with respect to A coordinates; multiply displacements by length_per_pixel/frame_interval_seconds; horizontal physical gradient dU/dX=d_dx_dx/frame_interval_seconds','model_note':'automatic image particle prior plus masked quadratic registration; compact C2 blend radius16 px; image patch radius13 px'}
# Midpoint remapping: a derived finite-time representation, not an instantaneous velocity measurement.
gmid=np.full_like(g[sl],np.nan)
for i,gi in enumerate(g[sl]):
 if np.all(np.isfinite(gi)) and abs(np.linalg.det(np.eye(2)+gi/2))>1e-4:gmid[i]=gi.dot(np.linalg.inv(np.eye(2)+gi/2))
mat.update(grid_midpoint_xy=q[sl]+d[sl]/2,grid_midpoint_displacement_gradient=gmid)
savemat(O/'velocity_results.mat',mat,do_compression=True)
np.savez_compressed(O/'velocity_results.npz',**{k:z[k] for k in z.files},frame_interval_seconds=np.nan,length_per_pixel=np.nan)
(O/'validation_summary.json').write_text(json.dumps(summary,indent=2))
print('Exported figures, tables and MATLAB/NumPy fields.',flush=True)

tracks=np.load('work/ptv_tracks.npz');tp=tracks['points'];td=tracks['disp'];tdepth=tp[:,1]-np.interp(tp[:,0],xs,sa);select=tracks['accepted']&(tdepth<40)
fig,ax=plt.subplots(figsize=(12,5.4));fig.subplots_adjust(top=.86,bottom=.23)
background(ax,(60,390,100,0));ax.set_title('Closest automatic particle tracks · provisional correspondences',loc='left',fontweight='bold',fontsize=15,pad=15)
quiv(ax,tp[select],td[select],CYAN,width=.0027)
ax.scatter(tp[select,0],tp[select,1],s=7,color=CYAN)
fig.text(.125,.12,f'{select.sum()} reciprocal tracks within 40 px of the surface; nearest source depth {tdepth[select].min():.2f} px. True displacement scale.',fontsize=10,color=INK)
fig.text(.125,.065,'These sparse candidates extend beyond the screened continuous field. No manual validation exists above 16.71 px depth.',fontsize=10,color='#536170')
save(fig,'near_surface_candidate_tracks')
with open(O/'automatic_particle_tracks.csv','w',newline='') as f:
 wr=csv.writer(f);wr.writerow(['feature','x_px','y_px','dx_px_per_pair','dy_down_px_per_pair','depth_px','ncc','back_ncc','forward_backward_error_px','ambiguity_gap','accepted'])
 for i in range(len(tp)):wr.writerow([i+1,*tp[i],*td[i],tdepth[i],tracks['ncc'][i],tracks['back_ncc'][i],tracks['fb'][i],tracks['ambiguity_gap'][i],int(tracks['accepted'][i])])
mat.update(automatic_feature_xy=tp,automatic_track_displacement=td,automatic_track_accepted=tracks['accepted'].astype(np.uint8),automatic_track_fb_error=tracks['fb'],automatic_track_ncc=tracks['ncc'])
savemat(O/'velocity_results.mat',mat,do_compression=True)
print('Exported provisional near-surface particle tracks.',flush=True)
