"""Export the tested cubic-sampling/quadratic-deformation refinement.
Only the explicitly named improved_* deliverables are written to outputs/.
"""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;WORK=ROOT/'work';OUT=ROOT/'outputs'
os.environ['MPLCONFIGDIR']=str(HERE/'mplcache')
import json,numpy as np
from scipy.io import savemat,loadmat
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
sys.path.insert(0,str(WORK));from field_model import LocalField
z=np.load(HERE/'forward_cubic_quality.npz');old=np.load(WORK/'final_results.npz');local=np.load(HERE/'forward_cubic_complete.npz');meta=np.load(WORK/'metadata.npz')
q=z['query'];d=z['disp'];g=z['gradient'];truth=old['manual_truth'];N=len(q)-200;ids=slice(200,None);sa=old['surface_a'];sb=old['surface_b'];dep=old['depth'][:len(q)];dest=q+d;targetdep=dest[:,1]-np.interp(dest[:,0],np.arange(len(sb)),sb)
frozen=old['accepted'][:len(q)];fresh=z['accepted'];frozen_g=old['gradient_accepted'][:len(q)];fresh_g=z['gradient_accepted'];err=np.linalg.norm(d[:200]-truth,axis=1)
checks=json.loads((HERE/'quality_metrics.json').read_text())
units=('A-to-B finite-time displacement in pixels per image pair. Source-A coordinates are zero-based pixel centers: x increases rightward, y downward. '+
       'gradient[n,component,axis] has component order [dx,dy_down] and axis order [x,y]. Displacement gradients are pixel/pixel, dimensionless per pair. '+
       'For isotropic length_per_pixel L and frame_interval_seconds dt: U=L*dx/dt; V_up=-L*dy_down/dt; dU/dX=gradient[:,0,0]/dt. '+
       'L and dt are unknown and marked NaN. These finite-time estimates are not instantaneous Eulerian velocities.')
notes=('The spatial deformation remains quadratic (six coefficients per component). The improvement is cubic image interpolation with analytic derivatives of that same interpolant. '+
       'Raw estimates are retained even where masks are false. Original frozen masks and new provisional masks are separate. '+
       'New checks use fresh cubic backward registration and the three frozen original-method alternatives; they are not a complete new set of cubic window/mask sensitivity runs. '+
       'Manual endpoints were excluded from fits; this pair has already been used for exploratory model selection and is not an untouched validation set.')
mat=dict(
 manual_xy_zero_based=q[:200],manual_xy_matlab=q[:200]+1,manual_displacement_px_per_pair=truth,
 predicted_at_manual_px_per_pair=d[:200],manual_displacement_gradient=g[:200],manual_endpoint_disagreement_px=err,
 manual_original_displacement_px_per_pair=old['disp'][:200],manual_original_endpoint_disagreement_px=old['manual_error'],
 manual_original_frozen_accepted=frozen[:200].astype(np.uint8),manual_new_provisional_accepted=fresh[:200].astype(np.uint8),
 manual_original_frozen_gradient_accepted=frozen_g[:200].astype(np.uint8),manual_new_provisional_gradient_accepted=fresh_g[:200].astype(np.uint8),
 grid_xy_zero_based=q[ids],grid_xy_matlab=q[ids]+1,grid_displacement_px_per_pair=d[ids],grid_displacement_gradient=g[ids],
 grid_original_frozen_accepted=frozen[ids].astype(np.uint8),grid_new_provisional_accepted=fresh[ids].astype(np.uint8),
 grid_original_frozen_gradient_accepted=frozen_g[ids].astype(np.uint8),grid_new_provisional_gradient_accepted=fresh_g[ids].astype(np.uint8),
 query_xy_zero_based=q,query_displacement_px_per_pair=d,query_displacement_gradient=g,
 query_original_frozen_accepted=frozen.astype(np.uint8),query_new_provisional_accepted=fresh.astype(np.uint8),
 query_original_frozen_gradient_accepted=frozen_g.astype(np.uint8),query_new_provisional_gradient_accepted=fresh_g.astype(np.uint8),
 query_ncc=z['ncc'],query_forward_backward_error_px=z['fb'],query_method_spread_px=z['method_spread'],query_gradient_sensitivity=z['gradient_spread'],
 query_local_valid_share=z['local_valid_share'],query_reverse_valid_share=z['reverse_valid_share'],query_depth_below_A_surface_px=dep,query_target_depth_below_B_surface_px=targetdep,
 query_reference_alternatives_available=old['alternatives_available'][:len(q)].astype(np.uint8),query_nearest_automatic_particle_px=old['nearest_feature'][:len(q)],
 query_feature_count_radius25=old['feature_count25'][:len(q)],query_mapping_determinant=np.linalg.det(np.eye(2)+g),
 local_model_centers_zero_based=local['points'],local_quadratic_coefficients=local['params'],local_minimum_mapping_determinant=local['mindet'],
 image_patch_radius_pixels=13.,compact_blend_radius_pixels=16.,surface_a_y_zero_based=sa,surface_b_y_zero_based=sb,
 frame_interval_seconds=np.nan,length_per_pixel=np.nan,units_note=units,method_and_screen_note=notes,
 query_row_note='MATLAB rows 1:200 are manual source positions; rows 201:927 are the 727 image-grid positions. No dense display grid is included.',
 polynomial_basis_note='Each local displacement component uses [1, qx, qy, 0.5*qx^2, qx*qy, 0.5*qy^2], q=(source-center)/13; local models are blended by normalized w(t)=(1-t)^4*(1+4*t), t=distance/16, for distance<16.',
 original_frozen_manual_count=int(frozen[:200].sum()),new_provisional_manual_count=int(fresh[:200].sum()),
 original_frozen_grid_count=int(frozen[ids].sum()),new_provisional_grid_count=int(fresh[ids].sum()),new_provisional_gradient_grid_count=int(fresh_g[ids].sum()),
 all_manual_mean_epe_px=float(err.mean()),all_manual_median_epe_px=float(np.median(err)),all_manual_rms_epe_px=float(np.sqrt(np.mean(err**2))),
 near40_manual_count=int(sum(dep[:200]<40)),near40_manual_mean_epe_px=float(err[dep[:200]<40].mean()),
 spatial_calibration_known=np.uint8(0),frame_interval_known=np.uint8(0),untouched_validation_set=np.uint8(0),
 provisional_thresholds=dict(source_depth_min_px=12.,source_depth_max_px=150.,target_depth_min_px=10.,nearest_particle_max_px=12.,feature_count_radius25_min=6.,ncc_min=.6,fb_max_px=1.,method_spread_max_px=1.5,mapping_determinant_min=.2,local_valid_share_min=.95,reverse_valid_share_min=.95,all_reference_alternatives_required=np.uint8(1),gradient_depth_min_px=20.,gradient_method_spread_max=.08,gradient_nearest_particle_max_px=10.)
)
mid_g=np.full_like(g,np.nan)
for i,gi in enumerate(g):
 if np.all(np.isfinite(gi)) and abs(np.linalg.det(np.eye(2)+gi/2))>1e-4:mid_g[i]=gi@np.linalg.inv(np.eye(2)+gi/2)
mat.update(query_midpoint_xy_zero_based=q+d/2,query_midpoint_displacement_gradient=mid_g)
path=OUT/'improved_quadratic_velocity.mat';savemat(path,mat,do_compression=True,long_field_names=True)

# Matched-position comparison with one shared displacement scale.
A=np.asarray(Image.open('data/ExpLCL_1_03-123_imgA.tif'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
INK='#17263b';CYAN='#25c4dc';ORANGE='#ed9a36';GRAY='#b4bac5';PURPLE='#946ca6'
fig,ax=plt.subplots(1,2,figsize=(15,6.1));fig.subplots_adjust(left=.07,right=.97,top=.79,bottom=.26,wspace=.12)
fig.suptitle('Quadratic deformation refined with cubic image sampling',x=.07,y=.985,ha='left',fontsize=19,fontweight='bold',color=INK)
fig.text(.07,.915,'A → B displacement  |  pixels per image pair  |  same 200 positions and true arrow scale',fontsize=11,color='#465365')
for a in ax:
 a.imshow(A,cmap='gray',vmin=35,vmax=180,origin='upper',extent=(-.5,500.5,500.5,-.5))
 a.fill_between(np.arange(501),sa,sa+10,color=PURPLE,alpha=.42,lw=0)
 a.plot(np.arange(501),sa,color='#f4d9ed',lw=1.15)
 a.set(xlim=(60,390),ylim=(180,0),xlabel='Horizontal image coordinate x (px)')
 a.set_aspect('equal');a.set_xticks([60,120,180,240,300,360]);a.set_yticks([0,30,60,90,120,150,180]);a.tick_params(colors=INK)
ax[0].set_ylabel('Image y (px, downward)');ax[1].set_ylabel('')
ax[0].set_title('Manual picks · 200 vectors',loc='left',fontweight='bold',pad=12)
ax[1].set_title('Quadratic model · cubic image interpolation',loc='left',fontweight='bold',pad=12)
def quiv(a,xy,uv,col):
 return a.quiver(xy[:,0],xy[:,1],uv[:,0],uv[:,1],angles='xy',scale_units='xy',scale=1,color=col,width=.003,headwidth=3.8,headlength=4.5,headaxislength=4.1,pivot='tail',minlength=.15)
quiv(ax[0],q[:200],truth,ORANGE)
quiv(ax[1],q[:200][~fresh[:200]],d[:200][~fresh[:200]],GRAY)
v=quiv(ax[1],q[:200][fresh[:200]],d[:200][fresh[:200]],CYAN)
ax[1].quiverkey(v,.72,-.23,10,'10 px / pair',labelpos='E',coordinates='axes',color=INK)
fig.text(.07,.18,'Mean disagreement: 0.478 px overall; 0.636 px for 26 manual picks within 40 px of the surface.',fontsize=11,color=INK)
fig.text(.07,.115,'Cyan: passes the new provisional screen (191/200). Gray: flagged. Purple: first 10 px below the surface trace.',fontsize=10,color='#536170')
fig.text(.07,.065,'The original screen is preserved separately in the MATLAB file. Calibration and frame interval remain unknown.',fontsize=10,color='#536170')
for suffix in ['png','svg']:fig.savefig(OUT/('improved_quadratic_quiver.'+suffix),dpi=250,bbox_inches='tight')
plt.close(fig)

# Independent read-back against source arrays and a fresh field evaluation.
loaded=loadmat(path);qa=[]
def check(name,a,b):
 aa=np.asarray(a);bb=np.asarray(b);passed=aa.shape==bb.shape and np.allclose(aa,bb,rtol=0,atol=1e-12,equal_nan=True)
 qa.append(dict(check=name,passed=bool(passed)))
for key,value in mat.items():
 if isinstance(value,np.ndarray) and value.dtype.kind in 'fbiu':
  vv=loaded[key]
  if value.ndim==1:vv=vv.ravel()
  check(key,vv,value)
check('MATLAB manual indexing matches original MAT',loaded['manual_xy_matlab'],meta['p'][:,0].T)
check('Manual displacement matches original MAT',loaded['manual_displacement_px_per_pair'],meta['p'][:,1].T-meta['p'][:,0].T)
pred,grad=LocalField(HERE/'forward_cubic_complete.npz').evaluate(q)
check('Exported displacement matches freshly evaluated field',loaded['query_displacement_px_per_pair'],pred)
check('Exported gradient matches freshly evaluated field',loaded['query_displacement_gradient'],grad)
check('Original frozen mask stays distinct and unchanged',loaded['query_original_frozen_accepted'].ravel(),old['accepted'][:927])
check('New provisional mask matches cubic quality check',loaded['query_new_provisional_accepted'].ravel(),z['accepted'])
check('Frame interval unknown',np.isnan(loaded['frame_interval_seconds']),np.ones((1,1),bool))
check('Length calibration unknown',np.isnan(loaded['length_per_pixel']),np.ones((1,1),bool))
check('New gradient mask is a subset of new displacement mask',np.all(~fresh_g|fresh),True)
result=dict(checks=len(qa),passed=sum(v['passed'] for v in qa),failures=[v['check'] for v in qa if not v['passed']],mat_fields=sorted(mat),manual_rows=200,grid_rows=N,query_rows=len(q),original_frozen_manual_pass=int(frozen[:200].sum()),new_provisional_manual_pass=int(fresh[:200].sum()),original_frozen_grid_pass=int(frozen[ids].sum()),new_provisional_grid_pass=int(fresh[ids].sum()),new_provisional_gradient_grid_pass=int(fresh_g[ids].sum()),details=qa)
(HERE/'improved_export_qa.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['details','mat_fields']},indent=2))
print('MAT fields:',', '.join(sorted(mat)))
