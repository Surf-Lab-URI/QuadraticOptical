import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json
from scipy.spatial import Delaunay
from ptv_model import PTVModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

z=np.load('work/ptv_tracks.npz');m=np.load('work/metadata.npz');s=m['surfa'].ravel()-1
p=z['points'];d=p[:,1]-np.interp(p[:,0],np.arange(501),s);accepted=z['accepted']
f=np.load('work/ptv_quad10.npz');pts=f['points'];n=int(f['grid_count']);hull=Delaunay(p[accepted]);inside=hull.find_simplex(pts)>=0
supported=inside&(f['nearest_feature']<=10)&(f['count']>=6)&(f['rms']<1.5)
np.savez('work/ptv_support.npz',points=pts,inside_feature_hull=inside,supported=supported,grid_count=n)
model=PTVModel();new=model.evaluate(pts)
assert np.max(abs(new['disp']-f['disp']))<1e-10
g1=model.gradient(pts,10,1);g2=model.gradient(pts,10,2);g14=model.gradient(pts,14,1)
np.savez('work/ptv_gradients.npz',points=pts,gradient_fd_h1=g1,gradient_fd_h2=g2,gradient_fd_scale14=g14,local_polynomial_gradient=f['gradient'],supported=supported,grid_count=n,scale=10)

raw=np.array(Image.open('data/ExpLCL_1_03-123_imgA.tif'))
fig,axs=plt.subplots(2,1,figsize=(13,9),constrained_layout=True)
for ax in axs:
    ax.imshow(raw,cmap='gray',vmin=30,vmax=230)
    ax.plot(np.arange(501),s,color='#00ffff',lw=1)
    ax.set(xlim=(60,390),ylim=(180,0),xlabel='Zero-based image x (px)',ylabel='Zero-based image y, down (px)')
q=axs[0].scatter(p[accepted,0],p[accepted,1],c=z['disp'][accepted,0],cmap='viridis',vmin=-2,vmax=26,s=8)
axs[0].set_title(f'{accepted.sum()} reciprocal particle tracks from {len(p)} automatic intensity peaks')
fig.colorbar(q,ax=axs[0],label='Tracked horizontal displacement (px/pair)',shrink=.8)
ok=supported[:n]
q=axs[1].quiver(pts[:n][ok,0],pts[:n][ok,1],f['disp'][:n][ok,0],f['disp'][:n][ok,1],color='#ffd663',angles='xy',scale_units='xy',scale=1,width=.0018)
axs[1].quiverkey(q,.84,.93,10,'10 px/pair',coordinates='axes',labelcolor='white')
axs[1].set_title('Robust local quadratic field (scale10px); vectors passing geometry/support diagnostics')
fig.savefig('work/ptv_diagnostics.png',dpi=170)
plt.close(fig)

counts={f'{a}–{b}':int(np.sum(accepted&(d>=a)&(d<b))) for a,b in [(14,20),(20,30),(30,50),(50,100),(100,170)]}
met=json.load(open('work/ptv_metrics.json'))
report=f'''# Automatic particle-tracking and local polynomial field

This is an image-derived hybrid particle-tracking model. It detects particles automatically and uses robust affine image-registration priors to guide correspondence, then reconstructs a spatial field from accepted tracks. No manual source points or displacements enter detection, tracking, filtering, or spatial regression. The manual data are accessed only afterward for evaluation. Comparing the three spatial scales against those same labels is post hoc model selection, not an independent test set.

## Detection and matching

The detector identifies local maxima (5×5 neighborhood) of Gaussian(σ0.6px)−Gaussian(σ2px) intensity, with peak response>8 gray levels and Gaussian(σ5px) background<180. Candidates lie in x60…390, y≤175 and depth≥14px below the saved surface converted to zero-based coordinates. There are {len(p)} automatically detected features.

Matching uses 9×9 patches of raw intensities smoothed by σ0.5px, with per-patch mean/variance normalization. A and B validity require depth≥10px; at least85% of patch samples must be valid in both frames. The correspondence search covers a17×17 displacement grid around a robust prior assembled from12 nearby affine image nodes with NCC>0.7. It then refines two candidate peaks continuously with Nelder–Mead. The optimized score is1−NCC+0.005|displacement−prior|². Consequently the prior can bias a weak patch match; high NCC alone does not prove accuracy.

Backward matching starts at the forward endpoint and uses independently assembled inverse-affine priors. Accepted tracks require forward/back NCC>0.68, forward/back closure<1px, and forward/back penalized-objective ambiguity gaps>0.015/>0.01. Runner-ups must be outside2.5px of the winning displacement; if refinements merge, the best distinct coarse-search candidate is used. These are practical diagnostics, not calibrated probabilities.

{accepted.sum()} tracks pass. Their minimum surface depth is{d[accepted].min():.3f}px, median forward NCC is{np.median(z['ncc'][accepted]):.3f}, median reciprocal error is{np.median(z['fb'][accepted]):.3f}px, and90th-percentile reciprocal error is{np.percentile(z['fb'][accepted],90):.3f}px. Counts by depth:{counts}.

## Field and gradient definitions

At each query, take at most45 nearest accepted tracks, truncated at the larger of2×scale and the12th-nearest distance. Fit basis[1,qx,qy,0.5qx²,qxqy,0.5qy²], q=(particle−query)/scale. Base weights are Gaussian(distance/scale) times clipped NCC quality, clipped ambiguity-gap quality, and exp(−reciprocal_error²/0.5). Seven reweighting iterations use1/(1+(vector_residual/(2s))⁴), with s=max(0.45,1.4826×median vector residual). Curvature coefficients have ridge0.04; slopes have ridge1e−5; intercept ridge1e−8. The field prediction is the fitted intercept. Spatial scales10,14,18px provide a sensitivity comparison; support diameter is usually up to4×scale and can increase where particles are sparse.

The gradient in ptv_quad*.npz is the derivative of the locally fitted polynomial at its center, i.e. slope coefficients divided by scale. It is not the exact derivative of the full moving-regression predictor, because neighbors and weights change with query position. ptv_gradients.npz separately stores central finite differences of that moving predictor with steps±1px and±2px, plus the scale14px result. gradient[velocity_component,spatial_direction] uses component/direction0=x and1=y_down. Thus gradient[:,0,0] is horizontal displacement gradient ∂d_x/∂x, per image pair. Physical velocity gradient equals this value divided by interframe time for isotropic calibration. Finite-difference step or smoothing-scale sensitivity is a stability diagnostic, not a confidence interval. Neither gradient formulation has manual derivative ground truth.

Support flags in ptv_support.npz require inclusion in the accepted-feature convex hull, nearest feature≤10px, at least6 regression weights>0.1, and weighted fit RMS<1.5px. They retain{supported[:n].sum()} of{n} grid nodes and{supported[n:].sum()} of200 manual query locations. A global convex hull is only a coarse support test; use actual surface masks and local texture diagnostics as well, especially around the depression.

## Validation (pixels per pair)

Scale10: all200 median endpoint error{met['ptv_quad10']['all']['median']:.3f}, mean{met['ptv_quad10']['all']['mean']:.3f},90th percentile{met['ptv_quad10']['all']['p90']:.3f}; depth<40px (26 picks) median{met['ptv_quad10']['near40']['median']:.3f}, mean{met['ptv_quad10']['near40']['mean']:.3f},90th percentile{met['ptv_quad10']['near40']['p90']:.3f}. Source-point geometry is converted with p−1. Other scale metrics are in ptv_metrics.json. Scale10 best overall on this comparison, while scale14 has a lower near-surface median; those differences should be presented as model sensitivity rather than an independently validated choice.

Files: ptv_tracks.npz(all and accepted automatic tracks and diagnostics), ptv_quad10/14/18.npz(field predictions and local polynomial slopes), ptv_support.npz(geometric support flags), ptv_gradients.npz(finite-difference derivatives and sensitivity), ptv_model.py(import-safe reusable evaluator), ptv_refine.py(reproducible detector/tracker), ptv_diagnostics.png(visual field audit).
'''
open('work/ptv_report.md','w').write(report)
print('accepted',accepted.sum(),'supported grid',supported[:n].sum(),'manual',supported[n:].sum())
print('gradient difference h1-h2 supported quantiles',np.quantile(abs(g1[supported,0,0]-g2[supported,0,0]),[.5,.9,1]))
print('gradient difference local-fd supported quantiles',np.quantile(abs(g1[supported,0,0]-f['gradient'][supported,0,0]),[.5,.9,1]))
