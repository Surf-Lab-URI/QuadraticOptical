from pathlib import Path
import csv
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates, gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('historical-user-files/Documents/Codex/2026-09-11/i-a')
SRC = Path('historical-user-files/Downloads')
z = np.load(ROOT/'work/metadata.npz')
imgs = {k: np.array(Image.open(SRC/f'ExpLCL_1_03-123_{k}.tif')) for k in ['imgA','imgB','imgA_mask','imgA_altmask','imgB_mask','imgB_altmask']}
p_saved=z['p']
p = p_saved-1; xy = p[:,0].T; uv = (p[:,1]-p[:,0]).T
x = np.arange(501)
sa = z['surfa'].ravel()-1; sb = z['surfb'].ravel()-1
da = xy[:,1]-np.interp(xy[:,0],x,sa)
db = p[1,1]-np.interp(p[0,1],x,sb)
dy,dx = np.mgrid[-7:8,-7:8]
corr = []
for i in range(200):
    patches = [map_coordinates(imgs['img'+f].astype(float), [dy.ravel()+p[1,j,i],dx.ravel()+p[0,j,i]],order=1) for j,f in enumerate('AB')]
    corr.append(np.corrcoef(*patches)[0,1])
corr = np.array(corr)
with open(ROOT/'work/manual.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['pick_id','x_A_px','y_A_px_down','x_B_px','y_B_px_down','u_px_per_pair','v_px_per_pair_down','depth_A_px','depth_B_px','patch15_raw_NCC'])
    for i in range(200): w.writerow([i+1,p[0,0,i],p[1,0,i],p[0,1,i],p[1,1,i],uv[i,0],uv[i,1],da[i],db[i],corr[i]])

fig,axs=plt.subplots(2,1,figsize=(13,9),constrained_layout=True)
for ax,f,s,j in zip(axs,'AB',[sa,sb],[0,1]):
    ax.imshow(imgs['img'+f],cmap='gray',vmin=30,vmax=230,origin='upper')
    ax.plot(x,s,color='#00e6ff',lw=1,label='Saved free surface')
    ax.plot(x,np.rint(s)+10,color='#f3aa51',lw=.8,ls='--',label='First retained altmask row')
    if j==0:
        q=ax.quiver(xy[:,0],xy[:,1],uv[:,0],uv[:,1],color='#f7cf55',angles='xy',scale_units='xy',scale=1,width=.0022)
        ax.quiverkey(q,.79,.95,10,'10 px / image pair',coordinates='axes',labelcolor='white')
    ax.scatter(p[0,j],p[1,j],s=6,color='#e554bd',alpha=.65,label='Manual endpoints')
    for i in np.argsort(da)[:7]: ax.text(p[0,j,i]+2,p[1,j,i]-2,str(i+1),color='white',fontsize=8)
    ax.set(xlim=(60,390),ylim=(180,0),xlabel='Local x (pixels)',ylabel='Local y, down (pixels)',title=f'Image {f}: raw particle image and manual '+('displacements' if j==0 else 'endpoints'))
    leg=ax.legend(loc='lower right',fontsize=8,facecolor='#444444')
    for lab in leg.get_texts(): lab.set_color('white')
fig.savefig(ROOT/'work/audit_manual_surface.png',dpi=180)
plt.close(fig)

near=np.argsort(da)[:8]
fig,axs=plt.subplots(4,4,figsize=(12,12),constrained_layout=True)
for k,i in enumerate(near):
    for j,f in enumerate('AB'):
        ax=axs[k//2,(k%2)*2+j]
        ax.imshow(imgs['img'+f],cmap='gray',vmin=30,vmax=255,origin='upper',interpolation='nearest')
        ax.plot(x,[sa,sb][j],color='cyan',lw=1)
        xx,yy=p[:,j,i]
        ax.plot(xx,yy,'+',color='#ff5353',ms=10,mew=1)
        ax.set(xlim=(xx-15,xx+15),ylim=(yy+15,yy-15),title=f'Pick {i+1}, {f}; depth {da[i] if j==0 else db[i]:.1f}px')
        ax.tick_params(labelsize=7)
fig.suptitle('Closest manual picks: paired 30 × 30 pixel crops; crosses show saved endpoints',fontsize=13)
fig.savefig(ROOT/'work/audit_nearest_patches.png',dpi=160)
plt.close(fig)

Y,X=np.mgrid[-4:4.01:.25,-4:4.01:.25]
fig,axs=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
peak_offsets=[]
for j,f in enumerate('AB'):
    im=imgs['img'+f].astype(float)
    patches=np.array([map_coordinates(im,[Y+p_saved[1,j,i],X+p_saved[0,j,i]],order=1) for i in range(200)])
    stack=patches.mean(axis=0)
    ij=np.unravel_index(np.argmax(stack),stack.shape)
    peak_offsets.append((float(X[ij]),float(Y[ij])))
    ax=axs[j]
    ax.imshow(stack,cmap='magma',extent=(-4.125,4.125,4.125,-4.125),interpolation='nearest')
    ax.plot(0,0,'+',color='cyan',ms=15,mew=2,label='Saved p')
    ax.plot(-1,-1,'x',color='white',ms=10,mew=2,label='p − 1')
    ax.set(xlabel='Image x offset from saved p (px)',ylabel='Image y offset from saved p (px)',title=f'Frame {f}: mean of 200 particle patches')
    ax.legend(loc='lower right',fontsize=8)
fig.suptitle('Direct particle centering favors MATLAB p − 1 for zero-based TIFF coordinates')
fig.savefig(ROOT/'work/audit_particle_origin.png',dpi=170)
plt.close(fig)

yy,xx=np.indices((501,501))
stats=[]
for f,s in zip('AB',[sa,sb]):
    im=imgs['img'+f]; depth=yy-s[None,:]
    hf=im.astype(float)-gaussian_filter(im.astype(float),2)
    for lo,hi in [(-10,0),(0,3),(3,5),(5,10),(10,15),(15,20),(20,30),(30,50),(50,100),(100,150)]:
        m=(xx>=60)&(xx<=390)&(depth>=lo)&(depth<hi)
        stats.append((f,lo,hi,int(m.sum()),float((im[m]==255).mean()),float(np.median(im[m])),float(np.std(hf[m]))))

t=cKDTree(xy); ds,ix=t.query(xy,k=2)
lines = [
'# Independent data audit',
'',
'## Coordinates and file content',
'',
'All six TIFFs are single-channel 8-bit 501 × 501 crops. The “mask” TIFFs are intensity images with excluded rows replaced by zero; they are not binary masks. Original particle intensities are unchanged at retained pixels.',
'',
'The MATLAB v7.3 file contains 200 manually selected point pairs. HDF5 array p has axes [coordinate (x,y), frame (A,B), pick]. Displacements are p[:,1,:] − p[:,0,:]. Positive v points downward in image coordinates. For zero-based TIFF coordinates use p − 1. This follows the conventional MATLAB-to-Python coordinate conversion and, crucially, direct particle-centering evidence below. For geometric depth use saved surface − 1 in the same convention; for pixel validity use the supplied mask images directly.',
'',
'There is a bookkeeping ambiguity: p_orig − (300,340) = saved p, and saved surface arrays equal the original arrays at Python slice [299:800], minus 340. Those arithmetic offsets alone could suggest saved p is already zero-based. However, averaging all 200 raw image patches centered at saved p places the particle-intensity peak at offset (−1,−1)px in A and (−1,−1.25)px in B. Bright-particle median centroids are likewise about (−0.9,−1.0)px. Visually, p lies systematically below/right of particle centers; p − 1 corrects that. The cropped TIFF alignment is stronger evidence than inferred author crop arithmetic. Preserve this potential one-pixel bookkeeping uncertainty in any absolute location or depth claim. Displacements themselves are identical under either convention.',
'',
'The mask’s first nonzero row is round(saved surface)+1 at every column (zero-based row). The alternative mask first nonzero row is round(saved surface)+9, exactly 8 rows deeper than the first mask. Relative to the converted geometric surface s=saved surface−1, these first retained rows are round(s)+2 and round(s)+10 respectively. Thus the alternative mask agrees with stored altmask_offset=10 under the chosen convention. Hard masked borders can create an artificial advecting edge; treat masks as validity regions and fit retained raw image intensities.',
'',
'No image time separation or physical pixel calibration exists in this MAT file. Thus displacement in px/pair is directly supported; m/s and s⁻¹ require calibration/time metadata. For isotropic image calibration, ∂u_physical/∂x_physical = (∂d_x/∂x_pixel)/Δt.',
'',
'## Manual sample coverage and limits',
'',
f'All 200 pairs are finite. No exactly duplicated A locations or point pairs occur. A locations span x={xy[:,0].min():.2f}–{xy[:,0].max():.2f}, y={xy[:,1].min():.2f}–{xy[:,1].max():.2f} px. Vertical depth below A surface ranges {da.min():.2f}–{da.max():.2f} px. The median nearest-neighbor A-point spacing is {np.median(ds[:,1]):.2f} px (minimum {ds[:,1].min():.2f} px), so errors are not independent at very close point clusters.',
'',
f'Depth counts: '+ '; '.join(f'{lo}–{hi}px: {np.sum((da>=lo)&(da<hi))}' for lo,hi in [(0,10),(10,20),(20,30),(30,50),(50,100),(100,150)]) + f'. Claims within {da.min():.2f}px of the surface cannot be validated by these manual picks. A-point membership in a depth band should be used for validation, and transformed B pixels must also remain in water.',
'',
f'Displacement component ranges are u={uv[:,0].min():.3f} to {uv[:,0].max():.3f} px/pair and v={uv[:,1].min():.3f} to {uv[:,1].max():.3f} px/pair. No picks are discarded as outliers from these values: real gradients are large, and neighbor disagreement alone is not evidence of a bad pick.',
'',
f'Closest picks are #65 (depth {da[64]:.2f}px; u={uv[64,0]:.2f},v={uv[64,1]:.2f}) and #64 (depth {da[63]:.2f}px; u={uv[63,0]:.2f},v={uv[63,1]:.2f}). Their 15×15 raw patch normalized correlations are {corr[64]:.3f} and {corr[63]:.3f}; correlation does not measure picking uncertainty and depends on neighborhood deformation.',
'',
f'The maximum-depth point of the converted surface shifts from x=200 in A to x=258 in B (58px) while its y increases from {sa.max():.2f} to {sb.max():.2f}px. This is motion of the free-surface shape; it is not a fluid-particle correspondence and should not be imposed as a 58px fluid velocity.',
'',
'A provisional pixel ROI x=60…390, y=0…180 encloses the depression in both frames, nearly all near-lump flow structure, and all 200 manual A endpoints. A surface-relative computation can retain depths 0…150px. Conversion of that ROI to the user’s 1–2cm region is pending calibration. Computation may require padded support outside the final plot bounds.',
'',
'## Near-surface texture audit (x=60…390)',
'',
'Saturation, reflections, nonuniform illumination, and a moving boundary make the first few pixels especially difficult. A is generally brighter than B. The following values refer to the raw images, not the zero-filled masks. High-frequency standard deviation is after subtracting a Gaussian blur of σ=2px; it describes texture, not independent particle density.',
'',
'| Frame | Vertical depth (px) | Pixels | Saturated (255) | Median intensity | High-frequency SD |',
'|---|---:|---:|---:|---:|---:|',
]
for f,lo,hi,n,sat,med,hf in stats: lines.append(f'| {f} | {lo}…{hi} | {n} | {100*sat:.2f}% | {med:.1f} | {hf:.2f} |')
lines += ['', '## Suggested interpretation', '', 'Use raw images and moving-surface validity masks, a local illumination-normalized residual, and image-only deformation fits. Keep all manual pairs independent of prediction fitting; report comparison errors by depth as well as globally. A quantitative reliability boundary should combine image agreement, forward/backward consistency, and neighborhood support, rather than simply displaying vectors to the masked edge. Higher-order deformation increases fit flexibility and may overfit sparse/changed texture. Spatial differentiation magnifies subpixel noise; present horizontal gradients only with a stated spatial support and uncertainty or sensitivity assessment.', '', 'Files: manual.csv contains all 200 pairs converted to zero-based TIFF coordinates and image-coordinate displacement/depth columns; audit_manual_surface.png shows source/target coverage; audit_nearest_patches.png exposes raw texture at the nearest manual pairs; audit_particle_origin.png shows average particle-origin evidence. No manual vectors were used to train or initialize an image prediction.']
(ROOT/'work/data_audit.md').write_text('\n'.join(lines)+'\n')
print('Saved audit, plots, and manual CSV.')
print('Raw 15px patch NCC quantiles:',np.quantile(corr,[0,.05,.25,.5,.75,.95,1]))
print('Retained mask differences:',{k:int(np.max(abs(imgs[k].astype(int)[imgs[k]>0]-imgs[k.split('_')[0]].astype(int)[imgs[k]>0]))) for k in imgs if '_' in k})
