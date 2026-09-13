from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('historical-user-files/Documents/Codex/2026-09-11/i-a')
z=np.load(ROOT/'work/metadata.npz'); p=z['p']-1
A=np.array(Image.open('data/ExpLCL_1_03-123_imgA.tif')).astype(float)
B=np.array(Image.open('data/ExpLCL_1_03-123_imgB.tif')).astype(float)
pred=np.load(ROOT/'work/baseline_reference_matlab.npz')['pred']
ids=[84,85,94,95,96,142,143]
def patch(im,center,r):
    dy,dx=np.mgrid[-r:r+1,-r:r+1]
    return map_coordinates(im,[dy+center[1],dx+center[0]],order=1)
def ncc(a,b):
    return float(np.corrcoef(a.ravel(),b.ravel())[0,1])

fig,axs=plt.subplots(7,4,figsize=(13,19),constrained_layout=True)
lines=['# Upstream manual-match diagnostic','','Coordinates below are zero-based (saved p−1); v is positive down. Manual vectors are only being diagnosed, not fit to any image-prediction model. For the alternative comparison this audit uses the existing baseline_reference_matlab flow, which has 0–13px horizontal displacements in this region. These alternatives are not claimed as ground truth.','','A single small bright blob has limited identifying structure: a high correlation over 5–9px is evidence of appearance consistency but can be reproduced by other similar particles. Larger support adds neighbor particles but also includes surface glare and strongly deformed/changed background.','', '| Pick | Manual (u,v) | Low-flow alternative (u,v) | NCC manual 7/15/25px | NCC alternative 7/15/25px |','|---|---|---|---|---|']
stats=[]
for row,k in enumerate(ids):
    i=k-1; a=p[:,0,i]; b=p[:,1,i]; uv=b-a; alt=a+pred[i]
    cors=[]
    for c in [b,alt]: cors.append([ncc(patch(A,a,r),patch(B,c,r)) for r in [3,7,12]])
    lines.append(f'| {k} | ({uv[0]:.2f},{uv[1]:.2f}) | ({pred[i,0]:.2f},{pred[i,1]:.2f}) | '+ '/'.join(f'{q:.2f}' for q in cors[0]) + ' | '+ '/'.join(f'{q:.2f}' for q in cors[1])+' |')
    stats.append((k,cors))
    for col,(im,c,txt) in enumerate([(A,a,'A source'),(B,b,'B manual match'),(B,alt,'B low-flow alternative')]):
        ax=axs[row,col]
        ax.imshow(patch(im,c,15),cmap='gray',vmin=30,vmax=255,extent=(-15.5,15.5,15.5,-15.5),interpolation='nearest')
        ax.add_patch(plt.Circle((0,0),3.5,fill=False,color='#22dfff',lw=.8))
        ax.set(xticks=[-10,0,10],yticks=[-10,0,10],title=f'#{k}: {txt}'+ (f'\nNCC₇ {cors[col-1][0]:.2f}, NCC₂₅ {cors[col-1][2]:.2f}' if col>0 else f'\nu={uv[0]:.1f}, v={uv[1]:.1f} px'))
        ax.tick_params(labelsize=7)
    ax=axs[row,3]
    ushifts=np.linspace(-5,30,141)
    for r,color in [(3,'#0078c0'),(7,'#bd6700'),(12,'#bc4caf')]:
        # A one-dimensional diagnostic follows the manual displacement direction.
        cc=[ncc(patch(A,a,r),patch(B,a+[u,u*uv[1]/uv[0]],r)) for u in ushifts]
        ax.plot(ushifts,cc,lw=1.2,color=color,label=f'{2*r+1}px window')
    ax.axvline(uv[0],color='black',ls='--',lw=1,label='Manual u')
    ax.set(xlim=(-5,30),ylim=(-.6,1),title='Correlation along manual vector',xlabel='Horizontal displacement (px)',ylabel='Raw-patch NCC')
    ax.grid(alpha=.15)
    if row==0:ax.legend(fontsize=7,loc='lower right')
fig.suptitle('Upstream glare region: individual particles versus neighborhood matching\n31 × 31px crops; cyan circle marks 7px central support. Same grayscale for all images.',fontsize=14)
fig.savefig(ROOT/'work/audit_upstream.png',dpi=150)
plt.close(fig)
lines += ['', '## Interpretation of the raw crops', '',
'Picks #84, #85, #95, #96 and #142 have identifiable compact central spots at the manually matched B locations; their 7px-window correlations are 0.76, 0.89, 0.96, 0.85 and 0.81. At the existing low-flow alternative those correlations are −0.06, 0.25, 0.77, 0.21 and −0.17. The manual matches therefore have direct local image support, especially #85, #96 and #142. These metrics do not by themselves uniquely identify a particle, because another round spot can also correlate strongly. #95 demonstrates this ambiguity: its alternative still gives 0.77, even though the manual match gives 0.96.', '',
'Picks #94 and #143 are substantially weaker. #94 is a faint source particle whose brightness changes; its 7px manual correlation (0.52) is essentially tied with the low-flow alternative (0.51). #143 is close to changed glare/background and gives a 7px score 0.63 at the manual location versus 0.72 at the low-flow alternative. Their manual correspondences may be supported by the larger particle constellation, but cannot be certified from the individual spots. No picks are discarded.', '',
'A 15–25px window includes strongly changing surface illumination and neighboring particles. It often reduces the distinction of the central particle correspondence; #84 even gives larger 15px correlation for the incorrect-looking low-flow alternative than at the manual position. The nearby raw surface/glare texture should not be assumed to move with the particle layer. Most spots are only about 2–4px across, with limited orientational structure; allowing rotation cannot recover information that the individual circular blobs do not encode.', '',
'The 19–24px manual translations exceed several particle diameters and skip past other visually similar spots. Several upstream source spots are separated by roughly 8–11px horizontally, so an optical-flow optimizer starting near zero can converge to a repeated-particle or background match. A broad search for multiple candidate particle correspondences, followed by 7–9px local photometric fits and image-only constellation/spatial consistency, is better justified here than simply increasing deformation order over a large window. Small windows alone are also ambiguous; candidate disambiguation matters.', '',
'I see no positive evidence in these crops for two physical velocity layers. The observed alternatives can be explained by changing glare and multiple similar particle blobs. Surface reflections, out-of-plane imaging or other optical multiplicity remain possible but cannot be established from these two frames alone. Do not describe the low-flow alternative as a second measured flow population.', '',
'The correlation curves in audit_upstream.png are one-dimensional diagnostics: horizontal shifts from −5 to30px, with vertical shift scaled along the manually picked vector direction. They reveal local appearance support and competing peaks along that direction; they are not an exhaustive 2D search or an independent accuracy test.']
(ROOT/'work/audit_upstream.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
