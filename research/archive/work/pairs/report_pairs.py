"""Write pair-specific scientific interpretation using saved results only."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def metrics(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
                rmse=float(np.sqrt(np.mean(a*a))) if len(a) else None,p95=float(np.percentile(a,95)) if len(a) else None)

def run(pair):
    w=ROOT/'work/pairs'/str(pair);o=ROOT/'outputs'/('pair_'+str(pair))
    inp=load(w/'inputs.npz');z=load(w/'results.npz');s=json.loads((o/'plot_and_export_summary.json').read_text());manifest=json.loads((w/'input_manifest.json').read_text())
    ok=z['accepted'].astype(bool);gx=z['gradient_accepted_xx'].astype(bool);gz=z['gradient_accepted_yy'].astype(bool)
    cv=z['classical_available_at_grid'].astype(bool);c=z['classical_displacement_native_at_grid'];d=z['disp'];g=z['gradient'];q=z['query_full'];dep=z['depth'];err=np.linalg.norm(d-c,axis=1)
    cor=z['classical_ncc_at_grid'];joint=ok&cv
    strong_cor=np.zeros(len(cor),bool);finite_cor=np.isfinite(cor)
    strong_cor[finite_cor]=cor[finite_cor]>=.8
    retained_boundary=inp['first_retained_row_a']
    depth_from_retained=q[:,1]-np.interp(q[:,0],np.arange(len(retained_boundary)),retained_boundary)
    comparisons={'all_available_classical':metrics(err[joint]),'finite_classical_correlation':metrics(err[joint&np.isfinite(cor)]),
                 'classical_correlation_ge_0_8':metrics(err[joint&strong_cor])}
    depthbins=[]
    for lo,hi in [(12,20),(20,40),(40,80),(80,150),(150,354),(354,800),(800,1200),(1200,1800)]:
        m=(dep>=lo)&(dep<hi)
        depthbins.append(dict(lo=lo,hi=hi,n=int(m.sum()),vector=int((m&ok).sum()),du_dx=int((m&gx).sum()),dw_dz=int((m&gz).sum()),comparison=metrics(err[m&joint])))
    xy_ranges={k:([np.min(q[v],axis=0).tolist(),np.max(q[v],axis=0).tolist()] if np.any(v) else None) for k,v in [('vectors',ok),('du_dx',gx),('dw_dz',gz)]}
    diag=dict(pair=pair,comparison_subsets=comparisons,depth_bins=depthbins,accepted_coordinate_ranges=xy_ranges,
              conservative_vector_count=int(ok.sum()),conservative_du_dx_count=int(gx.sum()),conservative_dw_dz_count=int(gz.sum()),
              minimum_accepted_vector_depth=float(dep[ok].min()) if ok.any() else None,maximum_accepted_vector_depth=float(dep[ok].max()) if ok.any() else None,
              minimum_du_dx_depth=float(dep[gx].min()) if gx.any() else None,minimum_dw_dz_depth=float(dep[gz].min()) if gz.any() else None,
              minimum_vector_distance_below_retained_boundary_px=float(depth_from_retained[ok].min()) if ok.any() else None,
              minimum_du_dx_distance_below_retained_boundary_px=float(depth_from_retained[gx].min()) if gx.any() else None,
              minimum_dw_dz_distance_below_retained_boundary_px=float(depth_from_retained[gz].min()) if gz.any() else None,
              classical_available_without_finite_dcor=int((cv&~np.isfinite(cor)).sum()),both_derivatives=int((gx&gz).sum()),
              planar_divergence_proxy_summary_assumed_per_s=metrics((g[:,0,0]+g[:,1,1])[gx&gz]/float(inp['DT'])),
              planar_divergence_note='Signed summary of du/dx+dw/dz, assuming DT is seconds, where both screens pass; no zero-divergence constraint or physical 2D assumption imposed.')
    (o/'coverage_and_comparison.json').write_text(json.dumps(diag,indent=2))
    fmt=lambda v:'unavailable' if v is None else '{:.4f}'.format(v)
    lines=['# Pair {}: full-image conservative hybrid velocity'.format(pair),'',
        '**{:,} of {:,} evaluated locations pass the velocity screen.** The separate component screens retain **{:,} estimates of du/dx** and **{:,} estimates of dw/dz**.'.format(int(ok.sum()),len(ok),int(gx.sum()),int(gz.sum())),'',
        'The calculation spans every image column and the entire observed water depth to the bottom of the 2048 × 2048 frame. The previous 354-pixel depth limit is removed. The supplied TIFFs already have a zeroed upper region; no values are inferred there. The grid spacing is 8 pixels, which is a sampling interval rather than a claim of independent 8-pixel spatial resolution.','',
        '**Near-surface agreement is weaker than the full-frame average.** At the {:,} accepted comparison locations 12–20 pixels below the inferred surface, the mean hybrid–PIV difference is {} pixels; at 20–40 pixels depth it is {} pixels ({:,} comparisons). The large deeper region dominates the overall mean. These differences do not identify which method is correct, and the supplied PIV is not independent ground truth.'.format(depthbins[0]['comparison']['n'],fmt(depthbins[0]['comparison']['mean']),fmt(depthbins[1]['comparison']['mean']),depthbins[1]['comparison']['n']),'',
        '## Figures and data','',
        '- [Full-image hybrid/classical quiver overlay](hybrid_piv_overlay.png): cyan hybrid and orange supplied PIV, with both arrow lengths magnified by 4 and sampled every 32 pixels for readability.',
        '- [Near-surface detailed overlays](hybrid_piv_near_surface.png): the same fields shown at actual displacement scale, every 16 pixels.',
        '- [Magnified deeper-water overlay](hybrid_piv_deep_water.png): the same fields below image y = 500, with a uniform 40-fold arrow magnification to reveal small displacements.',
        '- [Both velocity derivatives](du_dx_dw_dz.png), [near-surface derivative detail](du_dx_dw_dz_near_surface.png), plus individual [du/dx](du_dx.png) and [dw/dz](dw_dz.png) plots.',
        '- [Classical-PIV difference map and histogram](piv_comparison.png).',
        '- [MATLAB numerical data](conservative_field.mat) and [NumPy numerical data](conservative_field.npz). Rejected conservative estimates are NaN. Raw estimates, component masks, native PIV arrays, and diagnostic values are retained separately.','',
        'Each figure also has an SVG version. The purple line marks the boundary of retained image data, not an independently measured physical interface. Gray regions in derivative maps are withheld; the light region above the verified image mask has no usable data. The gradient color scale is shared between du/dx and dw/dz and between the two new pairs; its limits are stated on each plot. Numerical values are never clipped in the data files.','',
        '## The method used','',
        'These are fresh fits for this image pair; no image texture, particle tracks, or polynomial coefficients from observation 123 are inserted. The original detector and tracker are retained: difference-of-Gaussians peaks at scales 0.6 and 2 pixels, threshold 8 intensity units, local maxima in 5 × 5 neighborhoods, background at scale 5 below 180, and source depth at least 14 pixels. Reciprocal translations of 9 × 9 particle patches use the same correlation, ambiguity, and one-pixel return-error criteria. Accepted automatic tracks initialize the final image fit.','',
        'The main fit is masked, robust local quadratic image registration with radius 13 pixels, bilinear image sampling, regularization 0.00005, and up to 35 iterations. Affine radius-13, quadratic radius-19, and quadratic radius-13 with a 14-pixel surface margin provide sensitivity comparisons. The independently fitted reverse field is composed at the predicted B endpoint. The local polynomials use the original compact C2 blend with radius 16 pixels, including derivatives of its weights.','',
        'A vector passes only with actual source/target visibility; source depth at least 12 pixels and target depth at least 10; accepted-particle convex-hull support, nearest particle within 12 pixels and at least 6 within 25; blended patch NCC at least 0.60; composed return error at most 1 pixel; all three alternatives available with maximum vector disagreement at most 1.5 pixels; determinant of the blended deformation above 0.20; and at least 95% good-patch weight in each direction. A good patch has Gaussian-weighted target-valid fraction above 0.85 and minimum map determinant above 0.05. This fraction is normalized over already source-valid pixels, not the entire nominal square.','',
        'For each requested diagonal derivative separately, source depth must be at least 20 pixels, the nearest accepted particle must be within 10 pixels, and disagreement of that component across the three alternatives must be at most 0.08 pixels/pixel. Extending this criterion to G11 supplies a conservative dw/dz mask; passing the du/dx screen alone does not validate dw/dz. There is no incompressibility constraint, learned model, or individual reflection classifier.\n\nThe 8-pixel evaluation spacing is finer than the independent spatial resolution: the primary fit uses a 27 × 27 window and combines local fits within 16 pixels, so source-image support can extend about 29 pixels from an interior query. The radius-19 comparison extends farther. Windows are truncated at masks and image edges, where passing estimates can have one-sided support. Detected features require their 9 × 9 patches to lie within the frame; the final x = 2047 and y = 2047 grid edges consequently lie outside the feature hull and are withheld. Full-image evaluation therefore does not imply complete or uniform accepted coverage.','',
        '## Classical PIV comparison','',
        'Comparison vectors come from the supplied 511 × 511 `delta_x` and `delta_z` arrays on `xPIV`/`zPIV`, transposed from HDF5 order and mapped from one-based coordinates to zero-based image positions. The prediction grid coincides with every other native PIV node. Native values are copied exactly at those locations; absent values and positions outside the native axes stay absent. The dense `delta_x1`/`delta_z1` fields are used only to initialize coarse image fits and are not treated as extra PIV observations.','',
        'The supplied classical vectors are screened for their own finite components and source/target image availability. They are not filtered by agreement with the hybrid or by hybrid acceptance. `dcor` is exported separately. Some supplied native-grid vectors have no finite correlation score; the table below distinguishes them. The supplied files may contain PIV postprocessing, so “native” identifies the saved grid, not a guarantee that every entry is an unaltered correlation measurement.','',
        '**This is a comparison of related methods, not independent ground-truth validation.** Both methods use the same images, and the supplied PIV initializes the coarse stage. No manual reference vectors were provided for these pairs. The hybrid acceptance screen never uses hybrid–classical disagreement.','',
        '| Comparison subset, where hybrid passes | Locations | Mean difference (px) | Median (px) | RMS (px) |','|---|---:|---:|---:|---:|']
    for name,key in [('All available supplied PIV','all_available_classical'),('Finite supplied correlation score','finite_classical_correlation'),('Supplied correlation ≥ 0.8','classical_correlation_ge_0_8')]:
        m=comparisons[key];lines.append('| {} | {:,} | {} | {} | {} |'.format(name,m['n'],fmt(m['mean']),fmt(m['median']),fmt(m['rmse'])))
    lines += ['', 'The difference map saturates at its stated color scale; the histogram explicitly counts values beyond its plotted range. All comparison values are retained numerically.','',
        '## Coordinates, surface convention, and units','',
        'Primary image coordinates are x right and y downward, and raw displacement is B minus A. In both TIFFs, every first retained row agrees exactly with the rounded `surfacePIVImg`. These new MAT files do not contain an explicit unshifted physical surface trace. The earlier verified export convention `surface = surfacePIVImg - 12 pixels` is retained as an **inferred geometry convention**, and this inference is recorded in the input manifest. The actual image availability is verified independently. No masked-out pixels are restored for these pairs.','',
        'The physical-unit convention is u and x rightward, w and z upward. If DX is metres/pixel and DT seconds:', '',
        '`u = dx * DX / DT`', '', '`w = -dy * DX / DT`', '', '`du/dx = G00 / DT`', '', '`dw/dz = G11 / DT`','',
        'The two sign changes in the vertical component and vertical coordinate cancel, so dw/dz is positive G11/DT. Both derivatives are at fixed orthogonal image coordinates, not along the sloping surface.','',
        'The MAT provides DX = {:.15g} and DT = {:.15g}, without explicit unit labels. Conditional SI outputs are named `assumed`; `physical_units_confirmed = 0`. The derivative plots use DT = {} assumed seconds, and pixel-gradient versions are supplied numerically. Under SI calibration the frame is approximately {:.3f} cm wide. Displacement/DT describes interval-averaged tracer motion attached to the A position; its derivative is a finite-time gradient proxy rather than an independently resolved instantaneous Eulerian gradient.'.format(float(inp['DX']),float(inp['DT']),float(inp['DT']),2048*float(inp['DX'])*100),'',
        '## Coverage by vertical depth below the inferred trace','',
        '| Depth band (pixels) | Evaluated | Vectors | du/dx | dw/dz | PIV comparisons | Mean difference (px) |','|---|---:|---:|---:|---:|---:|---:|']
    for b in depthbins:lines.append('| {}–{} | {:,} | {:,} | {:,} | {:,} | {:,} | {} |'.format(b['lo'],b['hi'],b['n'],b['vector'],b['du_dx'],b['dw_dz'],b['comparison']['n'],fmt(b['comparison']['mean'])))
    lines += ['', 'The derivative maps retain visible fine-scale graininess, especially near the surface. Passing the image-consistency and sensitivity screens does not establish that each small-scale strain fluctuation is physical. These maps have not been further smoothed or constrained to cancel du/dx with dw/dz.','', 'The nearest accepted vector depth is {} pixels; the maximum accepted depth is {} pixels. These are local extrema and do not imply equally close recovery along the entire surface. The tests measure image consistency and limited model sensitivity, not a calibrated probability of correctness or a confidence interval.'.format(fmt(diag['minimum_accepted_vector_depth']),fmt(diag['maximum_accepted_vector_depth']))]
    lines += ['', 'Measured from the actual retained-image boundary, the closest accepted velocity is {} pixels below that boundary; the closest accepted du/dx and dw/dz are {} and {} pixels below it. These distances use the verified first retained pixel row and do not require the inferred physical-interface offset.'.format(fmt(diag['minimum_vector_distance_below_retained_boundary_px']),fmt(diag['minimum_du_dx_distance_below_retained_boundary_px']),fmt(diag['minimum_dw_dz_distance_below_retained_boundary_px']))]
    (o/'analysis_report.md').write_text('\n'.join(lines)+'\n')
    print('Wrote report for pair',pair)
    return diag

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--pair',type=int,required=True);a=p.parse_args();run(a.pair)
