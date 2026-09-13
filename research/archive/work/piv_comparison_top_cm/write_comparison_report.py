"""Write a self-contained explanation and numeric comparison summary."""
from quiver_compare import *

def main():
    rows=[];summary={};gradrows=[]
    for pair in [80,100]:
        p=read(WORK/('pair_%s_integral_comparison.npz'%pair))
        m=json.loads((WORK/('pair_%s_gradient_comparison.json'%pair)).read_text())
        q=json.loads((OUT/('pair_%s'%pair)/'velocity_comparison_metrics.json').read_text())
        summary[str(pair)]={'quiver_comparison':q['overall'],'selected_depths':[],
            'fixed_joint_width_cm':float(p['common_width_m'])*100,
            'original_of_fixed_width_cm':float(p['of_original_common_width_m'])*100}
        for h in [1.,2.,5.,10.]:
            i=int(np.argmin(np.abs(p['depth_m']*1000-h)))
            assert abs(p['depth_m'][i]*1000-h)<1e-9
            a=float(p['of_matched_integral_m2_per_s'][i])*1e4
            b=float(p['piv_matched_integral_m2_per_s'][i])*1e4
            coverage=float(p['matched_coverage_fraction'][i])*100
            rows.append('| %d | %.1f | %.2f | %.3f | %.3f | %+.3f |'%(pair,h,coverage,a,b,a-b))
            summary[str(pair)]['selected_depths'].append(dict(depth_mm=h,matched_width_percent=coverage,
                of_Q_cm2_per_s=a,piv_Q_cm2_per_s=b,difference_Q_cm2_per_s=a-b,
                of_mean_u_cm_per_s=float(p['of_matched_mean_u_m_per_s'][i])*100,
                piv_mean_u_cm_per_s=float(p['piv_matched_mean_u_m_per_s'][i])*100))
        for comp in ['du_dx','dw_dz']:
            a=m['metrics'][comp]['analytic_of_vs_piv']
            gradrows.append('| %d | %s | %d | %.3f | %.3f |'%(pair,comp,a['count'],a['pearson_correlation'],a['assumed_per_s']['root_mean_square_difference']))
    report=r'''# Image-only optical flow compared with supplied PIV

The completed optical-flow fields for pairs **80 and 100** are unchanged. Supplied PIV velocities are read only after prediction, for the comparisons here. All figures cover the whole horizontal image extent and the top **1 cm vertically below the local inferred free surface**.

## What “fixed common domain” meant

It meant choosing one set of horizontal **x-segments** and using exactly those same segments at every sampled depth. The segments can be disconnected. This was intended to keep changing coverage from changing the integral as depth changes; it did not mean the full image width.

In the previous optical-flow-only result, these fixed segments were the intersection of accepted segments over the sampled depths from **1.13009 to 10 mm**. They totaled **6.36 cm for pair 80** and **4.42 cm for pair 100**, within the 11.57 cm image width. Including all depths from 0 to 10 mm would leave no such segments because the shallowest image strip is masked.

Once native supplied PIV availability is also required, **neither pair has any nonempty horizontal segment supported at every sampled depth from 1.13 to 10 mm** under the stated no-gap sampling rule. PIV also does not cover the entire previous optical-flow-only fixed domain at any sampled depth. A numerical PIV curve on that old fixed domain would therefore require gap filling. The corresponding exported values are NaN, not zero.

## The direct comparison now plotted

At each depth h, use the horizontal segments S(h) where **both** methods have available estimates. The two curves are

\[
Q_{\rm OF}(h)=\int_{S(h)}u_{\rm OF}(x,z_s(x)-h)\,dx,
\qquad
Q_{\rm PIV}(h)=\int_{S(h)}u_{\rm PIV}(x,z_s(x)-h)\,dx.
\]

Here z is positive upward, z_s is the local surface height, and h is a vertical distance below that surface. The integral uses horizontal dx, with no surface arc-length factor. The red and blue curves use **identical intervals, quadrature weights and widths at each depth**. S(h) can change with depth. The mean horizontal velocity is Q(h)/L(h), where L(h) is the exact summed width of S(h).

This leaves the predicted velocities unchanged but recomputes the optical-flow integral on the smaller comparison support. Neither curve is presented as a full-width integral. Coverage is shown alongside it. Changes in coverage still affect either integral as a function of depth, which is why the mean and the support map are included.

- **horizontal_integral_vs_piv.png**: primary comparison, with integral, mean velocity and supported width.
- **horizontal_segments_used.png**: the actual intervals used. Dark green marks support shared by both methods at that depth.
- **original_profiles_and_piv_own_coverage.png**: preserves the previous optical-flow integral and compares it with PIV integrated over its own available intervals. These curves have different coverage; their integral difference is not solely a velocity difference.
- **quiver_overlay.png**: red optical flow and blue supplied PIV at common display-grid locations, including locations where only one estimate exists. Both have the same arrow scale. The displayed grid is thinned equally for clarity.
- **quiver_overlay_deeper_half.png**: the same vectors from 0.5 to 1 cm depth, with larger arrows to make the slower motion visible.
- **du_dx_vs_piv.png**, **dw_dz_vs_piv.png**: optical-flow and PIV Cartesian gradient maps plus their difference at shared available points.

## Selected integral values

Q is in cm²/s under the calibration assumption below. “Width” is the fraction of the full 11.57 cm image width included in **both** integrals. The difference is optical flow minus PIV.

| Pair | Local depth (mm) | Shared width (%) | Optical-flow Q | PIV Q | Difference |
|---|---:|---:|---:|---:|---:|
'''+'\n'.join(rows)+r'''

The main near-surface disagreement is larger for pair 100. At greater depth the integrated and mean horizontal velocities agree more closely. This comparison tests agreement with the supplied PIV; it does not identify either method as ground truth.

## Native PIV sampling and missing values

Only the native saved `compVel/delta_x` and `compVel/delta_z` arrays are used. The dense `delta_x1` and `delta_z1` fields are not read. “Native” describes the saved grid; the files do not establish that these values are untouched raw correlation estimates.

The native grid is spaced by 4 pixels, with zero-based coordinates 3, 7, …, 2043. MATLAB axis coordinates are reduced by one and arrays transposed to [image row, image column]. In image coordinates, the PIV displacement is `(delta_x, -delta_z)`. Quiver comparison uses exact native-node lookup wherever the existing reporting grid coincides; no interpolation is used for those PIV arrows. The optical-flow acceptance masks remain unchanged.

For surface-following integral samples, PIV uses bilinear interpolation on the native grid. Every corner with a positive interpolation weight must have finite supplied components and lie in the retained source image below the inferred surface. Zero-weight missing neighbors do not invalidate exact-node queries. No extrapolation or interpolation across missing contributing nodes is allowed. Each 2-pixel Simpson interval requires an available endpoint, midpoint and endpoint from both methods. A missing interval contributes no measured width; it is never treated as a measured zero velocity.

No arbitrary correlation-score threshold is applied. About 10.4% (pair 80) and 14.9% (pair 100) of source-supported finite native vectors within the top centimetre lack a finite `dcor` value. Their provenance is not inferred from that absence. The data include separate finite-correlation and source-plus-target-visibility sensitivity comparisons, each with its own identical support for the two methods. These are not confidence intervals. Requiring finite correlation substantially reduces the already limited shallow comparison width.

## Gradient comparison

The main optical-flow panels reuse the saved analytic derivatives and their existing conservative masks. PIV gradients use the Cartesian centered difference across ±8 pixels (16-pixel endpoint separation, approximately 0.904 mm):

\[
\partial_x u\simeq\frac{d_x(x+8,y)-d_x(x-8,y)}{16\,\Delta t},
\qquad
\partial_z w\simeq\frac{d_y(x,y+8)-d_y(x,y-8)}{16\,\Delta t}.
\]

Here d_x and d_y are image-coordinate displacements. Both w and z reverse sign relative to image y, so their signs cancel in dw/dz. PIV requires a valid center and every intervening native ±4 and ±8 pixel node on the differentiation axis. Derivatives are computed on the Cartesian grid **before** sampling onto the local-depth display. Thus du/dx is not a derivative taken along a curved constant-depth contour. Valid native nodes beyond the displayed 1 cm boundary may support a derivative at its edge.

The two main derivative estimators have different smoothing. As an additional check, the exported data apply the identical ±8-pixel central difference to **both** methods at existing exact reporting-grid points, requiring the corresponding accepted optical-flow neighbors and valid PIV stencil. On the same stricter support, switching from the analytic optical-flow derivative to this matched operator reduces the RMS discrepancy by only about 1–5%. It does not resolve the gradient disagreement.

| Pair | Gradient | Shared displayed samples | Pearson correlation | RMS difference (s⁻¹) |
|---|---|---:|---:|---:|
'''+'\n'.join(gradrows)+r'''

The detailed gradients therefore have much weaker agreement than the integrated horizontal velocity. Spatially overlapping estimates are correlated, so sample counts are not independent statistical degrees of freedom. Gradient colors use symmetric limits at the pooled 99th percentile of absolute values; colorbar extensions indicate clipping. All extrema remain in the numerical exports.

## Calibration, geometry and reproducibility

The existing calibration is preserved: DX = 5.650454946380008 × 10⁻⁵ assumed m/pixel and DT = 0.01 assumed seconds. The files do not explicitly label these units. The full pixel-edge width is 2048 DX = 11.57213 cm. The inferred source surface remains `surfacePIVImg - 12` pixels; this convention has not been independently recovered from the masked images. Approximately 12 pixels, or 0.678 mm, below that inferred surface are absent in the TIFFs.

The comparison does not refit, initialize, tune or reject optical-flow predictions using PIV. SHA-256 checks verify that the existing image-only inputs, tracked particles, models, reporting fields and integral samples remain unchanged. Native sampling, coordinate/sign conversion, identical-support integration, analytic quadratic derivative checks and missing-value handling are independently checked in the included audit records.

Each pair folder contains MATLAB and NumPy exports for velocities, gradients and horizontal integrals, with explicit availability masks and assumptions. The integral CSV contains depth profiles; JSON records give operators, masks, source hashes and agreement metrics. In gradient exports, `of_*`, `piv_*` and `difference_*` two-dimensional fields are the plotted arrays; `matched_*` fields hold the additional equal-differentiation-operator check. In integral exports, `*_matched_*` are the primary curves, `*_common_*` the unavailable fixed-domain comparison, and `of_original_*` / `piv_own_*` the separate-coverage supplement. Missing comparisons remain NaN.
'''
    (OUT/'README.md').write_text(report)
    (OUT/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Report saved',OUT/'README.md')

if __name__=='__main__':main()
