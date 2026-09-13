"""Write a concise scientific record from completed full-width results."""
from pathlib import Path
import json
W=Path('work/full_width');O=Path('outputs');P='full_width_conservative_'
s=json.loads((W/'summary.json').read_text());m=s['manual_metrics']
depthrows='\n'.join('| {low:g}–{high:g} | {evaluated:,} | {accepted:,} | {gradient:,} |'.format(**r) for r in s['depth_bins'])
widthrows='\n'.join('| {x_low}–{x_high} | {evaluated:,} | {accepted:,} | {gradient:,} | {near40_accepted:,} |'.format(**r) for r in s['horizontal_bins'])
text=f'''# Conservative optical flow over the whole image width

The original conservative quadratic method has been extended across all 2,048 image columns, with the same 354-pixel depth limit below the traced surface. **{s['accepted_grid']:,} of {s['grid_nodes']:,} evaluated grid locations pass the vector screen**, and **{s['gradient_grid']:,} pass the horizontal-gradient screen**. The 8-pixel grid spans full-image x = 7–2047; coverage at image edges is determined by actual image support rather than extrapolation beyond the TIFF.

If the supplied calibration is SI, the field of view is approximately **{s['full_width_cm_if_SI']:.2f} cm wide**, and the depth limit is **{s['depth_cm_if_SI']:.2f} cm**. The MAT contains DX = {s['DX']:.12g} and DT = {s['DT']:.12g} without unit labels. Pixel quantities remain primary. Conditional metre/second fields are explicitly named `assumed` and carry `physical_units_confirmed = 0`.

## Plots and numerical products

- [Full-width quiver plot](full_width_conservative_quiver.png).
- [Four overlapping enlarged views](full_width_conservative_quiver_details.png).
- [Horizontal gradient and sensitivity](full_width_conservative_horizontal_gradient.png).
- [Comparison with the existing manual matches](full_width_conservative_manual_comparison.png).
- [MATLAB data](full_width_conservative_field.mat), [NumPy data](full_width_conservative_field.npz), [grid table](full_width_conservative_grid.csv), and [manual table](full_width_conservative_manual.csv).

Each plot also has an SVG version. Quiver arrows show actual A-to-B displacement scale. Cyan dots identify accepted source locations even when the displacement is too small for a visible arrow. Crosses mark rejected grid locations. Purple shading follows the actual first-valid-pixel boundary, including the geometric exclusion and any additional unavailable glare pixels. For a readable background, pixels outside the fitting mask are hidden; no image texture was synthesized.

## Coverage across the frame

| Horizontal interval, px | Evaluated | Accepted vectors | Accepted horizontal gradients | Accepted vectors within 40 px of surface |
|---|---:|---:|---:|---:|
{widthrows}

Horizontal intervals are half-open, for example 0–512 means x < 512. The nearest accepted source depth is {s['min_accepted_grid_depth']:.3f} px, but this minimum describes a local part of the surface. It does not imply equally close recovery beneath every depression. The minimum depth for an accepted horizontal gradient is {s['min_gradient_grid_depth']:.3f} px.

| Depth below frame-A trace, px | Evaluated | Accepted vectors | Accepted horizontal gradients |
|---|---:|---:|---:|
{depthrows}

The depth limit remains unchanged from the preceding analysis; only horizontal scope expands. Accepted grid x coordinates range from {s['accepted_grid_x_min']:g} to {s['accepted_grid_x_max']:g}; the last two grid columns remain withheld. Twenty-seven accepted locations at x = 7 use one-sided image windows at the left boundary. They pass the same image tests but have no separate manual edge validation. Where particles or valid image data are insufficient, the model remains withheld.

## Validation and what is preserved

Only the existing 200 manual matches around the first depression are available. They were not used to fit the new fields. There are no additional manual labels for the remaining depressions or the deeper extension.

| Manual subset | Picks | Mean endpoint disagreement, px | Median, px | RMS, px |
|---|---:|---:|---:|---:|
| All manual matches | {m['all']['n']} | {m['all']['mean']:.5f} | {m['all']['median']:.5f} | {m['all']['rmse']:.5f} |
| Passing the vector screen | {m['all_screened']['n']} | {m['all_screened']['mean']:.5f} | {m['all_screened']['median']:.5f} | {m['all_screened']['rmse']:.5f} |
| Within 40 px of surface, all | {m['near40']['n']} | {m['near40']['mean']:.5f} | {m['near40']['median']:.5f} | {m['near40']['rmse']:.5f} |
| Within 40 px, passing screen | {m['near40_screened']['n']} | {m['near40_screened']['mean']:.5f} | {m['near40_screened']['median']:.5f} | {m['near40_screened']['rmse']:.5f} |

The maximum prediction change at these manual positions from the preceding narrower analysis is {s['previous_manual_max_change_px']:.3g} px; {len(s['previous_manual_changed_picks'])} picks change by more than 1e-9 px. There are {s['previous_manual_acceptance_changes']} changes in the manual-location acceptance flags. The same image pair informed earlier exploratory method comparisons, so this is not an untouched test set. Passing the image screen is not a calibrated probability of accuracy.

All 2,184 previous local polynomial fits, their three comparison fields, and their reverse fits were retained exactly. All previous grid acceptance flags are unchanged. New neighbors change 86 blended predictions near the previous grid boundary; the maximum change across those grid positions is 0.146 px, and the maximum among previously accepted positions is 0.116 px. Preserving polynomial coefficients does not require identical boundary evaluations after adding neighbors. All 3,748 previous automatic-particle candidate records and their 3,204 accepted flags are retained; the full field uses {s['accepted_features']:,} accepted automatic tracks in total. The full-width fit adds 8,762 grid locations.

The supplied `compVel` field provides only initial guesses for new coarse image fits. Reciprocal particle tracking, local image registration, and final image checks determine acceptance. Agreement with that supplied PIV field would therefore not be independent validation. No ML model or pretrained checkpoint was introduced in this extension.

One pair of automatic tracks had nearly coincident predicted targets. The suspect branch at source (265,430), and its nearest field grid point, fail the original correlation and variant-stability tests. A neighboring velocity estimate passes the checks but its horizontal gradient is withheld. The raw tracking flags were not manually edited; the detailed local audit is included in the bundle. Without a manual reference there, this check establishes internal consistency rather than the true local error.

## Masks and coordinates

Every column of both 2048-by-2048 TIFFs was checked. Their first nonzero row matches the rounded `surfacePIVImg` in the new MAT. That trace is exactly 12 px below the geometric trace used in the original analysis and marks an already-applied glare mask; it is not used as an additional physical surface. This preserves the earlier coordinate convention; the new mask does not independently refine the physical surface location.

The original 501-by-501 TIFFs align exactly at full-image zero-based origin (299,339). Their retained raw pixels were used inside that overlap, preserving the earlier near-surface evidence. Outside that crop, only the actually available pixels in the larger TIFFs are used. The previous ROI's raw pixels, masks, and surface geometry are unchanged.

Primary coordinates are full-image pixels, x right and y downward. Displacements are B minus A. One-based coordinates are also exported for MATLAB. The image ROI used for fitting is x = 0–2047 and y = 260–839; this contains the whole requested depth band plus fitting context.

## Conservative method

The method remains the original automatic-particle prior followed by robust, masked local quadratic registration with bilinear image sampling. Each displacement component has six spatial coefficients: translation, first derivatives, and second derivatives. The primary window is 27-by-27 pixels. Comparison models use an affine map, a 39-by-39 quadratic window, and a 14-pixel surface exclusion. The continuous field uses the original compact C2 blend with radius 16 px.

The original evidence tests are retained: source depth 12–354 px; target depth at least 10 px; accepted-particle convex-hull support; nearest accepted particle within 12 px; at least six accepted particles within 25 px; correlation at least 0.60; composed forward–backward error at most 1 px; disagreement across all three variants at most 1.5 px; map determinant above 0.20; and at least 95% blending weight in both directions from patches with overlap above 0.85 and minimum patch-map determinant above 0.05. All three comparison fields must be available.

Because the analysis now reaches the image boundaries and regions with pre-masked pixels, source and target centers are also required to lie in the actual valid image domain. Bilinear validity interpolation must exceed 0.99. Target coordinates follow the registration solver's own boundary convention (at least 1 px, strictly below image size minus 2). This explicit domain guard excludes {s['mask_visibility_exclusions_grid']} otherwise qualifying grid locations; it does not relax any evidence threshold.

For the horizontal derivative, source depth must additionally be at least 20 px, the nearest accepted particle must be within 10 px, and variant disagreement in the horizontal-horizontal derivative must be at most 0.08 px/px. This screen applies only to that derivative; the other raw tensor entries have no equivalent sensitivity guarantee.

Exact target-image gradients and floating-point mask arrays were cached to avoid redundant full-frame calculations. Independent edge, surface, affine, quadratic, forward, and reverse cases verified identical coefficients and diagnostics to the original solver. Caching changes runtime, not the fitted mathematical model.

## Velocity and horizontal-gradient conversion

If DX is metres/pixel and DT is seconds:

`U = d_x * DX / DT`

`V_up = -d_y * DX / DT`

`dU/dX = (d d_x / d x) / DT`

One pixel per pair corresponds to 0.00565045 m/s; divide the pixel displacement gradient by 0.01 s for the conditional velocity-gradient estimate. The derivative is taken at fixed image height, not along the sloping surface or a constant-depth contour. It includes derivatives of the blending weights, so it differentiates the same field shown in the quiver plots.

Independent checks at all 10,098 accepted grid positions found agreement between the analytic horizontal derivative and central finite differences to within 7.53e-9 px/px, using a 0.001-pixel difference step. This validates the numerical differentiation, not the physical accuracy of the inferred gradient.

Displacement divided by frame interval is an interval-averaged material-motion estimate attached to the frame-A position. Its derivative is a finite-time velocity-gradient proxy, not a directly resolved instantaneous Eulerian derivative. Variant sensitivity is not a confidence interval, and no independent gradient ground truth is supplied. The 8-pixel fit spacing and 4-pixel dense evaluation spacing are sampling choices, not spatial-resolution claims.

Use the arrays containing `conservative` for plotting: rejected values are NaN. Raw predictions, complete diagnostics, acceptance masks, manual references, particle tracks, and surface traces remain available for inspection. Previous deliverables were left unchanged.
'''
(O/(P+'report.md')).write_text(text);print('Full-width report written.')
