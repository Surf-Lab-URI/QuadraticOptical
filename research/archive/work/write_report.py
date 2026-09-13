import json,numpy as np
from pathlib import Path
s=json.load(open('work/final_summary.json'));a=s['metrics']['all'];near=s['metrics']['near40']
text=r'''# Near-surface velocity reconstruction — ExpLCL_1_03, pair 123

The selected estimate is a **hybrid of automatic particle tracking and quadratic optical flow**. It recovers the near-lump particle motion substantially better than the translation-only correlation baseline tested here. It does not establish the velocity exactly at the free surface.

The main result is [the side-by-side quiver comparison](quiver_comparison.png), with all 200 manual vectors and model predictions at precisely the same starting locations. [The overlay](quiver_overlay.png) makes individual discrepancies visible. Orange denotes manual picks, cyan denotes predictions that pass the provisional image-only screen, and gray denotes flagged predictions. Arrows use the true A→B displacement scale.

## What the comparison supports

| Method | Mean disagreement, all 200 (px) | Median, all 200 (px) | Mean within 40 px of surface, 26 picks (px) |
|---|---:|---:|---:|
| Translation-only masked NCC, 39×39 px windows | 2.915 | 0.632 | 12.192 |
| High-pass DIS optical flow | 1.090 | 0.389 | 5.46 |
| Automatic particle tracks + local quadratic reconstruction | 0.588 | 0.440 | 0.940 |
| Cubic spline image registration, intermediate bending penalty | 0.691 | 0.478 | 1.586 |
| Affine image refinement, same particle initialization and blending | 0.539 | 0.430 | 0.776 |
| **Selected quadratic image refinement, 27×27 px windows** | **0.497** | **0.393** | **0.665** |
| Quadratic refinement, 39×39 px windows | 0.561 | 0.466 | 0.810 |
| Quadratic refinement, 14 px surface exclusion | 0.494 | 0.399 | 0.642 |

The selected model's RMS endpoint disagreement is **0.641 px**, its 90th percentile is **0.906 px**, and **183/200** predictions lie within 1 px of the manual vectors. For the 26 picks within 40 px of the surface, the median is **0.447 px**, RMS **0.844 px**, and 90th percentile **1.460 px**. The 14 px mask variant is effectively comparable; the default 10 px exclusion retains the possibility of fitting closer to the boundary.

These are discrepancies from manual selections, not errors against perfect ground truth. No manual endpoint, displacement, or manually selected source position was used to detect particles, track them, construct the prior, or fit the final field. Manual positions were used afterward to query the field. However, the same 200 matches informed comparison and selection among methods, so this is an **exploratory validation on one pair**, not an untouched holdout test. No manual picks were silently removed from the all-200 scores. Closely spaced picks also make the errors statistically correlated.

The NCC reference is a transparent single-pass translation-only baseline with subpixel peak refinement; it is not a claim to have benchmarked every optimized or commercial PIV implementation. Modern correlation PIV can itself incorporate image deformation.

## How close to the surface?

- The closest manual source point is **16.71 px** below the A surface. Its manual displacement is **(11.087, −5.390) px**; the model gives **(10.820, −5.623) px**, a **0.355 px** discrepancy. It remains flagged by the stricter local-warp support screen despite this favorable agreement.
- The second closest is **17.27 px** below the surface. Manual: **(11.395, −6.929) px**; model: **(12.603, −7.940) px**. Its discrepancy is **1.575 px**, and it is flagged.
- Automatic matching produces **675 accepted reciprocal particle tracks** from 996 image-detected candidates. The closest starts **14.35 px** below the surface. These sparse correspondences have image checks but **no manual validation at depths shallower than 16.71 px**. See [near-surface candidate tracks](near_surface_candidate_tracks.png).
- On the 8 px output grid, the screened continuous field begins at **19.02 px** depth at its closest retained node; this is a grid-dependent minimum, not a uniform boundary across x. It retains **536/727 nodes** and **185/200 manual query locations**, including **16/26** near-surface manual locations.
- The gradient is screened separately and is deliberately not reported within **20 px** of the surface. **513/727** grid nodes pass its screen.

The first approximately 10 px below the supplied trace contain predominantly saturated reflection/glare: depending on the sub-band, roughly 70–85% of A pixels and 48–80% of B pixels are saturated. Information lost to saturation cannot be recovered by increasing the deformation order. The broad uncolored and crossed-out regions in the figures are intentional.

The surface depression's deepest point moves approximately **58 px** between A and B. The manual particle horizontal displacements range from **−1.21 to +23.79 px**. Consequently, matching the moving bright surface or its lowest point would estimate wave-shape motion rather than particle velocity. Near-circular individual particle spots also contain little uniquely identifiable rotation; deformation of a neighborhood of particles is the more useful signal.

## Coordinates, physical units, and region

Only the capillary-lump neighborhood is reported: approximately **x = 60–390 px, y = 0–180 px**, with retained field depths limited to **150 px below the A surface**. Larger image support is used where matching requires it. This contains all 200 manual source points. The exact requested 1–2 cm crop cannot be established without a spatial calibration.

The MATLAB file contains the 200 pairs, A/B surface traces, crop offsets, and mask offset, but **no time interval or physical pixel scale**. Therefore the directly supported output is displacement in **pixels per image pair**. Assume A precedes B. Internal and exported image coordinates are zero-based, x rightward and y downward. MATLAB's saved `p` is converted with `p−1`; displacement is unchanged. Averaging particle-centered patches verifies that conversion. The saved original-image crop offsets have an apparent one-pixel bookkeeping ambiguity, so absolute surface distances should not be interpreted more accurately than approximately one pixel.

The mask TIFFs are images with excluded pixels zeroed, not binary images of a different scene. Their retained intensities exactly match the raw TIFFs. The alternative mask begins approximately 10 px below the converted surface. Masks are used as validity regions; the artificial zero-filled mask edge is not used as an image feature.

For isotropic calibration ℓ (e.g. metres/pixel) and frame interval Δt (seconds):

- Horizontal velocity estimate: **U = ℓ dₓ / Δt**.
- Upward-positive vertical velocity estimate: **V = −ℓ dᵧ / Δt**.
- Horizontal velocity gradient at fixed laboratory height: **∂U/∂X = (∂dₓ/∂x) / Δt**.

These are finite-time, path-averaged velocity estimates from a two-frame displacement. The default field is indexed by the source A coordinates. It is not an exact instantaneous Eulerian velocity field. The MATLAB file also includes trajectory-midpoint coordinates and the chain-rule displacement gradient there, **G_mid = G_A (I + G_A/2)⁻¹**, for users who prefer a midpoint representation. A derivative along the sloping surface is different from the requested horizontal partial derivative at fixed height.

## Final estimator

1. **Preprocessing and coarse image-only seeds.** Compare dense DIS/Farneback/Lucas–Kanade variants and masked correlation seeds. Local affine image registration supplies coarse motion, shear, rotation and normal-strain information. Bright surface contamination is suppressed. Initial correlation searches cover x shifts −15…55 px and y shifts ±20 px; no manual displacements set those candidates.
2. **Automatic particle correspondences.** Detect peaks of a difference of Gaussians with a 5×5 local-maximum test, requiring depth ≥14 px and excluding very bright background. Match 9×9 px patches over a 17×17 displacement search around a robust median of nearby affine extrapolations, followed by subpixel optimization. The score combines normalized image agreement and a weak squared displacement-prior penalty. Reverse tracking, correlation and distinct-candidate separation retain 675 tracks. False correspondences can still pass these checks.
3. **Coherent prior.** A robust moving quadratic fit to accepted automatic tracks supplies the starting displacement and affine derivative for each grid location. This step resolves repeated-particle ambiguities that defeated unconstrained window matching, especially upstream.
4. **Quadratic image refinement.** In each 27×27 px window, optimize the full two-component spatial warp using the basis **[1, ξ, η, ξ²/2, ξη, η²/2]**, where ξ and η are coordinates relative to the window center, divided by radius 13 px. There are six coefficients per component. The linear terms include rotation, shear and extension; the quadratic terms allow these gradients to change across the window. Use mask-aware local intensity normalization, jointly updated local brightness gain/offset, a robust pseudo-Huber loss, and weak coefficient regularization. The source and warped target samples must be in liquid. Default regularization is 0.00005 times summed spatial weights on affine coefficients and four times that on quadratic coefficients; this is a weak ridge, not a pure curvature-only prior.
5. **Continuous field and derivatives.** Smoothly blend overlapping local polynomial predictions with the compact weight **w = (1−r/R)⁴(1+4r/R)** for r&lt;R, R=16 px. Differentiate this same blended field analytically, including derivatives of the blending weights. This avoids treating independently fitted local slopes as though they were automatically derivatives of an interpolated vector map.

The 8 px vector grid and 2 px rendered raster are sampling choices, **not independent spatial resolution**. The estimator uses 27×27 px image windows, neighboring particle support, and a 16 px blending radius. Sub-window-scale structures in the gradient should not be interpreted as independently resolved flow features.

No no-slip condition, zero surface velocity, 2D incompressibility, pressure, or unverified dynamical constraint was imposed. No pretrained neural network or exact published wavelet solver is being claimed here.

## Confidence screen and horizontal gradient

The provisional image-only screen requires: source depth 12–150 px; mapped B depth ≥10 px; position inside the accepted-feature convex hull; nearest accepted particle ≤12 px; at least six accepted particles within 25 px; blended local NCC ≥0.60; correctly composed forward–backward discrepancy ≤1 px; maximum displacement change across the affine/larger-window/larger-mask variants ≤1.5 px; positive blended mapping determinant >0.2; and all three alternatives available. At least 95% of the forward and reverse blending weight must come from local fits with positive Jacobian determinant throughout their image support and retained support >85%. Failed alternative fits are flagged as missing sensitivity evidence, not ignored.

These thresholds produce a practical screen, **not a calibrated probability or error bar**. A wrong match can be reciprocal. The screen is based on images and support, not on each point's manual discrepancy. The screened all-185 mean disagreement is **0.452 px**; the screened near-surface subset is smaller and does not necessarily have a lower mean than all 26 near-surface picks.

[The horizontal-gradient map](horizontal_gradient.png) shows **∂dₓ/∂x** and its maximum change across the three variants. It additionally requires depth ≥20 px, nearest particle ≤10 px, and gradient variation ≤0.08 px/px. Across retained grid nodes, the 1st–99th percentile is approximately **−0.154 to +0.157 per pair**, and median displayed sensitivity is approximately **0.016 per pair**. Divide by Δt for s⁻¹. This sensitivity excludes unknown optical distortion, picking uncertainty, and common biases shared by the methods; it is not a confidence interval. There is no direct manual gradient ground truth. Fine alternating structures and the boundary values should therefore be treated as exploratory.

The separately tested cubic-spline flow also showed substantial near-surface regularization sensitivity. This is a reason to avoid reporting a single precise surface strain rate from this pair, even though a visually smooth field can be produced.

## Verification and method context

Synthetic verification used eight independently generated Gaussian-particle textures and known affine/quadratic maps, with intensity changes and noise. Across 64 fits, signs and coefficient scaling were correct. For truly quadratic motion, the quadratic model reduced center-displacement errors from approximately 0.33–0.52 px for an affine fit to 0.019–0.030 px. Adding quadratic terms slightly worsened some truly affine cases despite improving image correlation. These tests verify the implementation; they do not reproduce the free-surface optics in this experiment.

The final blended analytical derivatives agree with finite differences to better than 3×10⁻⁸ on the checked points. Independent round-trip checks verified the CSV, MATLAB and NumPy displacement arrays, coordinate conversion, gradient tensor order, midpoint transform, screening flags and summary statistics (176 checks passed).

Published work informed the method choices. Iterative image deformation is an established PIV extension rather than a replacement for every use of correlation: [Scarano, 2002](https://research.tudelft.nl/en/publications/iterative-image-deformation-methods-in-piv-2/). Near-wall wavelet optical flow demonstrates the importance of regularization for resolving gradients; excessive smoothing can suppress them, but a solid-wall benchmark does not validate a moving free surface: [Nicolas et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9944726/). Recent neural optical flow fits a continuous field through nonlinear image warping and regularization; it is a relevant alternative, but was not reproduced here: [Masker et al., 2025](https://link.springer.com/article/10.1007/s00348-025-04058-1). Recent interface-focused wavelet work also discusses ambiguity and limitations of image-warp residuals as accuracy surrogates: [Wilson et al., 2026](https://link.springer.com/article/10.1007/s00348-025-04152-4).

## Files and reproducibility

- **quiver_comparison.png / .svg** — requested manual-versus-model quiver plot.
- **quiver_overlay.png / .svg** — superposed vectors at the same 200 locations.
- **predicted_field.png / .svg** — screened regular-grid field.
- **horizontal_gradient.png / .svg** — screened horizontal derivative and method sensitivity.
- **near_surface_candidate_tracks.png / .svg** — sparse provisional tracks closer to the surface.
- **validation.png / .svg** — endpoint disagreements versus depth and cumulative method comparisons.
- **velocity_results.mat** — MATLAB-readable fields, masks, manual comparisons, derivatives, midpoint representation and automatic tracks. Unknown calibration values are NaN.
- **velocity_results.npz**, **manual_comparison.csv**, **predicted_grid.csv**, **automatic_particle_tracks.csv**, **validation_summary.json** — numerical outputs and diagnostics. Unscreened estimates are preserved with explicit flags.
- **analysis_code_and_checkpoints.zip** — input copies, final-stage source, frozen image-only coarse seeds, final model checkpoints, and a runner. The final tracking/refinement/export pipeline can be rerun from those frozen seeds. Historical coarse-seeding and baseline checkpoints are supplied explicitly; this is not a claim of a single untouched end-to-end implementation of a published algorithm.

The original files were not modified. The remaining information needed for physical velocity, strain-rate units, and an exact 1–2 cm crop is the **pixel scale and A-to-B time interval**.
'''
Path('outputs/analysis_report.md').write_text(text)
print('Wrote report',len(text.split()),'words')
