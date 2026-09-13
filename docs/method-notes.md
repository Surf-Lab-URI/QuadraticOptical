# Method and interpretation notes

QuadraticOptical is a custom image-registration workflow for particle images with a supplied free-surface geometry. It preserves the conservative local quadratic model and records where the image evidence does not support a reportable result. It is not an implementation of a trained GOFLOW/U-Net model, wavelet optical flow, a global fluid solver or a reflection-object classifier.

See also the [method PDF](files/image_only_method.pdf), its [LaTeX source](files/image_only_method.tex), and the [research archive guide](../research/README.md). The 174 archived Python scripts are historical records; the supported implementation is `src/quadratic_optical/`.

## What enters prediction

The estimator uses only frame A, frame B, their image-availability masks, surface geometry, pixel calibration, A-to-B time delay and numerical settings. It computes a fresh initialization for each image pair. Supplied PIV displacements, manually picked velocities, IR surface velocities and earlier pairs' predicted fields are not used for initialization, fitting or acceptance gates.

A MAT file can contain both geometry and PIV. Preparation selectively reads only the requested calibration/surface leaves. This permits use of measured geometry without making the velocity estimate depend on supplied PIV. The prediction input schema rejects recognizable velocity/reference payloads. Comparison code is separate, reads completed fields, and verifies that frozen artifacts do not change during comparison.

## Image preprocessing and initialization

Availability masks are applied independently in A and B. The base fitting masks additionally exclude the first 10 pixels below each geometric surface. Intensities use a declared conversion to `[0,255]`; a mask-aware local normalization smooths with a 0.65-pixel Gaussian, removes a broader local mean, and divides by a local RMS contrast scale. Masked data are not treated as real dark particle observations.

The portable initializer performs a symmetric masked normalized-correlation search over `dx,dy` in `[-64,+64]` pixels, clipped to actual image bounds. It considers multiple separated peaks and a zero-displacement seed, fits local affine candidates and chooses image-supported candidates. A `±48` subset is retained in coarse diagnostics. This is a bounded initialization search, not a promise to recover arbitrarily large displacement or a full alternative final-field uncertainty calculation.

Difference-of-Gaussian particle candidates are detected at least 14 pixels below the source surface. Their image patches are matched forward and backward, with contrast, ambiguity and reciprocal-consistency checks. Accepted automatic tracks initialize a local polynomial motion estimate. Sparse or ambiguous image evidence can cause the run to fail rather than insert a supplied/theoretical velocity prior.

## Local affine and quadratic registration

For a patch centered at `c`, let `ξ=(x-cx)/r` and `η=(y-cy)/r`. Each displacement component uses the basis

```text
[1, ξ, η, ½ξ², ξη, ½η²].
```

The image warp is `T(x,y)=(x,y)+d(x,y)`. Its linear terms allow translation gradients, including rotation, shear and scale changes. Quadratic terms allow the displacement gradient to change across the patch. This is a deformation of the particle image pattern; the method does not assign a separately observed angular velocity to each individual blob.

The optimizer registers source intensities against bilinearly sampled target intensities. It fits a local brightness gain/offset and uses a smooth robust residual weighting to reduce the effect of unmatched particles. Gaussian spatial weights favor the patch center. A small penalty stabilizes nontranslation coefficients, with stronger penalties on quadratic terms; translation itself has no such ridge penalty. Damped Gauss–Newton updates use bounded movement and a line search that compares the same retained pixels and rejects excessive support loss.

The production family is fixed:

| Model | Patch half-width | Polynomial order | Excluded surface margin |
| --- | --- | --- | --- |
| Primary | 13 px | Quadratic | 10 px |
| Affine sensitivity | 13 px | Affine | 10 px |
| Larger-patch sensitivity | 19 px | Quadratic | 10 px |
| Surface-margin sensitivity | 13 px | Quadratic | 14 px |
| Reverse map | 13 px | Quadratic | 10 px |

These fits use regularization `5e-5` and at most 35 local iterations. Parameters are not tuned to supplied PIV or IR agreement. A radius-13 patch spans approximately 27 pixels before clipping/masking; a radius-19 patch spans approximately 39. Near a boundary, valid support can be asymmetric.

## Smooth field and analytic derivatives

Local models are blended within a 16-pixel radius using compact nonnegative weights proportional to

```text
w(t)=(1-t)^4(1+4t),  0 ≤ t ≤ 1,  t=distance/16.
```

The reported derivative differentiates both each local polynomial and its normalized blend weights. It is not a finite difference of plotted arrows, nor merely an average of local polynomial slopes. Missing/failed participating fits are not silently removed to manufacture a finite field.

Let `G[i,j]=∂d_i/∂q_j` in image coordinates, with component 0 horizontal and component 1 downward. The physical diagonal gradients are

```text
du/dx = G[0,0]/DT
dw/dz = G[1,1]/DT.
```

Both w and z reverse sign relative to the image vertical convention, so the second formula has a positive sign. A depth-rectified heatmap still shows these Cartesian derivatives. Differentiating the velocity along a sloping, constant-local-depth path would produce a different derivative and is not what the `du/dx` array represents.

## Conservative reporting gates

Velocity is reported only where all of the following hold:

| Check | Threshold/requirement |
| --- | --- |
| Source local depth | At least 12 px and no deeper than the requested depth. |
| Mapped target local depth | At least 10 px. |
| Actual source/target visibility | Bilinear fitting-mask availability above 0.99; target center lies inside the solver's interpolation domain. |
| Automatic-track support | Inside the accepted-track convex hull, nearest accepted track within 12 px, at least six within 25 px. |
| Patch image agreement | Blended patch NCC at least 0.6. |
| Forward/backward composition | `||d_A(q)+d_B(q+d_A(q))|| ≤ 1` px. |
| Displacement sensitivity | Maximum difference from the three forward variants at most 1.5 px. |
| Local orientation | `det(I+G)>0.2`. |
| Nearby fit quality | At least 95% of blending weight comes from qualifying forward and reverse fits. |
| Finite values | Main displacement and all required variant values/gradients are finite. |

For the fit-quality share, a contributing patch qualifies when its minimum sampled deformation determinant is above 0.05 and retained weighted support is above 0.85. The final NCC is a blend of neighboring **patch** NCC diagnostics; it is not a newly computed NCC for the entire blended warp at each query.

Each diagonal derivative has its own additional mask: accepted velocity, source depth at least 20 px, nearest accepted track within 10 px, and maximum disagreement of that same diagonal derivative across variants at most `0.08` per pair. Consequently `du/dx` and `dw/dz` coverage can differ. Other tensor components do not inherit these two component-specific guarantees.

Only the requested maximum-depth arithmetic includes a `1e-9` pixel tolerance to avoid losing an exact endpoint through floating-point subtraction. This does not relax any image-evidence or minimum-depth threshold. Extra fitted depth (`requested depth +64 px`) supplies support; it does not enlarge the reported depth range.

Accepted does not mean known ground truth. The gates detect particular failures and sensitivities; coherent reflections or an incorrect surface convention can still produce plausible image matches. At the historical calibration, 12 and 20 pixels correspond to approximately 0.678 mm and 1.130 mm, respectively. These physical distances change with `DX`.

## Horizontal integrals versus local depth

At requested depth `h` metres below the local source surface `s_A(x)`, samples lie at

```text
q(x,h) = [x, s_A(x) + h/DX].
```

The horizontal integral is

```text
Q(h) = DX ∫ u(q(x,h)) dx_pixel.
```

There is no surface arc-length factor. With `u=DX*d_x/DT`, the equivalent expression is `DX²/DT * ∫d_x dx_pixel`. `Q` has units m²/s, and is an integral along a horizontal-coordinate parameterized path; it is not a two-dimensional volume flux without further geometry.

The default target is the full pixel-edge width `[-0.5,width-0.5]`. Each nominal 2-pixel interval uses its two endpoints and midpoint with Simpson weights `1,4,1`. A shorter final interval is retained for an odd image width. All three samples must be finite and accepted. Missing intervals are excluded explicitly, without connecting across gaps or inserting zero velocity.

The output distinguishes:

| Quantity | Meaning |
| --- | --- |
| Covered integral | Sum over the accepted intervals at this depth. |
| Covered width and fraction | Physical horizontal width actually used, divided by the target width for the fraction. |
| Covered-width mean | Covered integral divided by covered width. |
| Full-width integral | Reported only if every target interval is supported; otherwise NaN. |
| Fixed common-domain integral | Uses one potentially disconnected horizontal intersection across the declared depth band. |

An empty domain has zero width and NaN integral/mean. The fixed-domain band starts at 20 pixels and ends at the requested depth. The common domain across *all* depths including `h=0` is generally empty because the surface is masked or unsupported. No surface-velocity extrapolation is hidden in the profile.

Because accepted width changes with depth, changes in covered `Q` can reflect both changing velocity and changing support. Examine the width/coverage and mean curves, or a nonempty fixed-domain curve. Variant envelopes use the same primary support and are model sensitivity, not confidence intervals. A nested quadrature check compares fine/coarse integration only on the same supported interval unions.

## PIV comparison

PIV is read only after prediction. It is sampled at the same saved source locations, using exact native values where grids coincide and strict bilinear interpolation otherwise. Every positive-weight contributor must satisfy the selected quality/source mask; no extrapolation or missing-value replacement is used. Default quality requires finite correlation. Explicit `finite` quality reproduces the earlier saved-finite-vector comparison policy.

PIV diagonal gradients use centered `±8` pixel differences, giving a 16-pixel baseline. At 4-pixel native spacing, the center and all four intermediate/endpoint nodes (`−8,−4,0,+4,+8`) must be valid along that axis. Other spacings use exact physical pixel endpoints with linear interpolation and require every intervening native node and bracket. The derivative is computed on the Cartesian native grid before sampling at the depth-rectified plotting positions.

The main comparison retains analytic OF gradients and finite-difference PIV gradients, so their regularization differs. A separate matched-operator audit applies the same `±8` difference to saved OF displacements at exact available Cartesian neighbors. Center/neighbor acceptance is required; a stricter recorded subset also retains the original OF component mask. This audit does not replace the primary analytic derivatives.

Joint depth profiles use the same frozen Simpson samples. An interval contributes to both methods only when all three samples pass both availability policies. OF and PIV therefore use identical width **at each depth**, but that joint width can still change with depth. A separate fixed joint-domain profile is supplied if its horizontal intersection is nonempty. Default finite-correlation screening may leave that fixed domain empty; the variable joint profiles remain meaningful on their explicitly reported width.

The report places changing-support profiles in the upper row and the fixed-domain profiles in the lower row. An empty lower-row domain is labelled explicitly; it is not replaced with zeros or a different horizontal support.

PIV differences are not ground-truth errors. The fields share the same images, and the saved native PIV may already contain upstream processing. No discrepancy gate is used to make the comparison appear closer. Changing quality policy changes the comparison support, never the frozen OF fit.

## IR surface comparison

An optional surface star uses the explicitly mapped IR record, in downstream-positive m/s. A raw fresh-dot value is preferred; a filtered fallback is identified as smoothed. The package does not convert a point surface velocity into a missing full-width integral, apply a free-surface kinematic boundary constraint, enforce no-slip, or impose 2-D incompressibility. IR and PIV observations do not extend the accepted OF domain.

The exact IR selection is stored and reused by later comparisons unless a new record is explicitly selected. Repeating the same star in both profile rows does not make it a second observation or an estimate of either subsurface horizontal mean.

## Validation and practical limits

Synthetic tests independently generate signed shifts and a known quadratic image deformation, check local recovery, differentiate the blended field, verify mask/coordinate conventions and enforce resume signatures. Additional tests validate MATLAB formats, default/finite PIV quality, strictly supported interpolation, component masks, Simpson gap behavior and IR indexing. A small full pipeline test runs without experimental files.

The portable finite-quality comparison was also checked against the earlier frozen pairs 80/100: the gradient/mask and integral/mask arrays were identical, including missing values. This is a regression check for the comparison implementation. The portable initializer's current `±64` search and generic metadata conventions should not be confused with a bitwise reproduction of every earlier research run; use frozen fields and their recorded source/settings when reproducing a particular result.

The fitted patch dimensions and blend radius limit spatial resolution. A finer output grid or dense color texture does not resolve smaller physical scales. Intensity changes, saturation, reflection patterns, particle loss, occlusion and out-of-plane motion remain potential model errors. Adding higher-order polynomial terms is not automatically beneficial when the surviving image information is weak. These outputs should be interpreted with their masks, sensitivity diagnostics, source images and known experimental geometry.
