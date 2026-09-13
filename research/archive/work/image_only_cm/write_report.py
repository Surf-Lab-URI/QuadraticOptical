"""Summarize the completed fresh analysis without using supplied velocities."""
from pathlib import Path
import json,hashlib,zipfile,shutil
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'outputs/image_only_top_cm'

def main():
    rows=[];details=[]
    for pair in [80,100]:
        folder=HERE/str(pair);s=json.loads((folder/'summary.json').read_text())
        with np.load(folder/'integration_profile.npz') as z:p={k:z[k] for k in z.files}
        ok=p['coverage_fraction']>0
        rows.append('| %d | %d / %d | %d | %d | %.3f | %.1f%% |'%(
            pair,s['accepted_grid'],s['grid_nodes'],s['gradient_grid_xx'],s['gradient_grid_yy'],
            s['accepted_grid_depth_range_px'][0]*s['DX']*1000,float(p['common_domain_fraction'])*100))
        t=json.loads((folder/'bootstrap_sensitivity.json').read_text())
        details.append('Pair %d: accepted coverage ranges from %.1f%% to %.1f%% over depths with any accepted segments. The fixed common domain is %.3f cm wide (%.1f%% of the image). The ±48/±64-pixel coarse bootstrap has %d common usable nodes differing by more than 0.1 pixel. This bootstrap diagnostic is not a final-field uncertainty bound.'%(pair,
            p['coverage_fraction'][ok].min()*100,p['coverage_fraction'].max()*100,
            float(p['common_domain_width_m'])*100,float(p['common_domain_fraction'])*100,t['differing_usable_over_point1_px']))
        details.append('The first supported integral sample for pair %d is at %.1f mm local depth and contains only %.2f%% of the horizontal width.'%(pair,p['depth_m'][ok][0]*1000,p['coverage_fraction'][ok][0]*100))
        branch_path=folder/'bootstrap_field_comparison.json'
        if branch_path.exists():
            b=json.loads(branch_path.read_text());g=b['grid'];i=b['integrals']
            details.append('Pair %d additional initialization check: fresh ±48-pixel tracking and a complete primary-quadratic refit changed accepted reporting-grid displacement by at most %.4f pixel. The largest integral differences on the identical primary horizontal segments were %.6f cm²/s (covered domain) and %.6f cm²/s (fixed common domain). Applying the existing finite-value and disagreement criteria to this extra field would remove %d reporting-grid vectors, %d du/dx estimates and %d dw/dz estimates. This branch has no separate reverse or three-variant repeat and is a sensitivity check, not a separately accepted complete field.'%(pair,
                g['accepted_displacement_difference']['maximum'],i['covered_abs_delta_assumed_m2_per_s']['maximum']*1e4,
                i['common_abs_delta_assumed_m2_per_s']['maximum']*1e4,g['primary_accepted_count']-g['fourth_alternative_vector_count'],
                g['primary_gradient_xx_count']-g['fourth_alternative_gradient_xx_count'],g['primary_gradient_yy_count']-g['fourth_alternative_gradient_yy_count']))
        dest=OUT/('pair_'+str(pair))
        for name in ['input_manifest.json','independent_inputs_audit.json','summary.json','tracking_summary.json',
                     'bootstrap_sensitivity.json','integration_metadata.json','independent_image_only_audit.json',
                     'independent_field_reconstruction.json']:
            if (folder/name).exists():shutil.copy2(folder/name,dest/name)
        for path in folder.glob('bootstrap*.json'):shutil.copy2(path,dest/path.name)
    text='''# Image-only velocities in the top centimetre

Pairs 80 and 100 were recomputed from their A/B TIFF images. **No supplied PIV velocity, displacement, correlation, manual endpoint, previous track, or previous fitted field was used in initialization, fitting, screening, or these plots.** Supplied PIV arrows are therefore absent. This run supersedes the earlier PIV-assisted predictions for this request.

Only four metadata datasets were read from each MAT file: `compVel/DX`, `compVel/DT`, `imSurfa/surfacePIVImg`, and `imSurfb/surfacePIVImg`. The first two supply scale and frame separation; the latter two supply geometry. Their group/dataset names include “PIV,” but no computed velocity array was loaded. Fresh-input hashes and exact reads are recorded in each input manifest and independently audited.

## Results and coverage

| Pair | Accepted vectors / tested grid nodes | du/dx estimates | dw/dz estimates | Shallowest accepted vector (mm) | Fixed common horizontal coverage |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'\n\n'+'\n\n'.join(details)+'''

The requested field spans the complete 2048-pixel horizontal image extent and the first 1 cm below the local frame-A surface. Reporting nodes are spaced 8 pixels (about 0.452 mm) apart. A 64-pixel fitting halo and an additional 25-pixel detector halo below the requested depth provide support; deeper fields are not part of the reported results. Grid spacing is not independent spatial resolution.

The tested-node counts above start at the 12-pixel minimum reporting depth and exclude the erased surface strip. They are not percentages of the entire requested 0–1 cm domain. The integral profiles separately report horizontal coverage at every depth across the full requested interval.

## Figures and data

- `quiver_comparison.png` / `.svg`: both pairs, with Cartesian arrow directions preserved and a common scale. Cyan marks the inferred local surface, orange marks 1 cm below it. Height is relative to the median surface ordinate in each image; the local surface is therefore slightly curved. The displayed arrows are thinned only for readability; all accepted reporting-grid vectors are in the data files.
- `quiver_deeper_half.png` / `.svg`: the same estimates between 0.5 and 1 cm, with a larger stated arrow gain to make slower motion visible. Small arrows in the overview do not imply that those locations were rejected.
- `du_dx.png` and `dw_dz.png` / `.svg`: derivative maps displayed against local depth. Values remain Cartesian derivatives, not derivatives in a flattened coordinate system. Gray cells are unreported. Each component uses the same color scale across both pairs, with percentile clipping explicitly indicated by colorbar extensions.
- `horizontal_integral_comparison.png` / `.svg`: accepted-segment integral, horizontal coverage and covered-width mean against local depth. Per-pair copies are in `pair_80/` and `pair_100/`.
- `velocity_gradients.mat` / `.npz`: all reporting-grid estimates, acceptance masks, diagnostics, alternative models and directly masked physical arrays. Raw `disp` and `gradient` arrays include rejected estimates for audit; use the acceptance masks or the explicitly masked physical fields for analysis.
- `horizontal_integral.mat` / `.npz` / `.csv`: depth profiles, coverage, common domain and model sensitivity. NPZ/MAT additionally retain sample-level support and interval contributions. Missing values remain NaN.

## What “hybrid” means in this run

1. Mask-normalized image preprocessing smooths at 0.65 pixels, removes a broad local mean at 7 pixels, and normalizes contrast using a 9-pixel local variance with a floor. It never restores the zeroed surface strip.
2. At every coarse grid location, normalized cross-correlation searches **±64 pixels in both image directions**, using only common valid image pixels. The three separated image peaks and zero displacement initialize separate affine image registrations. A fitted candidate must retain support, preserve orientation and have finite image correlation. The best admissible image fit supplies the local coarse map. A ±48-pixel search subset is evaluated as a bootstrap diagnostic. No velocity supplied by the user enters this step.
3. Candidate particles are local maxima of a difference-of-Gaussians response (0.6 and 2 pixels), with brightness/background, depth and image-support checks. A 9×9 image patch is matched near the fresh image-derived coarse prediction and refined continuously. Forward and backward patch correlation, match uniqueness and reciprocal displacement consistency select the tracks.
4. Robust local quadratic regression of accepted automatic tracks seeds the image registration. The final field consists of overlapping robust **quadratic displacement maps** fitted directly to normalized image intensity, with local gain/offset correction and regularization. Each 27×27-pixel primary window has a 12-parameter displacement map, allowing translation, rotation, shear, stretch and second-order variation. This is classical numerical registration; no U-Net/GOFLOW weights were trained or applied in this run.
5. A compact smooth blend of the local maps produces displacement and its analytic derivative. The derivative includes the derivative of the blending weights. Fresh affine, wider quadratic and tighter surface-mask fits quantify model sensitivity; a reverse image fit tests forward/backward composition.

Reflections are **not positively identified or individually labeled**. Actual TIFF availability, the geometric margin, particle/background checks, bidirectional matching and model consistency reduce unreliable observations. A reflection that satisfies these tests can still survive. The conservative mask is evidence of internal consistency, not a calibrated probability of correct physical particle identity.

The unchanged vector screens require sufficient nearby accepted tracks, inclusion in their convex hull, blended image correlation ≥0.6, forward/backward error ≤1 pixel, disagreement with all three alternatives ≤1.5 pixels, positive deformation determinant >0.2, sufficient valid-patch contribution in both directions, and visible source and target pixels. Source depth is at least 12 pixels. Each reported diagonal derivative additionally requires depth ≥20 pixels, an accepted track within 10 pixels, and componentwise alternative-model spread ≤0.08 pixel/pixel. No incompressibility constraint is imposed.

## Coordinates, gradients and local-depth integration

Image coordinates are x right and y down. The image displacement d=(d_x,d_y) is converted by u=DX·d_x/DT and w=−DX·d_y/DT, with physical z upward. If G is the full derivative of the displacement field with respect to initial image position, then du/dx=G00/DT and dw/dz=G11/DT. The two vertical signs cancel in dw/dz. These are finite-time, initial-position velocity-gradient estimates over one frame pair, rather than independently measured instantaneous gradients.

For local depth h, sample the field at y=s_A(x)+h/DX and calculate

    Q(h) = DX ∫ u(x, s_A(x)+h/DX) dx.

The integration measure is horizontal distance. There is no arc-length/surface-slope multiplier. This is a horizontal velocity integral at each depth, not an integral through water depth.

The full target width is the pixel-edge interval [−0.5,2047.5], or 11.5721 cm under the assumed calibration. Fixed 2-pixel Simpson intervals contribute only when both endpoints and the midpoint pass the original conservative vector screen. Missing intervals are not bridged, imputed, set to zero or silently renormalized. The plotted primary curve is the **integral over accepted horizontal segments**. Covered width and covered-width mean are separate outputs. A strict full-width integral remains NaN whenever any target interval is unsupported.

Because accepted width changes with depth, a second curve uses one fixed horizontal domain: the intersection of supported intervals at every sampled depth from 20·DX (about 1.1301 mm) through 10 mm. This domain can contain disconnected intervals. There is no nonempty common observed domain across the complete 0–10 mm range, since the shallow masked strip has no observations. Profiles are sampled every 0.1 mm with the 20·DX depth explicitly added; sampling density does not establish vertical resolution.

The fixed horizontal domain is computed separately for each image pair. It is constant with depth within that pair, but its width and intervals differ between pairs; the coverage panel states the retained fraction for each.

All alternative-model integrals use exactly the same accepted segments as the primary. Shaded envelopes describe **model sensitivity**, not statistical confidence intervals or bounds on experimental error. A nested 4-pixel versus 2-pixel Simpson comparison uses identical supported unions to check quadrature separately from changing coverage.

## Limits and validation

DX=5.650454946380008×10⁻⁵ and DT=0.01 are assumed to mean metres/pixel and seconds. The MAT datasets do not explicitly certify these units. Physical values in the figures and data are conditional on that interpretation.

The geometric surface is inferred as `surfacePIVImg−12 pixels`, following the earlier export convention. The retained TIFF boundary was independently checked against the rounded supplied trace. Approximately the first 12 pixels, or 0.678 mm, below the inferred surface are missing from the images. The algorithm cannot recover observations there. Surface-position uncertainty is not included in the model-sensitivity envelope.

Synthetic translation tests, analytic-versus-finite-difference derivative checks, exact fresh-image/mask audits, source-dependency checks, and independent integration bookkeeping verify implementation consistency. They do not substitute for experimental ground truth. No supplied PIV or manual velocity was used for tuning or validation in this recomputation.

The derivative maps retain appreciable fine spatial texture. Passing the internal screens does not establish that each small fluctuation is physical strain. Overlapping windows, shared image ambiguities and differentiation can produce correlated structure. No additional cosmetic smoothing was applied to hide this texture; numerical sensitivity values and the separate derivative masks are retained in the data.
'''
    (OUT/'README.md').write_text(text)
    print('Report written',OUT/'README.md')

if __name__=='__main__':main()
