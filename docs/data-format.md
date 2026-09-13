# Data formats and coordinate conventions

This document describes what the package reads, how coordinates are converted, and which files may affect image-only prediction. Configuration examples are in [`examples/`](../examples/); processing commands are in the [README](../README.md).

## Image names and dimensions

Discovery recognizes paired stems with `_imgA`/`_imgB` or `_a`/`_b` suffixes, case-insensitively, and `.tif`, `.tiff` or `.png` extensions. For example, `Run_80_imgA.tif` and `Run_80_imgB.tif` form `Run_80`. Pairing never depends on directory order or timestamps. Duplicate candidates for a frame are rejected. Multi-page TIFFs and RGB images are rejected rather than silently converted or split.

Both frames must have the same two-dimensional shape `[height,width]`, with at least 48 pixels on each axis. Shapes need not be square. Processing remains in the images' supplied pixel coordinates: there is no hidden resize or crop. The requested depth limits the reported field; a deeper fitting halo remains available for local support.

When recursive discovery is enabled, a pair in `subdir/Run_80_imgA.tif` is named `subdir__Run_80` in output and configuration. Companion files remain beside their source images, named with the original stem:

```text
Run_80_PIV.mat
Run_80_surface.npz
```

## Calibration and axes

| Quantity | Convention |
| --- | --- |
| Source/target pixel coordinates | Zero-based `(x,y)`, x to the right and y downward; arrays use `[y,x]`. |
| Displacement `disp[...,0]` | Positive rightward pixels from A to B. |
| Displacement `disp[...,1]` | Positive downward pixels from A to B. |
| `DX` / `dx_m_per_px` | Metres per pixel; scalar, positive and finite. |
| `DT` / `dt_s` | Seconds from A to B; scalar, positive and finite. |
| Physical `u` | `disp_x * DX / DT`, rightward m/s. |
| Physical `w` | `-disp_y * DX / DT`, upward m/s. |
| Physical `du/dx` | `gradient[...,0,0] / DT`, s⁻¹. |
| Physical `dw/dz` | `gradient[...,1,1] / DT`, s⁻¹; the two vertical sign changes cancel. |
| Local depth `h` | Vertical distance below the source-frame surface at that x, increasing downward. |

Pixel-center locations plotted from the left image edge are `(x+0.5)*DX`. Displayed height can use the median surface ordinate as a reference: `z=(median(surface_a)-y)*DX`. Changing this constant reference does not change velocities or gradients.

The package does not infer the physical units of an unlabeled scalar. Supplying or accepting metadata calibration means confirming metres/pixel and seconds. An exposure interval, between-pair cadence, or IR frame period is not interchangeable with the A-to-B delay. Native PIV comparison requires its saved calibration to agree with the frozen image-only calibration.

## JSON configuration

Unknown top-level keys are rejected. Values under `pairs` override the base configuration for exactly one discovered output name. JSON does not support comments.

| Key | Default | Meaning |
| --- | --- | --- |
| `dx_m_per_px` | `null` | Explicit calibration; null reads `compVel.DX` from metadata. |
| `dt_s` | `null` | Explicit A-to-B delay; null reads `compVel.DT`. |
| `depth_m` | `0.01` | Maximum reported local source depth. Current integrated workflow requires at least `20*DX`. |
| `grid_spacing_px` | `8` | Cartesian fitting/report-grid spacing; positive integer. |
| `grid_phase_px` | `7` | First Cartesian grid coordinate, between zero and spacing minus one. |
| `surface` | See below | Geometry source and its indexing/offset. |
| `availability` | `"full"` | `"full"` or `"nonzero_boundary"`; combines with explicit sidecar masks. |
| `intensity_scale` | `1.0` | Multiplier applied to raw intensity. |
| `intensity_offset` | `0.0` | Offset added after multiplication. |
| `workers` | `1` | Local worker count; CLI `--workers` overrides base and per-pair values. Can change on resume. |
| `integration_interval_px` | `2.0` | Horizontal Simpson-interval width; the last interval may be shorter. |
| `depth_step_m` | `0.0001` | Depth-profile increment; the exact requested endpoint and 20-pixel depth are also included. |
| `piv_quality` | `"correlation"` | Native comparison policy: `"correlation"` or explicit `"finite"`; `run --piv-quality` overrides it. Can change on resume. |
| `pairs` | `{}` | Per-discovered-name overrides; nested `pairs` are not supported. |

Example override, using illustrative calibration values that must be replaced:

```json
{
  "dx_m_per_px": 0.00005,
  "dt_s": 0.01,
  "surface": {"mode": "sidecar", "index_base": 0, "offset_px": 0},
  "pairs": {
    "Run_80": {"dt_s": 0.005},
    "subdir__Run_100": {"depth_m": 0.015}
  }
}
```

For byte/16-bit inputs, intensity conversion is exactly `converted=raw*intensity_scale+intensity_offset`. Converted intensities must be finite and in `[0,255]`. Non-8-bit input requires an explicit nonidentity conversion; the package does not auto-stretch each frame independently. For a true full-range unsigned 16-bit acquisition, `255/65535` is one possible scale; 12-bit data stored in 16-bit containers need a different acquisition-appropriate conversion.

## Surface geometry

The `surface` object defaults to:

```json
{"mode": "auto", "index_base": 1, "offset_px": 0.0}
```

The local zero-based row used by the method is:

```text
surface_y = supplied_trace - index_base + offset_px
```

The trace has one finite row ordinate per image column. Positive offset moves it downward; negative offset moves it upward. Sidecar and detected nonzero-boundary traces are always zero-based regardless of `index_base`. The resulting surface must lie between row edges `-0.5` and `height-0.5`.

| `surface.mode` | Geometry source |
| --- | --- |
| `"auto"` | Use `NAME_surface.npz` if present, otherwise PIV MAT surface metadata; fail if neither exists. |
| `"piv_mat"` | Read `imSurfa.surfacePIVImg` and `imSurfb.surfacePIVImg` only. Apply explicit index base and offset. |
| `"sidecar"` | Require the NPZ described below; its coordinates are zero-based. |
| `"constant"` | Use `row_a`, with optional `row_b` defaulting to `row_a`; apply index base and offset. |
| `"nonzero_boundary"` | Use the first nonzero pixel in every image column as a zero-based trace, then apply offset. Only appropriate for a verified exported boundary convention. |

For a flat surface at zero-based row 80 in A and row 79 in B:

```json
{
  "surface": {
    "mode": "constant",
    "index_base": 0,
    "offset_px": 0,
    "row_a": 80,
    "row_b": 79
  }
}
```

Historical ExpLCL files used an exported trace interpreted with `index_base=0` and `offset_px=-12`. This explicit convention places the inferred geometric surface above the retained-image boundary. It is not applied to generic MATLAB traces. The separate `xPIV`/`zPIV` coordinate conversion for native PIV always follows their documented one-based MATLAB convention and is unaffected by `surface.index_base`.

### NPZ sidecar

`NAME_surface.npz` contains:

| Array | Shape and meaning |
| --- | --- |
| `surface_a` | `[width]`, zero-based geometric source-surface row. |
| `surface_b` | `[width]`, zero-based geometric target-surface row. |
| `availability_a` | Optional Boolean `[height,width]`, True where source image data are available. |
| `availability_b` | Optional Boolean `[height,width]`, True where target image data are available. |

Use ordinary numeric/Boolean NumPy arrays, not pickle objects. For example:

```python
import numpy as np

# Replace these illustrative dimensions and traces with measured geometry.
height, width = 512, 768
surface_a = np.full(width, 80.0)
surface_b = np.full(width, 79.0)
availability_a = np.ones((height, width), dtype=bool)
availability_b = np.ones((height, width), dtype=bool)
np.savez_compressed(
    "Run_80_surface.npz",
    surface_a=surface_a, surface_b=surface_b,
    availability_a=availability_a, availability_b=availability_b,
)
```

Geometry and availability are separate. `availability="full"` retains a supplied sidecar availability mask; it does not discard it. Without a sidecar mask it starts from full image availability, and the fitting method still applies its surface margins. `availability="nonzero_boundary"` additionally removes rows above the first nonzero pixel in each column. Every column must contain a nonzero pixel.

The nonzero rule recognizes a pre-existing zeroed boundary; it does not identify particle/reflection objects, infer physical water from texture, or remove dark holes inside the retained region. An intensity offset that changes exported zeros to nonzero values destroys this convention. Use explicit availability masks when the convention is uncertain. No automatic reflection/ML classifier is included.

## Native PIV MAT file

Both MATLAB v5 MAT files (including compressed structs) and v7.3 HDF5 files are supported. Selective reads preserve MATLAB logical dimensions: HDF5's reversed storage dimensions are reversed exactly once. The reader does not guess orientation from square shapes.

Required native comparison fields:

| Field | Convention |
| --- | --- |
| `compVel.xPIV` | One-dimensional increasing MATLAB pixel-center x coordinates; converted by subtracting one. |
| `compVel.zPIV` | Increasing MATLAB image-row pixel-center coordinates; converted by subtracting one. |
| `compVel.delta_x` | Native displacement in pixels, shape `[len(zPIV),len(xPIV)]`, positive rightward. |
| `compVel.delta_z` | Same shape, displacement in pixels positive upward; image-down displacement is its negative. |
| `compVel.DX` | Positive scalar metres/pixel. |
| `compVel.DT` | Positive scalar seconds between A and B. |

Optional fields:

| Field | Use |
| --- | --- |
| `compVel.dcor` | Same native grid shape. Required by default `correlation` comparison mode; only finiteness is tested. |
| `compVel.mask` | Same shape, `1=valid water`, `0=invalid air`; nonfinite entries are invalid. Other numeric values are rejected. |
| `compVel.IW`, `compVel.GS` | Recorded metadata; no undocumented interpretation is used to alter prediction. |
| `imSurfa.surfacePIVImg`, `imSurfb.surfacePIVImg` | Per-column surface metadata, usable for geometry before prediction. |

Prediction preparation reads only the requested calibration and surface fields. Native velocity arrays, correlations and the PIV validity mask are loaded later by the comparison module. Dense extrapolated fields such as `delta_x1`/`delta_z1` are never substituted for the native grid.

Default comparison validity combines finite displacement components, finite `dcor`, optional native valid-water mask, geometric source depth and actual source-image availability. It does not require target endpoint visibility, impose a score cutoff, or reject PIV because it disagrees with optical flow. The explicit `finite` mode removes only the finite-correlation requirement. Off-grid samples require every positive-weight native contributor to pass the chosen rule; exact grid nodes do not require zero-weight neighbors.

## Optional IR results

Pass a campaign or run MAT file through `--ir-results PATH`; raw IR imagery is not needed. The package assumes no fixed directory for this file. Campaign `exps` entries are selected by explicit `exp_name`; a run file can contain top-level sections. Ambiguous duplicate experiment names fail. An unlabelled top-level run file records its identity as unverified rather than guessing from its filename.

Selection uses:

| Field | Meaning |
| --- | --- |
| `PIV.pairNum` | Stored nonnegative pair labels, matched directly to the requested suffix/`--pair-number`. |
| `PIV.IR_idx` | One-based MATLAB index into the IR surface series; converted once to Python indexing. |
| `PIV.t`, `USurf.t` | Timing records used for diagnostics; `PIV.t` denotes A-frame time. |
| `USurf.usurf0` | Preferred finite raw fresh-dot surface velocity at the mapped index. |
| `USurf.usurffilt` | Finite smoothed estimate used only if indexed `usurf0` is missing. |
| `Surfs.pairNum`, `Surfs.t`, `Surfs.dt_pair`, `Surfs.spp` | Optional within-pair and cadence/timing diagnostics. |
| `USurf.IRfps`, `IRDX`, `Ndots_used`, `corrmax0`, `TMVTech` | Optional acquisition/quality context retained in selection records. |

Surface velocities are downstream-positive **m/s as stored**. There is no image-y sign flip or PIV `DX/DT` conversion. The indexed observation is not shifted to a guessed time or replaced by a neighboring raw sample. Nearby finite raw records may be listed as context but are not selected instead. Missing mappings or values produce an unavailable annotation with a reason. No theoretical/composite velocity is substituted.

`surface_ir_reference.json` persists the selected record separately from the image-only field. Subsequent comparisons without a new `--ir-results` argument reuse it. The renderer accepts only a finite available `USurf.usurf0` or `USurf.usurffilt` value, explicitly marked comparison-only and placed at zero depth. A top-level run file with no identifying name remains marked identity-unverified; it is not silently promoted to a verified experiment match.

## Numeric output use

`results.npz` can contain finite model values at locations that fail conservative screens. Always use `accepted` for velocity and the independent `gradient_accepted_xx`/`gradient_accepted_yy` flags for the two derivatives. `velocity_gradients.*` already applies these masks to the exported physical arrays.

The display grid in `plot_samples.npz` follows local surface depth; it is not the same Cartesian grid as `results.npz`. Comparison gradient maps retain this display geometry but use Cartesian derivatives. Integration samples are a third query set, saved explicitly with their own masks. Do not apply a mask from one query set to another by array position.

The comparison Python API is documented in `quadratic_optical.comparison.compare_pair`. It exports separate quality directories containing velocity, gradients, integrals and summary files. Output units are explicit in array names. For conversion, metres to centimetres multiply by 100; m²/s to cm²/s multiply by 10,000; s⁻¹ is unchanged.

Rendered profile rows separate changing accepted/shared support from fixed common horizontal support. An empty fixed intersection is displayed as “No supported domain” and keeps NaN integral/mean arrays. `piv_comparison.mat` gathers the currently selected comparison arrays for MATLAB; original image-only profiles remain in `horizontal_integral.mat`.
