# Working notes for QuadraticOptical

Orientation for anyone (human or AI assistant) picking up this repository. Everything
below was verified by reading the source, not inferred from the documentation. Where the
documentation and the code disagree, that is recorded explicitly in the last section.

## What this code does, physically

Given two particle images of a free-surface flow separated by a short time `DT`, estimate
a two-component velocity field beneath the surface, plus the Cartesian velocity gradients
`du/dx` and `dw/dz`, and horizontal integrals at fixed depths below the *local* surface.

The estimator is deliberately conservative: it withholds a vector wherever the image
evidence does not support one, and records a `NaN` rather than a zero. Coverage is a
result, not a nuisance.

Two stages, which is what "hybrid" refers to:

1. **Particle-patch tracking** proposes a starting displacement field. Compact bright
   spots are found in frame A with a difference-of-Gaussians filter, and 9x9 patches are
   matched into frame B by normalized cross-correlation, with reverse and ambiguity checks.
2. **Deforming image registration** refines it. At each grid node a 12-coefficient
   quadratic displacement map is fitted to *all* valid pixels in a 27x27 window, allowing
   translation, rotation, shear and spatially varying deformation, plus a brightness
   gain/offset.

Overlapping local fits are then blended into one continuous field and differentiated
analytically. No neural network, no reflection classifier, no incompressibility or
free-surface kinematic constraint is imposed.

## Environment

Use the conda environment `quadoptical` (Python 3.12). It was created by asking conda for
Python only, so all five dependencies come from a single source and their compiled halves
match:

    conda activate quadoptical

Verified on 2026-09-14: 72 tests pass in ~18 s; the synthetic demo completes end to end in
~9 s. Resolved versions (numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, Pillow 12.3.0,
h5py 3.16.0) exactly match the author's own reference environment recorded in
`research/verification/release_checks.json`.

Do not install into conda `base` — it is Python 3.13.5, outside the repo's tested
3.10/3.12 CI matrix. Set `MPLBACKEND=Agg` when running headless or over SSH.

## Pipeline order

| # | Stage | File | Role |
|---|-------|------|------|
| 0 | Dispatch | `cli.py:246` | Pins BLAS thread counts before importing numpy; hence the lazy imports |
| 1 | Discovery | `discovery.py:20` | Filename-regex A/B pairing; sidecars matched by exact string concatenation |
| 2 | Config | `config.py:7` | defaults -> JSON -> merge -> validate, re-merged per pair |
| 3 | Ingestion | `prepare.py:31` | The only place pixels and calibration enter and are frozen |
| 4 | Tracking | `core/tracking.py:384` | Detect + match particles -> `ptv_tracks.npz` |
| 5 | Local fits | `core/fit_fields.py:368` | IRLS seed, 4 forward variants + 1 reverse fit |
| 6 | Blend + screen | `core/finalize_fields.py:293` | Wendland C2 blend with analytic derivative, then acceptance |
| 7 | Integration | `core/integrate_profile.py:149` | Q(h) below the local surface; gaps never bridged |
| 8 | Comparison | `comparison.py:496` | Post-freeze only; SHA-256s every artifact before and after |
| 9 | Reporting | `reporting.py:231` | Terminal stage; re-reads frozen NPZs, computes no science |

Fastest complete mental model: read `cli.py:50-145` (one screen, names every stage in
order), then the single acceptance expression at `core/finalize_fields.py:266-274`.
Between those two you can explain nearly any observed output.

## Velocity and gradient field panels

`reporting.field_panels()` draws five extra figures per pair from
`plot_samples.npz` (the depth-rectified display grid), called from `render_pair`
so they are produced by every `run` and `compare`:

| File | Contents |
|------|----------|
| `field_u` | horizontal velocity |
| `field_w` | vertical velocity, positive up |
| `field_u_smooth40px` | smoothed u, with 1 cm/s isotachs |
| `field_w_smooth40px` | smoothed w |
| `field_dudx_from_smooth40px` | du/dx by finite difference of the smoothed u |

Settings are module constants at the top of `reporting.py` (`FIELD_SMOOTH_PX`,
`FIELD_U_RANGE`, `FIELD_W_ABS`, `FIELD_DUDX_ABS`, `FIELD_CONTOUR_CM_S`). They are
deliberately **not** JSON configuration keys: every config key except `workers`
and `piv_quality` enters the preparation signature (`prepare.py:121`), so adding
one would invalidate every existing output directory for a change that alters no
number.

Each panel also has an absolute-height twin ending `_z`, drawn against height z
with the free surface plotted on it. Because every sample keeps its own image
row, that lattice is not rectangular in z and is drawn as a mesh rather than
resampled onto a regular grid. The depth-below-surface panels are unchanged.

`z = 0` is the still-water level: the mean of the first `FIELD_DATUM_FRAMES`
(20) frames of `Surfs.surfsPIV` in the campaign results file, which is the same
reference that file's `Surfs.eta` uses, so z here and eta there share a zero.
That read is cached per file. When no campaign file was supplied, or it cannot
be read, the datum falls back to that frame pair's own mean surface and the
panel says so on its face. `field_panels.json` records which was used and why.

For the ExpLCL data note that `surfsPIV` is the originally detected surface,
while the fitting trace is `surfacePIVImg + offset_px` = `surfsPIV + 10 - 12` =
`surfsPIV - 2`. So the plotted free-surface line sits about 2 px (0.113 mm)
above where eta would put it. That is the known consequence of choosing
`offset_px = -12` for comparability with the method document rather than -10.

All field panels are drawn at a true 1:1 aspect, with **both axes in
millimetres**, so feature and wave slopes are not distorted. The units have to
match for this: an earlier version used cm horizontally and mm vertically, where
an "equal" aspect would have locked 1 cm to 1 mm and made the distortion ten
times worse. The panels are consequently wide and short, and their colour bars
are attached to the drawn axes rather than the subplot slot, which an
aspect-locked axes no longer fills.

Colour limits are fixed rather than per-pair percentiles, so panels from
different pairs are directly comparable. Clipped samples are flagged magenta
(below) and green (above) instead of silently saturating; grey means no accepted
estimate. Each pair records limits, kernel and clipping fractions in
`field_panels.json`.

The `du/dx` panel is masked only by acceptance of `u`. It is **not** the screened
analytic gradient, which is stricter (depth >= 20 px, a track within 10 px,
cross-variant agreement <= 0.08) and remains in `velocity_gradients.csv`. Do not
quote numbers off the panel where the screened array says NaN.

## Raw-frame workflow (`extract`)

`quadratic-optical extract` builds a standard pair directory from the raw 12-bit
frames (`<run>/PIVRaw/PIV/<exp>_Piv_NNN_a|b.mat`, variable `imgPiv`) plus the
originally detected surface (`Surfs.surfsPIV` in the campaign results file). It
writes `NAME_imgA/B.tif`, a `NAME_surface.npz` sidecar, a ready `config.json` and
an `extract_manifest.json` recording every source hash and which campaign leaves
were read.

It is a separate step rather than an image reader inside `prepare` because the
raw frames sit beside a results file full of supplied velocities, and this
package's central promise is that prediction never reads those. An exported TIFF
is unambiguously an image, so the boundary stays checkable.

Facts established from the data, worth not rediscovering:

- The pre-masked TIFFs are `clip(raw, 0, 255)` exactly, so they discard both the
  surface band and about 1.8-2.2% of pixels at the 255 ceiling. Those clipped
  pixels are the particle cores, raw values to 4095.
- `surfacePIVImg = surfsPIV + 10`; `surfsPIV` is the originally detected surface.
  With the sidecar route the surface IS that detection, `offset_px` is 0, and the
  historical -10/-12 ambiguity does not arise.
- Surface row `2n` is frame a of pair n, `2n+1` is frame b, checked against the
  stored pair labels rather than assumed.
- The `ts` stamp in each raw frame is a camera clock, NOT the A-to-B delay: frames
  a and b differ by ~0.068 s, not 0.01. `compVel.DT` and `Surfs.t` both give 0.01 s.
- `compVel.DX` is metres per IMAGE pixel (2048 px = 11.572 cm), even though the
  campaign README calls it the velocity-grid cell size; the PIV grid is 4x coarser,
  so reading it that way gives velocities four times too large.

`--clip` sets the retained ceiling: 255 reproduces the pre-masked TIFF intensities
exactly, so a first run isolates the effect of the mask alone; higher values keep
more of the 12-bit range and write 16-bit TIFFs, with `intensity_scale` in the
emitted config set to match.

## Surface exclusion and the contrast floor

`surface_exclusion_px` (default 10) replaces the hard-coded near-surface mask.
The margin alternatives shift with it, so `margin14` stays four pixels deeper than
the primary whatever the base is. Note there are **three** separate near-surface
constants: this mask, the `depth >= 12` grid and acceptance floor
(`prepare.py`, `finalize_fields.py`), and the detector's 14. Lowering only the
exclusion changes which pixels feed each fit but does **not** extend reported
coverage nearer the surface; the 12 gates that and is still hard-coded.

`normalize()` takes a `floor`, and `prepare.contrast_floor(scale)` returns
`25 * scale**2`. The bare 25 is a variance in 0-255 units and only means a
five-count standard deviation when one converted unit is one 8-bit count. Scaling
it makes the operator invariant under a linear rescale of intensity. Without it a
12-bit frame mapped into 0-255 has its contrast divided by sixteen against a fixed
floor: measured median local contrast falls from 22.8 (4.6x the floor) to 3.7
(0.73x), the normalization goes flat, and nothing raises an error.

**`normalize` exists twice**, in `prepare.py` and `fit_fields.py`. They are
functionally identical but NOT textually identical. Change both or the margin
alternatives silently stop being the same model family.

## Re-running against an output directory built by older code

`prepare`'s plan includes `implementation_sha256`, the hash of `prepare.py`
itself, so **editing that file invalidates every existing output directory for
re-running**. That is deliberate: the code that built the inputs is part of what
the inputs are. No configuration shim avoids it.

Those directories stay valid and readable. Use `compare` to re-render one (it
never calls `prepare`, and it regenerates the report and all field panels), or a
fresh `--output` to recompute.

Note what a refused `run` costs: it overwrites that pair's `status.json` with
`failed` at stage `preparation`, clobbering `stage` and `prediction_status`, and
writes `error_traceback.txt`. The results themselves are untouched. `compare`
restores the record, keying off the frozen arrays it has just verified rather
than the status string, so a refusal is recoverable. A stale `error_traceback.txt`
is left behind and can be deleted by hand.

## Hand-matched particle comparison

`--manual-ptv <dir>` on `run` or `compare` adds a third held-out comparison beside
PIV and IR, read only after the prediction is frozen. Manual picks are a
reference, never seeds, constraints, training labels or corrections.

Files are matched to a pair by the `exp_name` and `image_pair_number` stored
**inside** each `.mat`, not by filename: the picking tool writes
`ExpLCL_1_03-123.mat` with a hyphen where image pairs use an underscore. The
pair's own identity comes from its frozen `input_manifest.json`, so this behaves
the same under `run` and `compare`. A pair with no picks renders exactly as
before; only a few pairs are ever hand-matched.

`p_orig` is `[N, 2, 2]`: particle, frame (1=A, 2=B), axis (1=x, 2=y), in MATLAB
one-based original-camera coordinates, converted to zero-based on read. The
package's `read_mat_fields` handles both v5 and v7.3 files, including the
dimension reversal in v7.3, and was checked to give `(N,2,2)` for all five.

The endpoint disagreement is the distance between predicted and hand-matched
arrow endpoints. Reproducing the tutorial PDF's table for pair 123 gives a
screened median of 0.377 px against its published 0.386 px over the same 200
picks, which confirms the coordinate convention: a one-pixel convention error
would move the median by about a pixel.

Each pair's manual section carries a static figure, the statistics table (all
picks, screened, and a shallow subset) and a small inline canvas with an arrow
magnification slider. The manual file's `surfa_orig` is bit-identical to
`Surfs.surfsPIV`, and its `altmask_offset` is 10, so it agrees with the raw-frame
extract route on where the surface is.

## Field viewer (`viewer`)

`quadratic-optical viewer <pair dir>` writes one self-contained HTML page: the
particle frames with vector layers drawn over them, an A/B flip, zoom and pan,
and sliders for arrow length and density. No external dependencies and nothing
fetched at open time, so it can be copied or emailed and still work.

Layers are whatever that pair has. The image-only field is always present;
`--piv` adds the supplied field, decimated by `--piv-stride` (default 4) before
embedding; `--manual-ptv` adds hand-matched picks where they exist. Controls list
only what is present, so the 58 pairs with no manual picks still get images plus
optical flow plus PIV, which is most of the value.

Frames are embedded as **lossless PNG**, deliberately. The whole purpose is to
flip A/B and judge by eye whether the arrows match the particles that moved, and
lossy compression smears exactly the specks being judged. Size is kept down by
cropping instead: a margin above the highest surface point down to the requested
depth below the lowest. For a 2048-wide frame that is about 430 rows rather than
2048, and a page with all three layers lands near 1.9 MB.

Verified in a browser rather than assumed: all three layers draw over the
particles, zoom resolves individual specks, the density and length sliders and
layer toggles work, the A/B flip changes the image, and the console is clean.
Note the browser pane cannot open `file://` URLs; serve the directory over
localhost to check a page.

## Traps

These cost hours if discovered by debugging rather than by being told.

**1. `SETTINGS` and `THRESHOLDS` are provenance records, not configuration.**
`core/tracking.py:37`, `core/fit_fields.py:34` and `core/finalize_fields.py:24` are only
ever consumed as `settings=SETTINGS` / `dict(THRESHOLDS, ...)` inside signature and
summary payloads. No algorithm reads them. Every real threshold is an independently
written literal — the entire track acceptance rule is one line at `tracking.py:438`, the
entire velocity acceptance rule is one expression at `finalize_fields.py:266-274`.
Editing a dict value retunes nothing; it only changes the run signature, which then
rejects the existing output directory.

**2. `finalize_fields.py` is NOT provenance-hashed.** `core/_provenance.py:24-25` hashes
only `tracking.py`, `fit_local_cached.py`, `_provenance.py`, `fit_fields.py` and
`ptv_model.py`. Changing a screening literal in `finalize_fields.py` is therefore silently
accepted into an existing output directory, *and* `summary.json` keeps reporting the
unedited `THRESHOLDS` dict — which now describes rules the code no longer applies. This is
the easiest way in this repo to produce a result whose own provenance record is wrong.
When changing a threshold, change the literal AND mirror it into the dict.

**3. `fit_local` is shared.** `core/fit_local_cached.py:11-80` is called both by the
fitting stage and by the coarse bootstrap at `tracking.py:190`. Check what a solver change
does to `ptv_tracks.npz` before concluding it improved the fields.

**4. `results.npz` `disp` and `gradient` are raw and unmasked.** Apply `accepted`,
`gradient_accepted_xx` and `gradient_accepted_yy` yourself; `reporting.field_export` does
this at `reporting.py:82-85`, and the file carries its own `screening_note` saying so.
There are three distinct query grids — `results.npz` (fitting), `plot_samples.npz`
(display), `integration_profile.npz` (Simpson lattice) — whose masks are not
interchangeable by array position.

**5. `.processing.lock` has no staleness detection.** It is a plain `O_EXCL` file
(`cli.py:15-28`). A Ctrl-C or kill leaves it behind and the next run refuses to start;
delete it by hand. Its presence also suppresses the post-failure `status.json` and
`error_traceback.txt` write (`cli.py:131`), so a stale lock can leave no diagnostics.

**6. Resume coverage is partial.** Only prepare, tracking and fit_fields consult existing
artifacts. `finalize_fields.run`, `integrate_profile.run` and all reporting recompute and
overwrite unconditionally on every invocation. Convenient when iterating on `depth_m`;
surprising if a no-op was expected.

**7. `grid_phase_px` defaults to 7 while `grid_spacing_px` defaults to 8**, and validation
requires `0 <= phase < spacing` (`config.py:79-82`). Lowering spacing for a denser grid
without also lowering phase fails with a message that does not mention the coupling.

**8. Discovery pairing is case-sensitive on the stem** even though the A/B suffix match is
case-insensitive, and sidecars are matched by exact concatenation — `foo_piv.mat` is never
found. Run `discover` before `run` on any new dataset.

**9. Only `workers` and `piv_quality` may legitimately change on a resume**
(`prepare.py:121-122`). Every other configuration key is part of the preparation
signature, and changing it requires a new `--output` directory. There is no `--force`.

## Documentation that disagrees with the code

- `README.md:117` says a batch containing failures returns a nonzero exit status. The exit
  code at `cli.py:143` counts only pairs attempted in the *current* invocation, so a
  successful rerun returns 0 with an older failure still listed. `README.md:123` is
  accurate. Judge a batch from `batch_summary.json`, not the exit code.
- `docs/data-format.md:145` claims an intensity offset destroys the nonzero-boundary
  convention. The code prevents this: `prepare.py:53-56` computes the boundary from the
  original pixels three lines before the conversion at `:59`, and
  `tests/test_prepare.py:109-114` asserts it survives.
- `docs/data-format.md:143` implies a sidecar availability mask is always honoured. It is
  read only when the resolved surface mode is `sidecar` (`prepare.py:66-77`); otherwise a
  present `NAME_surface.npz` is silently ignored — which is what
  `examples/legacy_expLCL.json` does.
- `README.md:182` "reuse completed compatible stages" applies to preparation, tracking and
  fitting only. See trap 6.
- `README.md:182` says source implementations participate in provenance checks;
  `reporting.py` is covered by nothing. See trap 2.
- `docs/method-notes.md:19` attributes contrast checks to patch matching. There is no
  contrast gate in the matching acceptance (`tracking.py:438`); the contrast-like screens
  belong to the detector.
- `docs/method-notes.md:105` says a short final integration interval occurs for odd image
  widths. It occurs whenever width is not an integer multiple of
  `integration_interval_px`, regardless of parity (`core/integrate_profile.py:16`).

## Relationship to the method documents

`docs/files/image_only_method.pdf` matches the shipped code and is authoritative alongside
`docs/method-notes.md`.

A separate tutorial (`hybrid_optical_flow_tutorial.pdf`, not in this repo) documents the
historical `ExpLCL_1_03_123` calculation implemented by the `full_width_*.py` scripts,
which now live under `research/archive/work/` and are explicitly **not supported entry
points**. The core mathematics is preserved in the portable package, but the initializer
differs: the tutorial seeds coarse affine fits from supplied PIV, whereas the shipped
package runs a symmetric masked NCC search over +/-64 px with no PIV seeding. The shipped
package also adds `dw/dz`, horizontal integrals, depth profiles, IR comparison and a batch
CLI. Read the tutorial for physical intuition; trust `docs/method-notes.md` for behaviour.

## Testing

    conda activate quadoptical
    pytest -q                     # full suite, ~18 s, 72 tests
    pytest -q tests/test_core.py  # ~4 s; the fast loop while editing the estimator

`tests/test_core.py` covers NCC, `fit_local` bit-exactness, quadratic ground-truth
recovery, blending, integration and the resume signatures. Note `tests/test_core.py:261`
pins displacement recovery to `atol=0.03` and `:267` asserts a NaN full-width integral on
purpose. `core/integrate_profile.py:220-246` (`self_test`) is the clearest executable
statement of the missing-data policy.

The package must be installed (`pip install -e .`) for pytest to collect at all; there is
no `conftest.py`. The `slow` marker registered in `pyproject.toml` is used by no test, so
`pytest -m "not slow"` deselects nothing.

## Conventions when editing

- Work on a branch, never directly on `main`.
- A scientific change requires a new `--output` directory. Do not delete or overwrite a
  completed prediction to make a rerun succeed.
- Preserve the conservative contract: missing estimates stay missing. Never substitute
  zero, an interpolated value, or a supplied PIV/IR velocity for an unsupported result.
- Keep numeric arrays together with their acceptance masks in any export.
