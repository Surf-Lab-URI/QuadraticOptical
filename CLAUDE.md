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

Colour limits are fixed rather than per-pair percentiles, so panels from
different pairs are directly comparable. Clipped samples are flagged magenta
(below) and green (above) instead of silently saturating; grey means no accepted
estimate. Each pair records limits, kernel and clipping fractions in
`field_panels.json`.

The `du/dx` panel is masked only by acceptance of `u`. It is **not** the screened
analytic gradient, which is stricter (depth >= 20 px, a track within 10 px,
cross-variant agreement <= 0.08) and remains in `velocity_gradients.csv`. Do not
quote numbers off the panel where the screened array says NaN.

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
