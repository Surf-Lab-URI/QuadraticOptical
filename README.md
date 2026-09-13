# QuadraticOptical

[Local landing page](docs/index.html) · [Method PDF](docs/files/image_only_method.pdf) · [Research archive guide](research/README.md)

QuadraticOptical estimates a two-component velocity field from particle images near a free surface. It combines automatic particle tracking with robust local quadratic image registration, then reports conservative velocity coverage, Cartesian `du/dx` and `dw/dz`, and horizontal integrals at depths below the **local** surface. The default requested depth is 1 cm across the image width.

Each A/B pair is estimated independently. Supplied PIV and IR surface velocities enter only **after prediction**, as optional comparisons. Surface geometry, image availability and physical calibration can come from metadata in the same MAT file without reading its velocity arrays during fitting. This is a custom numerical method: it does not train or run GOFLOW, a U-Net, a reflection classifier, or another learned model. Missing estimates stay missing.

## Install locally

Use Python 3.10 or newer. Python 3.10 and 3.12 are the versions selected for the test matrix. MATLAB and a GPU are not required. Download the repository or clone it, then enter its root:

```sh
git clone https://github.com/Surf-Lab-URI/QuadraticOptical.git
cd QuadraticOptical
```

On macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m quadratic_optical --help
```

On Windows, from PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m quadratic_optical --help
```

The direct interpreter paths avoid requiring environment activation. In the commands below, replace `python` with `.venv/bin/python` on macOS/Linux or `.\.venv\Scripts\python.exe` on Windows. If the environment is activated, both `python -m quadratic_optical` and the installed `quadratic-optical` command work. Quote paths containing spaces. Initial dependency installation needs access to the Python package index; analysis itself runs locally.

## Try the synthetic example

```sh
python -m quadratic_optical demo --output demo_run
```

This creates a reproducible particle pair with a known translation, estimates it from its images, and produces a local report. Its supplied PIV and `truth.json` are validation data, not initialization inputs. To generate files first and inspect them before computation:

```sh
python -m quadratic_optical demo --output demo_run --generate-only
python -m quadratic_optical run demo_run/input --output demo_run/results --config demo_run/demo.json --workers 2
```

Use a new directory when generating a demo again. The printed `run` command resumes an existing demo. A successful synthetic test checks the numerical workflow; it does not establish accuracy on glare, reflections, out-of-plane motion or missing particles in an experiment.

## Organize an image series

Pair names, not file ordering, determine A/B correspondence:

```text
images/
  ExpLCL_1_03_80_imgA.tif
  ExpLCL_1_03_80_imgB.tif
  ExpLCL_1_03_80_PIV.mat       optional metadata/PIV comparison
  ExpLCL_1_03_80_surface.npz   optional explicit geometry and availability
  ExpLCL_1_03_100_imgA.tif
  ExpLCL_1_03_100_imgB.tif
  ExpLCL_1_03_100_PIV.mat
```

`NAME_a.tif` and `NAME_b.tif` are also recognized, as are `.tiff` and `.png` extensions. Images must be single-frame, two-dimensional grayscale arrays with equal shape. The default intensity convention is 8-bit `[0,255]`; other integer/float formats require an explicit conversion. There is no requirement that images be square or 2048 pixels wide. See [data-format.md](docs/data-format.md) for indexing, masks, MAT layout and sidecars.

List what will be processed:

```sh
python -m quadratic_optical discover images
python -m quadratic_optical discover images --recursive --skip-incomplete
```

Discovery reports incomplete pairs; it does not infer a missing frame. With recursive discovery, subdirectory names are joined to the pair stem using `__`, producing distinct output names. Use exactly these names with `--pair` and per-pair configuration overrides.

## Set calibration and surface conventions

The package needs `DX` in **metres per pixel** and `DT` in **seconds between A and B**. Set `dx_m_per_px` and `dt_s` explicitly, or confirm that `compVel.DX`/`compVel.DT` in the supplied metadata use those units. A positive numeric value alone cannot certify its physical unit. Do not substitute the time between successive A frames for the A-to-B delay.

The provided configurations serve different purposes:

| Configuration | Purpose |
| --- | --- |
| [metadata.json](examples/metadata.json) | Surface/calibration from a PIV MAT file, ordinary MATLAB one-based surface coordinates, no historical offset, finite-correlation PIV comparison. |
| [calibrated_sidecar.json](examples/calibrated_sidecar.json) | Explicit calibration and zero-based NPZ surface/availability sidecar. **Replace the illustrative calibration values.** |
| [legacy_expLCL.json](examples/legacy_expLCL.json) | Historical ExpLCL convention: zero-based exported trace, `-12` pixel offset, availability from the existing zeroed image boundary, and saved-finite PIV comparison. Applies only when these conventions are verified for the input files. |
| [sidecar_16bit.json](examples/sidecar_16bit.json) | Example explicit `0..65535` to `0..255` conversion with a sidecar. **Replace the illustrative calibration and intensity mapping if the acquisition differs.** |

The historical `-12` offset is **not a generic correction** and is not enabled by default. A retained-image boundary can differ from the actual geometric free surface; do not silently equate them. Supplying geometry does not recover image information removed above a mask.

All JSON options, defaults and pair overrides are described in [data-format.md](docs/data-format.md). The current integral workflow requires the requested depth to reach at least `20*DX` so that its declared common-depth band exists.

## Process one pair or a directory

```sh
python -m quadratic_optical run images --output analysis --config examples/metadata.json --workers 4
```

For the verified historical ExpLCL format, select its explicit configuration:

```sh
python -m quadratic_optical run images --output analysis_legacy --config examples/legacy_expLCL.json --workers 4 --pair ExpLCL_1_03_80 --pair ExpLCL_1_03_100
```

The workflow processes pairs in sequence and uses `--workers` within a pair. A large image and a dense grid can take substantial CPU time. Changing the plotting density does not remove the cost of fitting the underlying field.

Useful batch options:

| Option | Behavior |
| --- | --- |
| `--recursive` | Discover pairs in nested directories. |
| `--pair NAME` | Select one discovered name; repeat to select several. |
| `--skip-incomplete` | Skip stems missing A or B; report them in the batch record. |
| `--continue-on-error` | Record a failed pair and continue to later pairs. A batch containing failures still returns a nonzero exit status. |
| `--no-piv-comparison` | Complete image-only estimation without comparing native PIV velocities. Geometry/calibration metadata may still be needed. |
| `--piv-quality correlation\|finite` | Override the comparison policy for this run, including per-pair configuration. It does not change prediction. |
| `--ir-results PATH` | Add optional surface-velocity comparisons from a campaign/run MAT file. |
| `--experiment NAME` | Explicit experiment identity for IR selection; otherwise inferred from a suitable pair stem. |

The report is `analysis/index.html`; open it in a browser. No web server, upload or hosted account is needed. `status.json` records the current/completed stage for each pair. If an optional PIV comparison fails after prediction, the finished image-only arrays are retained and an image-only report is produced when possible. The batch status still records the comparison failure. `--continue-on-error` returns a nonzero status if any selected pair fails during the current invocation; entries for other previously processed pairs remain in the batch index.

## Compare an already completed prediction

```sh
python -m quadratic_optical compare analysis/ExpLCL_1_03_80 --piv images/ExpLCL_1_03_80_PIV.mat --quality correlation
```

This reads the frozen prediction and its saved integration samples without refitting. Default `correlation` quality requires finite supplied `dcor`, finite vector components, an optional valid-water PIV mask, and actual source-image availability. It does not impose an arbitrary correlation-score cutoff. If `dcor` is absent, the command fails with an actionable message.

To explicitly compare the producer's saved finite velocities even where correlation is absent or NaN:

```sh
python -m quadratic_optical compare analysis/ExpLCL_1_03_80 --piv images/ExpLCL_1_03_80_PIV.mat --quality finite
```

For an existing batch configuration, the equivalent comparison-only override is `run ... --piv-quality finite`. The `compare` command also accepts `--piv-quality` as an alias of `--quality`.

The two modes write separate `comparison/correlation` and `comparison/finite` directories. Historical comparison figures used finite saved vectors; new default figures may have less PIV coverage because they require finite correlation. Neither policy changes the image-only field. The producer may have interpolated or otherwise postprocessed its native PIV values; finite saved values are not automatically raw independent measurements.

## Optional IR surface point

```sh
python -m quadratic_optical run images --output analysis --config examples/metadata.json --ir-results path/to/results.mat --experiment ExpLCL_1_03
python -m quadratic_optical compare analysis/ExpLCL_1_03_80 --piv images/ExpLCL_1_03_80_PIV.mat --ir-results path/to/results.mat --experiment ExpLCL_1_03 --pair-number 80
```

`results.mat` is optional and its location is supplied by the caller. **No raw IR directory is assumed, searched or required.** The reader selects the named experiment and matches the stored pair label to `PIV.IR_idx`. It prefers a finite indexed `USurf.usurf0`; if that is missing, it can display finite indexed `USurf.usurffilt`, explicitly labelled a smoothed estimate. It records timing and missing-data details and never substitutes a theoretical velocity or neighboring raw time. See [data-format.md](docs/data-format.md#optional-ir-results).

The exact selection is stored in `surface_ir_reference.json`. Later report/comparison commands retain this record when `--ir-results` is omitted; an explicitly selected new record replaces it. Only finite, indexed raw/filtered observational fields marked comparison-only are accepted for a star at zero depth. An unavailable selection remains a recorded reason, without a fabricated star.

The surface star is a velocity comparison at depth zero. It does not extend optical-flow coverage to the surface, constrain the fitted field, or provide a horizontal integral at the surface. When repeated in the fixed-domain row, it is the same external reference, not another measurement.

## Read the outputs

Each pair directory contains:

| Output | Contents |
| --- | --- |
| `index.html`, `quiver.png/.svg`, `profiles.png/.svg` | Local report, velocity vectors, and two profile rows: changing accepted/shared intervals and one fixed common domain. |
| `gradients.png/.svg` or `gradient_comparison.png/.svg` | Separate Cartesian `du/dx` and `dw/dz`, with gray unavailable cells. |
| `velocity_gradients.csv/.mat/.npz` | Directly masked physical velocity/gradient arrays and acceptance flags. |
| `results.npz`, `plot_samples.npz` | Full diagnostics and the separate reporting/display samples. Finite raw values still require their acceptance masks. |
| `integration_profile.npz`, `horizontal_integral.mat` | Saved horizontal sample paths, interval masks, accepted-width and fixed-domain integrals, means and model sensitivity. Some legacy array keys contain `assumed` to preserve compatibility; calibration responsibility is unchanged. |
| `summary.json`, `input_manifest.json`, `integration_metadata.json` | Counts, settings, data conventions, signatures and provenance. |
| `comparison/<quality>/*.npz`, `*.csv`, `summary.json` | Optional held-out velocity, gradient and joint-integral comparisons. |
| `piv_comparison.mat` | Optional MATLAB export of compared velocity, gradient and integral arrays for the currently rendered comparison. |
| `comparison_notes.json` | Selected comparison settings and optional IR selection/timing record. |
| `surface_ir_reference.json` | Optional persistent IR field selection, mapping, timing and missing-data record. |
| `quiver_display_metadata.json` | Displayed arrow source indices, scales, tips and axes limits for visual audit. |
| `status.json`, `error_traceback.txt` | Per-pair processing stage/status and diagnostic traceback when a failure occurred. |
| Tracking, seed and local-model NPZ/checkpoint files | Resumable intermediate work; retain these for audit or later comparisons. |

The batch root also contains `batch_summary.json` and `index.html`. Keep numeric arrays and masks together. The quiver is thinned for readability, so a missing plotted arrow is not by itself an indication that the corresponding numeric location was rejected.

The upper profile row integrates over the **accepted horizontal segments at that depth**. Shared OF/PIV curves use exactly the same intervals, but this width can change with depth. Compare their coverage and width-averaged velocities alongside the integral. The lower row uses a fixed common horizontal domain when one exists. If it is empty, the report says “No supported domain”; unavailable velocity/integral values remain NaN, not zero. [Method notes](docs/method-notes.md) explain the operators and limits.

## Resume and provenance

Rerun the same command to reuse completed compatible stages and continue checkpoints. Inputs, relevant settings and source implementations participate in provenance checks. A mismatch is rejected; use a new output directory for a different image, calibration, surface convention, intensity mapping or numerical method. Completed predictions are not silently repurposed.

Worker count and PIV comparison quality are execution/comparison settings and may be changed when resuming. They are excluded from the scientific preparation signature. For example, increasing `--workers` or using `--piv-quality finite` reuses compatible numerical stages. The initial manifest retains the original full configuration; current run/comparison records identify the selected comparison quality. Changes to scientific settings still require a new output directory.

One process owns each pair directory through `.processing.lock`. If a process was forcibly terminated, first verify it has stopped; only then remove its stale lock and rerun. Do not run two workers as separate commands against the same pair output directory. Increasing `--workers` is handled by one orchestrating process.

## Method document, archive and project website

The [method PDF](docs/files/image_only_method.pdf) and its [LaTeX source](docs/files/image_only_method.tex) describe the numerical method and the historical comparison conventions. The [research archive guide](research/README.md) describes 174 earlier Python scripts under `research/archive/work/`. Those scripts preserve exploratory work, diagnostics and earlier processing recipes; they are **not supported entry points** and expect research intermediates that are not distributed. Some earlier experiments used supplied PIV or different coverage rules. Use the installed `quadratic-optical` package for new analysis.

The project landing page is static HTML/CSS/JavaScript under `docs/`; it requires no Node installation or build. Its intended published address is [surf-lab-uri.github.io/QuadraticOptical](https://surf-lab-uri.github.io/QuadraticOptical/). To enable GitHub Pages after the `main` branch has been pushed:

1. Open the repository's **Settings → Pages**.
2. Under **Build and deployment**, select **Deploy from a branch**.
3. Select branch **main**, folder **/docs**, and save.
4. Check the Pages deployment status before treating the public URL as live.

Later updates to `docs/` on `main` update that site. The Python test workflow is separate from Pages publishing. See the [official GitHub Pages instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) for repository settings. Local pair reports do not need Pages and are not automatically uploaded.

## Tests and interpretation

```sh
python -m pytest -q
```

Tests cover signed image-only initialization, masked image registration, known quadratic deformations, analytic derivatives, conservative missing-data behavior, checkpoint identities, MATLAB v5/v7.3 orientation, IR selection, and matched PIV/OF integrals. The synthetic suite needs no experiment files. GitHub Actions is configured for Windows, macOS and Linux on Python 3.10 and 3.12; a configured matrix is not a claim that every remote job has already run.

On the existing frozen experimental pairs 80/100, the portable `finite` comparison was checked against the earlier comparison implementation: gradient values/masks and joint-integral values/masks matched exactly, including NaNs. This verifies the comparison port, not experimental ground-truth accuracy or reproduction of every historical fitting run.

Near-surface glare, missing particles and out-of-plane motion can remain unsupported. The method imposes neither no-slip at the free surface nor 2-D incompressibility. Fine texture in a gradient map is not evidence of resolved small-scale physical strain. Sensitivity across the fitted variants is not a confidence interval. See [method-notes.md](docs/method-notes.md) before interpreting the closest accepted vectors or spatial derivatives.
