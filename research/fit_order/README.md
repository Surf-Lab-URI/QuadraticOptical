# Refitting with a different polynomial order

`refit_order.py` re-runs the fitting stage of one pair with every fitted
polynomial forced to a chosen order, so the effect of the quadratic terms can be
measured directly. Use `--order 1` for a fully linear refit.

## What it changes, and what it keeps

The package fits polynomials in five places, and this script controls all of
them: the starting field built from particle tracks (package order 2), the four
forward fits (orders 2, 1, 2, 2) and the backward fit from image B to image A
(order 2). The rough survey in tracking is already linear, and particle matching
fits no polynomial at all, so the pair's prepared inputs and particle tracks are
copied unchanged. A comparison against the package therefore differs in
polynomial order and nothing else.

## Why it is a script and not a package option

The package hard-codes order 2 inside `core/fit_fields.py`, which is fingerprinted
as part of every fitting stage's provenance. Editing it would stop `run` from
resuming any existing output directory. So this script calls the package's own
solver (`fit_local`) and particle-surface model (`PTVModel`) directly, and
restates only the stage plumbing.

## Why it can be trusted

Restating plumbing risks a silent divergence, so `--verify N` first runs every
stage at the package's own orders on N sampled grid points and refuses to
continue unless the results are **bit-identical** to the pair's existing fits.
For `ExpLCL_1_03_123` they were, on all six stages; and a full-field refit at
order 2 reproduced every stage whose package order is 2 exactly, across all
13,210 grid points. A whole refit takes about 10 seconds with 24 workers.

## A quirk it deliberately reproduces

The backward fit's surface margin is a literal 10 px in the package, whereas the
forward fits follow the configured `surface_exclusion_px`. With the `L2` settings
the forward fits see pixels from 4 px below the surface but the backward fit is
blind above 10 px. This looks like an oversight from when the exclusion was made
configurable, and it may be costing near-surface coverage, since the backward
fit feeds the forward-backward agreement test and the backward usable-pixel
share. The script copies it on purpose, so that it does not confound an order
comparison.

## Result for ExpLCL_1_03 pair 123, aggressive screening

Removing every quadratic term changed the velocity field very little and the
gradient field a lot.

- Accepted velocities: 10,622 quadratic, 10,650 linear; 28.4% each at 5-9 px.
  Only 1 of 11,162 nodes was rejected solely by the simpler-model check the
  linear run loses, so the lost check does not explain the difference.
- Where both accept, velocities differ by a median of 0.02-0.14 cm/s, largest
  at 15-30 px (0.85-1.7 mm), where the profile bends most sharply.
- Against 189 hand-matched picks scored by both: median endpoint disagreement
  0.406 px quadratic, 0.426 px linear; quadratic closer on 56% of picks
  (p = 0.11, not distinguishable from chance).
- Best-available profile: at most 0.64 cm/s different, in the 1-3 mm band;
  under 0.12 cm/s elsewhere; 0.007 cm/s below 8 mm.
- du/dx: median change 0.43 /s against typical values near 1 /s.
