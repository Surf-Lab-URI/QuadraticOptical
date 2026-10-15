# How the estimate is made, step by step

A plain-language walkthrough of what happens between two particle images going in
and a velocity field coming out. It describes the shipped package in
`src/quadratic_optical/`, not the historical `hybrid_optical_flow_tutorial.pdf`,
which documents an earlier calculation that differs in places — most importantly
in how the first guess is obtained.

For the mathematics see the [method PDF](files/image_only_method.pdf) and
[method-notes.md](method-notes.md). For what the output files contain, see
[data-format.md](data-format.md). This document is about the *order of
operations* and why each step exists.

Numbers quoted throughout come from one real image pair, `ExpLCL_1_03` pair 121,
to give a sense of scale.

---

## The steps at a glance

| | What happens | What comes out |
|---|---|---|
| **1** | A rough survey: large patches of image A are slid around image B to find roughly where the water went | An approximate displacement map over the whole field |
| **2** | Particle-like bright spots are found in image A | A list of candidate particles (no matching yet) |
| **3** | Each particle is matched into image B, starting from the rough map | A displacement per particle, to a fraction of a pixel |
| **4** | Every match is repeated backwards and must close the round trip | The subset of particle tracks that are trustworthy |
| **5** | A small patch of image A is allowed to *deform* until it matches image B | Twelve numbers per grid point describing the local motion |
| **6** | The overlapping local descriptions are blended into one smooth field | Velocity and its derivatives, continuous everywhere |
| **7** | Everything is screened against the image evidence | The reported field, with gaps where evidence is lacking |

Steps 1–4 are the tracking half: getting a reliable but *scattered* set of
measurements. Steps 5–7 are the registration half: turning those into a
continuous field and deciding how much of it to stand behind.

---

## Step 1 — A rough map of where the water went

This happens **before** any particles are found.

A regular grid of points is laid over the underwater part of image A — one every
8 pixels, which came to 13,208 points for this pair. At each one:

**A square of image A is cut out**, 39 pixels on a side, centred on that point.

**It is slid around image B** and the match scored at every whole-pixel shift out
to 64 pixels in each direction — roughly 16,000 candidate shifts per point.

**The score ignores brightness and contrast.** Two patches match well if their
light-and-dark *pattern* lines up, even if one is overall brighter or flatter.
The illumination is neither uniform nor identical between frames, so a measure
that cared about absolute brightness would be measuring the lighting.

**Pixels that are not water in both images are excluded**, rather than treated as
zeros: air above the surface, masked regions, and anything off the edge of the
image. The search also insists on a decent overlap — at least 40 shared pixels,
or 65% of the square, whichever is larger — and discards any shift that falls
short.

**The three best shifts are kept, plus "no shift at all" as a baseline.** The
three are forced at least 3 pixels apart, so they are genuinely different
candidates rather than three samples of the same peak.

**Each candidate is then refined** with a fit that allows the square to stretch,
rotate and shear as well as slide. This is a *linear* transformation: straight
lines stay straight. Nothing quadratic happens here.

**The winner is chosen by match quality**, and must be well-behaved — not
squashed flat, and supported by enough real pixels. If no candidate qualifies the
grid point is marked unusable rather than given a dubious answer. Here, 12,220 of
the 13,208 points came through usable.

**As a self-check**, the whole search is run a second time limited to 48 pixels
instead of 64 and the two answers compared. Disagreement means the result
depended on how far the search was allowed to look. It is recorded but does not
change the answer.

The output is a rough displacement map covering the whole underwater field:
approximately right everywhere, not yet precise anywhere.

> **No external PIV is involved.** The first guess comes from this search, run on
> the images themselves. The historical tutorial seeded its coarse fits from
> supplied cross-correlation PIV; the shipped package does not, which is what
> allows the prediction to be independent of the PIV it is later compared
> against.

---

## Step 2 — Finding the particle-like spots

This works on **image A only**. Image B is not involved.

**Particle-sized detail is isolated first.** The image is blurred twice — once
very slightly, once more heavily — and the heavier blur subtracted from the
lighter one. What survives is detail at the scale *between* the two blurs. Broad
features such as the lighting gradient appear in both blurs and cancel;
single-pixel noise is removed by even the light blur. What is left is small
compact bright blobs, which is what a seeded particle looks like.

**Then peaks are located.** A pixel qualifies only if it is the brightest in the
5×5 box around it, so one particle cannot register as a dozen neighbouring
candidates.

**Then a series of checks.** The pixel must stand out brightly enough after
filtering; sit at least 4 pixels from the image edge, so a patch fits around it
later; be deep enough below the surface and within the analysed depth range; be a
valid water pixel; and **have a dim enough local background**.

This produced 19,297 candidate particles for this pair.

> **The background check is worth knowing about.** The local background is
> measured by blurring heavily, and a candidate is rejected if its surroundings
> are too bright — the reasoning being that a peak in a blazing region is more
> likely part of a reflection than a particle. Where laser glare near the surface
> saturates the image, however, *every* pixel in a band exceeds that ceiling and
> **no** candidates are produced there regardless of how good the seeding is.
> In `ExpLCL_4_05` this blinded everything above about 10–12 pixels depth: the run
> had 15% *more* particles than its neighbours overall but none near the surface,
> and its near-surface coverage collapsed for a reason unconnected to the water.
> See the near-surface section of `CLAUDE.md`.

At this point nothing has been matched. There is a rough map from step 1 and a
list of about 19,000 bright spots, produced independently of each other.

---

## Step 3 — Matching each particle into image B

**Each particle gets a starting guess from the rough map.** The twelve nearest
grid points that produced a usable answer in step 1 are found. Each of those says
"the water moved by *this much* at *my* location, and it was stretching and
shearing at *this rate*" — so each can extrapolate a prediction to where the
particle actually is.

Those twelve predictions will not all agree. The one sitting most centrally among
the others is identified, any prediction disagreeing with it by more than 4
pixels is discarded, and what remains is averaged with nearer grid points
weighted more heavily. One bad grid point in the neighbourhood is outvoted rather
than dragging the guess off.

**A 9×9 patch** — 81 pixels — is cut around the particle in image A. This is much
smaller than step 1's 39×39 square, deliberately: step 1 wanted a stable average
over a large area, this step wants to follow one specific particle.

**Every whole-pixel shift within 8 pixels of the guess is tried** — 289
possibilities — scored by the same brightness-independent similarity used in step
1. Only pixels valid in both images count, and at least 85% of the 81 must be
usable or the shift is rejected outright.

**Straying from the guess carries a mild penalty** that grows with distance. At 2
pixels it costs almost nothing; at the 8-pixel edge of the search it costs about
a third of the available score. A clearly better match further away still wins; a
marginal one does not get to drag the answer away from what the neighbourhood
suggests.

**The winner is then refined to sub-pixel precision** by an optimiser that
adjusts the shift continuously, sampling image B between pixels. Whole-pixel
answers are not good enough — the displacements of interest are often a fraction
of a pixel.

**Ambiguity is measured.** The best candidate at least 2.5 pixels away — a
genuinely different answer, not a neighbour of the same peak — is also refined,
and the margin between the two recorded. A particle where two quite different
shifts score almost equally well is one where the image is telling two stories.

> **This step only slides.** The 9×9 patch is not allowed to stretch, rotate,
> shear or bend. Over 9 pixels there is not enough information to fit anything
> more complicated reliably, and attempting it would make the answer less stable
> rather than more accurate. Deformation is estimated at coarse scale in step 1
> and properly in step 5, at scales where it can be.

---

## Step 4 — Checking the match by running it backwards

**Each particle is matched a second time, in reverse**: from where it landed in
image B back into image A, by exactly the procedure of step 3.

The backward pass builds its own starting guess from the rough map read in
reverse. It is not handed the forward answer, so it is a genuinely independent
attempt at the same question.

**If both passes are right they cancel.** Go forward by some amount, come back by
the same amount, and you land where you started. How far you actually land from
the starting point is the round-trip error.

This catches a specific and awkward failure. These images are full of
similar-looking specks. If the forward match latches onto the *wrong* particle
the match quality can still look excellent — it found a perfectly good particle,
just not the right one — but coming back from that wrong particle usually lands
somewhere else. The round trip exposes what the match score alone cannot.

**A track is kept only if all five hold:** forward match quality above threshold,
backward match quality above threshold, round trip closing within a pixel, and
neither the forward nor the backward answer being ambiguous.

For this pair, 17,024 of the 19,297 candidates survived — about 88%.

**The rejected ones are not deleted.** They remain in the output with their
measurements and a flag saying they did not pass, so what was discarded and why
stays inspectable. That pattern holds throughout: this tool marks things
unsupported rather than quietly dropping them.

What exists now is about 17,000 individual particles with displacements trusted
to a fraction of a pixel, scattered irregularly — dense where the seeding was
good, sparse where it was not, absent near the surface where the background check
blocked detection.

---

## Step 5 — Fitting a deforming patch of image A onto image B

The step the method is named after. It has two halves which use completely
different information.

### 5a — Turning the scattered particles into a smooth starting field

Back to the grid from step 1. At each grid point, the accepted particle tracks
nearby are gathered — up to 45 of them, within roughly 20 pixels, but always at
least the dozen closest — and a gently curved surface fitted through their
displacements. Not a flat plane: a curved one, so it can follow a shear layer
that is bending.

Two things make this robust rather than a simple average. Nearer tracks count
more than distant ones, and better-quality tracks more than marginal ones. And
the fit is repeated seven times, each pass down-weighting the tracks that
disagree most with the current answer, so a single badly-matched particle is
squeezed out instead of tugging the result.

**This half uses no image brightness at all** — only the displacements from step
4.

**It fits twelve numbers and passes on six.** The surface has six terms per
direction, the same form as 5b. But only the constant (the displacement itself)
and the two linear terms (the local rate of change) are kept. The three quadratic
terms per direction are computed and discarded.

They are not wasted. Their job is to absorb the curvature in the local cloud of
tracks so that it does not contaminate the lower-order terms. Fit a flat plane
through genuinely curved data and the plane is biased; let the fit bend, then
read off only the value and slope at the centre, and those two come out cleaner.
The quadratic terms protect the answer rather than being the answer.

### 5b — The actual quadratic fit

Now the images come back in.

At each grid point a **27×27 window** of image A — 729 pixels — is taken, and the
question asked: what deformation of this window makes it look most like the
corresponding part of image B?

**The deformation is described by twelve numbers.** Six control how the
horizontal displacement varies across the window, six the vertical. Within each
six: one constant, two describing how the displacement changes linearly across
the window, and three describing how that change itself changes.

That last group is the quadratic part, and it is the point of the method. A
linear map lets the window shift, stretch, rotate and shear, but straight lines
stay straight. The quadratic terms let the window **bend**. Near the surface,
where velocity changes rapidly with depth and that rate is itself changing, a
patch of water does not stretch uniformly — it curves. Twelve numbers can follow
that; six cannot.

**Two further numbers handle brightness** — a gain and an offset — because the
lighting is not identical between frames. Without them the fit would try to
explain a brightness change by moving the water.

**The fit starts from the 5a guess** and adjusts until the match stops improving.
Along the way: pixels near the window centre count more than those at the edge;
pixels that disagree badly are down-weighted, so one unmatched particle cannot
drag the fit; only pixels valid in **both** images count; the deformation is
prevented from folding the window inside out; and each adjustment is capped so it
cannot leap somewhere implausible.

**The twelve parameters start at zero except the first six**, which come from 5a.
The three quadratic terms per direction begin at exactly zero and are discovered
entirely by the image comparison. So the curvature in the final answer comes from
the images, not from the particle tracks. The tracks say roughly where the water
went and roughly how fast that is changing; the bending is established only by
comparing brightness.

### Five fits, not one

Four forward fits with deliberately different settings, plus one backward fit
from B to A:

| | window | deformation | distance kept from surface |
|---|---|---|---|
| primary | 27×27 | quadratic | 10 px |
| | 27×27 | **linear only** | 10 px |
| | **39×39** | quadratic | 10 px |
| | 27×27 | quadratic | **14 px** |

The purpose is to find out whether the answer depends on the choices. If a bigger
window, a simpler deformation, and staying further from the surface all give the
same displacement, that displacement is a property of the water. If they
disagree, it is a property of the settings.

> **What the comparison tests is displacement stability, not curvature.** The
> screening in step 7 measures how far each alternative's *displacement* lands
> from the primary fit's and requires the worst to be within tolerance. The
> curvature coefficients never enter that test, so a point where the fit found a
> lot of bending is not penalised — curvature is expected in a shear layer and
> costs nothing.
>
> The concern is different. Twelve free parameters work on a 27×27 window, and
> how much real information that window holds depends on how much texture is in
> it. Where texture is rich the parameters are well determined. Where it is thin
> they can trade off against each other, tilting and bending in ways that barely
> change the match quality while sliding the centre displacement somewhere it
> should not be. A simpler model has fewer knobs and cannot do that as easily. So
> disagreement is not evidence of curvature; it is evidence that the displacement
> was not pinned down well enough for the extra freedom to be safe.
>
> In practice this term rejects little. For this pair the variant landing
> furthest from the primary fit is the larger window about two-thirds of the
> time and the linear-only fit about a third, with median disagreements of a few
> tenths of a pixel at every depth. Curvature is judged more directly elsewhere:
> for `du/dx` to be reported at all, the variants must additionally agree about
> the *rate of change*, to a much tighter tolerance than the displacement. That
> is why the gradient field is always sparser than the velocity field.

---

## Step 6 — Blending the overlapping fits into one continuous field

Every grid point now owns its own description of how the water moved *in its own
neighbourhood* — twelve numbers each. These overlap heavily: grid points sit 8
pixels apart, each was fitted over a 27-pixel window, and the blending reaches 16
pixels. Any location falls inside a dozen or more, and they will not agree
exactly.

**To get the velocity anywhere, every nearby fit is asked.** Each grid point
within 16 pixels evaluates its polynomial *at the location being asked about*,
not at its own centre — so each neighbour gives its own opinion about this
particular spot, extrapolated a short distance from where it was fitted. The
opinions are averaged, weighted by distance, with the weight falling smoothly to
exactly zero at 16 pixels.

The weighting is chosen so that both the value **and its slope** are continuous
everywhere. A cruder weighting would leave faint seams at the boundaries between
grid points, and those seams would appear as spurious structure the moment a
derivative was taken.

**The derivatives are worked out exactly, not by subtracting neighbouring
values.** Both the polynomials and the weights are known formulas, so the slope
of the blended field can be written down directly. It has two parts: how each
polynomial is changing, plus how the *weights* are changing — which matters
because the neighbours disagree slightly, so shifting the balance between them
shifts the answer. Obtaining this analytically avoids the noise amplification
that comes with differencing gridded data.

**The blend also scores its own ingredients.** Each grid point's fit is marked
usable or not — did it stay well-conditioned, did it have enough real pixels. The
blend computes the weighted fraction of usable neighbours contributing at each
location, and step 7 requires that fraction to be high.

---

## Step 7 — Deciding what to keep

A velocity is reported at a location only if **all** of the following hold, and
each traces back to a particular earlier step:

**From the particle tracks (step 4)** — there must be an accepted track within 12
pixels, at least 6 accepted tracks within 25 pixels, and the location must fall
inside the region those tracks span. This is the test that says real particles
were actually observed nearby, and it is the one that does most of the rejecting
near the surface.

**From the quadratic fit (step 5b)** — the match quality must be good; the
forward and backward fits must agree to within a pixel; the four variants must
agree about the displacement; and the deformation must not have folded the window
inside out.

**From the blend (step 6)** — a high enough fraction of the contributing fits
must have been usable ones.

**From the geometry** — the location must be deep enough below the surface, the
place it maps *to* in image B must also be deep enough, and both must sit on
retained pixels rather than air or mask.

Anything failing any of these is recorded as missing. Not zero, not interpolated
from neighbours, not filled in from the supplied PIV — missing. That commitment
runs through the whole method: the output says where it does not know.

**Gradients face an extra bar** — a deeper minimum depth, a track required within
10 rather than 12 pixels, and agreement between the variants about the rate of
change. Hence `du/dx` and `dw/dz` are always sparser than the velocity.

The exact rule is one expression in `core/finalize_fields.py`; the thresholds are
tunable via the acceptance profiles described in `CLAUDE.md`.

---

## Points that are easy to get wrong

**The quadratic fit is a late step.** It happens after the rough survey, after
particle detection, after particle matching, and after the backward check. It is
fitted to a 27×27 window, not to the 9×9 patch used for matching.

**The 9×9 patches only ever slide.** No stretching, rotation, shear or bending is
fitted at that scale.

**The first guess is not external PIV.** It comes from the package's own coarse
search over the images, which is what keeps the prediction independent of the PIV
it is later compared against. This differs from the historical tutorial.

**Steps 1, 5a and 5b all fit something, and they are not the same thing.** Step 1
fits a linear map to a 39×39 square of image brightness. Step 5a fits a quadratic
surface to nearby particle displacements and keeps only its constant and linear
parts. Step 5b fits a quadratic map to a 27×27 square of image brightness and
keeps all of it.

**Curvature in the answer is expected and is not penalised.** The variant
comparison tests whether the *displacement* moves when the model is simplified,
which is a question about whether the fit is overfitting thin texture — not a
question about whether the flow is curved.
