# Acceptance-profile trial scripts

`refinalize.py` re-screens existing fits under a different acceptance profile by
copying a pair directory and re-running only the post-fit stages, so a profile
comparison is a pure re-screen of identical fits rather than a refit.

**It deletes and recreates the destination directory**, keeping only the inputs,
tracks and fits it needs. Everything else is rebuilt from scratch -- including
`viewer.html`, which is NOT restored. Always rebuild the viewers after running
it, or the pages will simply be missing:

    quadratic-optical viewer <pair dir> --piv <pair PIV.mat> --manual-ptv <dir>

`build_index.py` discovers whatever has been screened and writes
`acceptance_profiles.html`; a pair with no `viewer.html` still appears in the
table but with no link, which is the symptom of a skipped rebuild.

`believe.py` checks newly admitted vectors for profile continuity, dispersion
against baseline, and agreement with the held-out IR surface velocity.
