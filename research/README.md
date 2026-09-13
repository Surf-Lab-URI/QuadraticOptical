# Research history

Use `quadratic-optical` and `src/quadratic_optical/` for all new analysis. The portable package consolidates preparation, tracking, fitting, conservative screening, integration, MAT/IR readers, comparison, and report generation.

`archive/work/` preserves the earlier processing scripts, diagnostics, plotting recipes, and experimental alternatives, including the initial cropped/large-frame analyses and later image-only recomputation. These are historical source material, **not supported entry points**. They expect intermediate files and the original research directory layout, which are not included. Workstation roots have been replaced with relative paths; the manifest records both original and archived hashes. Do not run these recipes over production outputs.

Earlier experiments may initialize from supplied PIV or use less conservative coverage. They must not be described as the current image-only method. The authoritative implementation and thresholds are the portable package plus the updated method PDF. Historical figures on the website use the final image-only predictions for pairs 80 and 100, with PIV applied afterward.

Large raw image/campaign files and generated fit checkpoints are intentionally external. The documented directory workflow regenerates them from user-provided images and geometry.
