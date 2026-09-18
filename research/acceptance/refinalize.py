"""Re-screen existing fits under a different acceptance profile.

The fits (main/affine/large_window/margin14/reverse) do not depend on the
acceptance rule, so a relaxed profile is a pure re-screen of identical fits.
Copying the pair directory and re-running only the post-fit stages gives a true
A/B -- every difference is the rule, not a refit -- and avoids hours of work.
"""
import json, shutil, sys, time
from pathlib import Path
sys.path.insert(0,'/home/surflab/GitRepos/QuadraticOptical/src')
from quadratic_optical.core import finalize_fields, integrate_profile
from quadratic_optical.reporting import sample_plot_grid, render_pair, write_json, surface_reference
from quadratic_optical.comparison import compare_pair
from quadratic_optical.cli import manual_for

SRC=Path(sys.argv[1]); DST=Path(sys.argv[2]); PROFILE=sys.argv[3]
PAIRS=[int(x) for x in sys.argv[4].split(',')]
CFG=json.loads(Path(sys.argv[5]).read_text())
PIVDIR=Path(sys.argv[6]) if len(sys.argv)>6 and sys.argv[6]!='-' else None
# Hand-matched picks are read only after the prediction is frozen, and only a
# few pairs have any; manual_for keys off the pair's own frozen manifest.
MANUAL=sys.argv[7] if len(sys.argv)>7 and sys.argv[7]!='-' else None
DST.mkdir(parents=True, exist_ok=True)
for n in PAIRS:
    name=SRC.name.replace('analysis_L2','') or None
    src=SRC/('ExpLCL_1_03_%d'%n); dst=DST/src.name
    t0=time.time()
    if dst.exists(): shutil.rmtree(dst)
    # Copy only what the post-fit stages need; skip prior outputs so nothing
    # stale can survive into the new screening.
    keep={'inputs.npz','input_manifest.json','ptv_tracks.npz','main.npz','affine.npz',
          'large_window.npz','margin14.npz','reverse.npz','tracking_settings.json',
          'tracking_summary.json','particle_candidates.npz','tracks_forward.npz',
          'tracks_reverse.npz','fit_summary.json','coarse_affine.npz','coarse_search.npz',
          'seeds.npz','bootstrap_sensitivity.json','coarse_bootstrap48.npz',
          'source_access_audit.json','surface_ir_reference.json'}
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.name in keep and f.is_file(): shutil.copy2(f,dst/f.name)
    _,summary=finalize_fields.run(dst, requested_depth_m=CFG['depth_m'], acceptance=PROFILE)
    sample_plot_grid(dst, CFG['depth_m'], PROFILE)
    integrate_profile.run(dst, requested_depth_m=CFG['depth_m'],
        interval_px=CFG['integration_interval_px'], depth_step_m=CFG['depth_step_m'],
        acceptance=PROFILE)
    sel=surface_reference(dst,None)
    comparison=None
    if PIVDIR is not None:
        piv=PIVDIR/('ExpLCL_1_03_%d_PIV.mat'%n)
        if piv.exists():
            try: comparison=compare_pair(dst,piv,quality=CFG['piv_quality'],surface_records=sel)
            except Exception as e: print('  PIV comparison skipped:',e)
    manual_record = manual_for(MANUAL, dst, PROFILE)
    render_pair(dst, comparison, sel, manual_record)
    write_json(dst/'status.json', dict(pair=src.name, status='complete', stage='complete',
        image_only=True, prediction_status='complete', prediction_summary=summary,
        acceptance_profile=PROFILE, PIV_used_in_prediction=False,
        PIV_comparison=comparison is not None, surface_record=sel,
        manual_comparison=None if manual_record is None else {
            'source': manual_record['manual']['path'], 'count': manual_record['count'],
            'accepted': manual_record['accepted'], 'statistics': manual_record['statistics']}))
    note='' if manual_record is None else '  manual %d picks / %d screened'%(
        manual_record['count'], manual_record['accepted'])
    print('  %s %-10s accepted %6d  (%.0f s)%s'%(src.name,PROFILE,summary['accepted_grid'],
                                                 time.time()-t0,note),flush=True)
