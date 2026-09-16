"""Run independent image pairs, then optionally compare frozen fields with PIV/IR."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .config import load_config, for_pair
from .discovery import discover


@contextmanager
def output_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / '.processing.lock'
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError('Output is locked: '+str(path)+'. If the previous process has stopped, remove this lock file and rerun.')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump({'pid': os.getpid(), 'created_unix': time.time()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def surface_annotations(pairs, results_path, experiment=None):
    if results_path is None:
        return {}
    from .ir import select_surface_values
    grouped = {}
    for pair in pairs:
        exp = experiment or pair.experiment
        if exp is None or pair.pair_number is None:
            raise ValueError('IR matching requires a NAME ending in _<zero-based pair number> and an experiment name (or --experiment).')
        grouped.setdefault(exp, []).append(pair)
    records = {}
    for exp, members in grouped.items():
        rows = select_surface_values(results_path, exp, [p.pair_number for p in members])
        if len(rows)!=len(members) or any(row.get('pair_number_zero_based')!=pair.pair_number for pair,row in zip(members,rows)):
            raise ValueError('IR selection returned a different pair mapping from the requested image pairs.')
        records.update({p.name: row for p, row in zip(members, rows)})
    return records


def manual_for(folder, directory):
    """Held-out hand-matched comparison for this pair, or None when absent.

    The pair's identity comes from its own frozen manifest rather than from
    command-line arguments, so this works the same under run and compare. Read
    only after the prediction is frozen; a missing file is not an error, since
    only a few pairs are ever hand-matched.
    """
    if not folder:
        return None
    manifest_path = Path(directory)/'input_manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    pair_number = manifest.get('pair_number')
    if pair_number is None:
        return None
    from .manual import find_manual, compare_manual
    record = find_manual(folder, manifest.get('experiment'), int(pair_number))
    if record is None:
        return None
    return compare_manual(directory, record)


def run_batch(args):
    from .prepare import prepare
    from .core import tracking, fit_fields, finalize_fields, integrate_profile
    from .reporting import sample_plot_grid, render_pair, render_batch, write_json, surface_reference
    from .comparison import compare_pair
    config = load_config(args.config)
    if args.workers is not None:
        if args.workers < 1:
            raise ValueError('--workers must be positive.')
        config['workers'] = args.workers
    pairs, incomplete = discover(args.input, args.recursive, args.skip_incomplete)
    if args.pair:
        missing = set(args.pair)-{p.name for p in pairs}
        if missing:
            raise ValueError('Requested pairs were not found: '+', '.join(sorted(missing)))
        pairs = [p for p in pairs if p.name in args.pair]
    annotations = surface_annotations(pairs, args.ir_results, args.experiment)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output/'batch_summary.json'
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    saved_rows = {row['pair']: row for row in previous.get('pairs', [])}
    rows = []
    for pair in pairs:
        start = time.monotonic()
        row = {'pair': pair.name, 'status': 'running', 'stage': 'preparation',
               'prediction_status': 'pending', 'PIV_used_in_prediction': False}
        print('\nProcessing '+pair.name, flush=True)
        try:
            directory = output / pair.name
            cfg = for_pair(config, pair.name)
            if args.workers is not None:
                cfg['workers'] = args.workers
            if args.piv_quality is not None:
                cfg['piv_quality'] = args.piv_quality
            with output_lock(directory):
                def stage(name):
                    row['stage'] = name
                    write_json(directory/'status.json', row)
                stage('preparation')
                manifest = prepare(pair, directory, cfg)
                print('Image-only preparation complete; fitting '+str(manifest['grid_nodes'])+' local nodes.', flush=True)
                stage('tracking')
                if not tracking.run(directory, workers=cfg['workers']):
                    raise RuntimeError('Tracking stopped before completion; resume this directory.')
                stage('fitting')
                fit_fields.run(directory, workers=cfg['workers'])
                stage('finalization')
                _, summary = finalize_fields.run(directory, requested_depth_m=cfg['depth_m'])
                sample_plot_grid(directory, cfg['depth_m'])
                stage('integration')
                integrate_profile.run(directory, requested_depth_m=cfg['depth_m'],
                    interval_px=cfg['integration_interval_px'], depth_step_m=cfg['depth_step_m'])
                row.update(prediction_status='complete', prediction_summary=summary, image_only=True)
                selected_surface=surface_reference(directory,annotations.get(pair.name))
                comparison = None
                if pair.piv_mat is not None and not args.no_piv_comparison:
                    stage('comparison')
                    print('Prediction frozen. Reading supplied PIV for comparison.', flush=True)
                    comparison = compare_pair(directory, pair.piv_mat, quality=cfg['piv_quality'],
                                              surface_records=selected_surface)
                manual_record = manual_for(args.manual_ptv, directory)
                if manual_record is not None:
                    print('Hand-matched comparison: %d picks, %d passing the screen.'
                          % (manual_record['count'], manual_record['accepted']), flush=True)
                stage('reporting')
                render_pair(directory, comparison, selected_surface, manual_record)
                row.update(status='complete', image_only=True,
                    manual_comparison=None if manual_record is None else {
                        'source': manual_record['manual']['path'], 'count': manual_record['count'],
                        'accepted': manual_record['accepted'], 'statistics': manual_record['statistics']},
                    PIV_used_in_prediction=False, PIV_comparison=comparison is not None,
                    piv_quality=cfg['piv_quality'] if comparison else None,
                    surface_record=selected_surface, stage='complete')
                write_json(directory/'status.json', row)
        except Exception as error:
            row['status'] = 'failed'
            row['error'] = str(error)
            row['error_type'] = type(error).__name__
            # Preserve a reviewable image-only report after a failed optional
            # comparison; the numerical prediction has already completed.
            if row['prediction_status'] == 'complete' and row['stage'] == 'comparison':
                try:
                    with output_lock(directory):
                        render_pair(directory, None, annotations.get(pair.name))
                    row['prediction_report_available'] = True
                except Exception as report_error:
                    row['report_error'] = str(report_error)
            if directory.exists() and not (directory/'.processing.lock').exists():
                write_json(directory/'status.json', row)
                (directory/'error_traceback.txt').write_text(traceback.format_exc(), encoding='utf8')
            print('Failed '+pair.name+': '+str(error), file=sys.stderr, flush=True)
        row['elapsed_seconds'] = round(time.monotonic()-start, 3)
        rows.append(row)
        saved_rows[pair.name] = row
        all_rows = list(saved_rows.values())
        write_json(summary_path, {'pairs': all_rows, 'skipped_incomplete': incomplete})
        render_batch(output, all_rows)
        if row['status'] == 'failed' and not args.continue_on_error:
            break
    failed = sum(r['status'] != 'complete' for r in rows)
    print('\nReport: '+str(output/'index.html'), flush=True)
    return 1 if failed else 0


def compare_existing(args):
    from .comparison import compare_pair
    from .reporting import render_pair, surface_reference, write_json, render_batch
    directory = Path(args.directory).expanduser().resolve()
    surface_record = None
    if args.ir_results:
        from .ir import select_surface_values
        path = directory/'input_manifest.json'
        metadata = json.loads(path.read_text()) if path.exists() else {}
        experiment = args.experiment or metadata.get('experiment')
        pair_number = args.pair_number if args.pair_number is not None else metadata.get('pair_number')
        if experiment is None or pair_number is None:
            raise ValueError('Provide --experiment and --pair-number for IR matching.')
        surface_record = select_surface_values(args.ir_results, experiment, [pair_number])[0]
    with output_lock(directory):
        surface_record=surface_reference(directory,surface_record)
        comparison = compare_pair(directory, args.piv, quality=args.quality, surface_records=surface_record)
        manual_record = manual_for(getattr(args, 'manual_ptv', None), directory)
        render_pair(directory, comparison, surface_record, manual_record)
        write_json(directory/'comparison_status.json',dict(status='complete',quality=args.quality,
            PIV_used_in_prediction=False,refitting_performed=False,surface_record=surface_record))
        status_path=directory/'status.json'
        summary_path=directory/'summary.json'
        if status_path.exists():
            status=json.loads(status_path.read_text())
            # compare_pair has just required every frozen prediction array and
            # checked their hashes before and after, so the prediction here is
            # complete whatever the status record happens to say. Trusting that
            # record instead would leave a pair permanently marked failed after a
            # later run was refused against an older output directory: the refusal
            # overwrites stage and prediction_status, and nothing could then put
            # them back. Evidence from the artifacts is the sounder test.
            if summary_path.exists():
                status.update(status='complete',stage='complete',prediction_status='complete',
                              PIV_used_in_prediction=False,image_only=True,
                              prediction_summary=json.loads(summary_path.read_text()),
                              PIV_comparison=True,piv_quality=args.quality,
                              surface_record=surface_record)
                for key in ['error','error_type','report_error']:status.pop(key,None)
                write_json(status_path,status)
                batch_path=directory.parent/'batch_summary.json'
                if batch_path.exists():
                    batch=json.loads(batch_path.read_text())
                    for index,row in enumerate(batch.get('pairs',[])):
                        if row['pair']==directory.name:batch['pairs'][index]=status
                    write_json(batch_path,batch);render_batch(directory.parent,batch.get('pairs',[]))
    print('Comparison report: '+str(directory/'index.html'))
    return 0


def make_demo(directory):
    """Generate reproducible integer-translated particles and a held-out PIV file."""
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from scipy.io import savemat
    directory = Path(directory).expanduser().resolve()
    inputs = directory/'input'
    inputs.mkdir(parents=True, exist_ok=True)
    names = ['Demo_0_imgA.tif', 'Demo_0_imgB.tif', 'Demo_0_PIV.mat']
    if any((inputs/name).exists() for name in names) or (directory/'demo.json').exists():
        raise ValueError('Demo files already exist. Resume with the printed run command, or choose a new demo directory.')
    rng = np.random.default_rng(1907)
    shape = (160, 192)
    spikes = np.zeros(shape)
    count = int(np.prod(shape)*.04)
    np.add.at(spikes, (rng.integers(0, shape[0], count), rng.integers(0, shape[1], count)), rng.uniform(160, 420, count))
    a = np.clip(np.rint(gaussian_filter(spikes, .9)+5), 0, 255).astype(np.uint8)
    dx, dy = 6, -3
    b = np.full_like(a, 5)
    b[:dy, dx:] = a[-dy:, :-dx]
    Image.fromarray(a).save(inputs/names[0]);Image.fromarray(b).save(inputs/names[1])
    x, y = np.arange(4, shape[1]-3, 4), np.arange(4, shape[0]-3, 4)
    grid_shape = (len(y), len(x))
    # Native MATLAB logical arrays are [y,x]; delta_z is positive upward.
    savemat(str(inputs/names[2]), {'compVel': {'DX': .0001, 'DT': .01,
        'xPIV': x+1, 'zPIV': y+1, 'delta_x': np.full(grid_shape, dx),
        'delta_z': np.full(grid_shape, -dy), 'dcor': np.ones(grid_shape), 'IW': 32, 'GS': 4},
        'imSurfa': {'surfacePIVImg': np.full(shape[1], 17)},
        'imSurfb': {'surfacePIVImg': np.full(shape[1], 14)}})
    config = {'depth_m': .006, 'grid_spacing_px': 16, 'grid_phase_px': 7,
        'surface': {'mode': 'piv_mat', 'index_base': 1, 'offset_px': 0}, 'depth_step_m': .0005}
    (directory/'demo.json').write_text(json.dumps(config, indent=2)+'\n')
    (directory/'truth.json').write_text(json.dumps({'dx_px': dx, 'dy_down_px': dy, 'u_m_per_s': .06, 'w_up_m_per_s': .03,
        'du_dx_per_s': 0, 'dw_dz_per_s': 0, 'note': 'Synthetic validation only; never loaded by the estimator.'}, indent=2)+'\n')
    return directory, inputs


def parser():
    p = argparse.ArgumentParser(description='Conservative quadratic optical flow. Supplied PIV and IR velocities are comparisons only.')
    p.add_argument('--version', action='version', version='quadratic-optical 0.1.0')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('discover', help='List matched image pairs without processing.')
    d.add_argument('input');d.add_argument('--recursive', action='store_true');d.add_argument('--skip-incomplete', action='store_true')
    r = sub.add_parser('run', help='Estimate every discovered pair and write numerical/visual reports.')
    r.add_argument('input');r.add_argument('--output', required=True);r.add_argument('--config')
    r.add_argument('--workers', type=int);r.add_argument('--recursive', action='store_true')
    r.add_argument('--skip-incomplete', action='store_true');r.add_argument('--continue-on-error', action='store_true')
    r.add_argument('--pair', action='append');r.add_argument('--no-piv-comparison', action='store_true')
    r.add_argument('--piv-quality', choices=['correlation', 'finite'],
                   help='Comparison only: require finite dcor (default), or retain supplied finite vectors without dcor.')
    r.add_argument('--ir-results');r.add_argument('--experiment')
    r.add_argument('--manual-ptv', help='directory of hand-matched particle .mat files; matched to a '
                   'pair by the exp_name and image_pair_number stored inside them, not by filename')
    c = sub.add_parser('compare', help='Compare an existing frozen prediction without refitting.')
    c.add_argument('directory');c.add_argument('--piv', required=True);c.add_argument('--quality', '--piv-quality', choices=['correlation','finite'], default='correlation')
    c.add_argument('--ir-results');c.add_argument('--experiment');c.add_argument('--pair-number', type=int)
    c.add_argument('--manual-ptv', help='directory of hand-matched particle .mat files')
    e = sub.add_parser('extract', help='Export raw PIV frames and campaign surfaces into a readable pair directory.')
    e.add_argument('--run-dir', required=True, help='experiment run directory containing PIVRaw/PIV')
    e.add_argument('--results', required=True, help='campaign results file holding Surfs.surfsPIV')
    e.add_argument('--output', required=True)
    e.add_argument('--experiment', help='defaults to the single experiment named by the raw frames')
    e.add_argument('--pair', action='append', type=int, help='pair number; repeat to select several')
    e.add_argument('--first', type=int);e.add_argument('--last', type=int)
    e.add_argument('--clip', type=float, default=255.,
                   help='retained intensity ceiling; 255 reproduces the pre-masked TIFF intensities, '
                        'higher keeps more of the 12-bit range (default 255)')
    e.add_argument('--depth-m', type=float, default=.02)
    e.add_argument('--surface-exclusion-px', type=float, default=10.)
    e.add_argument('--dx-m-per-px', type=float);e.add_argument('--dt-s', type=float)
    e.add_argument('--overwrite', action='store_true')
    w = sub.add_parser('viewer', help='Build a self-contained browser viewer for a completed pair.')
    w.add_argument('directory', help='a completed pair output directory')
    w.add_argument('--output', help='HTML path (default viewer.html inside the pair directory)')
    w.add_argument('--piv', help='supplied PIV MAT to overlay; omit to leave that layer out')
    w.add_argument('--manual-ptv', help='directory of hand-matched particle .mat files')
    w.add_argument('--piv-stride', type=int, default=4, help='decimate native PIV before embedding (default 4)')
    w.add_argument('--depth-m', type=float, help='crop depth below the surface (default: the run\'s requested depth)')
    w.add_argument('--margin-px', type=float, default=24., help='headroom above the highest surface point')
    w.add_argument('--all', action='store_true', help='treat the directory as a batch root: build a '
                   'viewer for every completed pair and an index page linking them')
    w.add_argument('--piv-dir', help='with --all, folder holding NAME_PIV.mat companions')
    m = sub.add_parser('demo', help='Generate and process a small synthetic particle pair.')
    m.add_argument('--output', required=True);m.add_argument('--generate-only', action='store_true');m.add_argument('--workers', type=int, default=1)
    return p


def main(argv=None):
    # Set conservative defaults before command handlers import NumPy/SciPy.
    # Explicit user environment settings remain authoritative.
    for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        os.environ.setdefault(variable,'1')
    p = parser();args = p.parse_args(argv)
    try:
        if args.command == 'discover':
            pairs, incomplete = discover(args.input, args.recursive, args.skip_incomplete)
            print(json.dumps({'pairs': [{'name': v.name, 'A': str(v.image_a), 'B': str(v.image_b),
                'PIV': str(v.piv_mat) if v.piv_mat else None, 'surface': str(v.surface_file) if v.surface_file else None,
                'experiment': v.experiment, 'pair_number': v.pair_number} for v in pairs], 'incomplete': incomplete}, indent=2))
            return 0
        if args.command == 'run':return run_batch(args)
        if args.command == 'compare':return compare_existing(args)
        if args.command == 'viewer':
            from .viewer import build, build_all
            if args.all:
                record = build_all(args.directory, args.piv_dir, args.manual_ptv,
                                   args.piv_stride, args.depth_m, args.margin_px)
                print('\nIndex: %s  (%d pairs, %.0f MB total)'
                      % (record['index'], record['pairs'], record['bytes']/1e6), flush=True)
                return 0
            record = build(args.directory, args.output, args.piv, args.manual_ptv,
                           args.piv_stride, args.depth_m, args.margin_px)
            print('Viewer: %s  (%.1f MB, rows %d-%d, %s)'
                  % (record['path'], record['bytes']/1e6, record['crop_rows'][0], record['crop_rows'][1],
                     ', '.join('%s %d' % kv for kv in record['layers'].items()) or 'no vector layers'),
                  flush=True)
            return 0
        if args.command == 'extract':
            from .extract import extract, discover_raw
            pairs = args.pair
            if args.first is not None or args.last is not None:
                available, _, _ = discover_raw(args.run_dir, args.experiment)
                low = args.first if args.first is not None else min(available)
                high = args.last if args.last is not None else max(available)
                pairs = sorted(set(pairs or []) | {n for n in available if low <= n <= high})
            manifest = extract(args.run_dir, args.results, args.output, args.experiment, pairs,
                               args.clip, args.depth_m, args.surface_exclusion_px,
                               args.dx_m_per_px, args.dt_s, args.overwrite)
            print('\nWrote %d pairs to %s' % (len(manifest['pairs_written']), args.output), flush=True)
            print('Config written to %s' % (Path(args.output)/'config.json'), flush=True)
            return 0
        if args.command == 'demo':
            directory, inputs = make_demo(args.output)
            command = ['run', str(inputs), '--output', str(directory/'results'), '--config', str(directory/'demo.json'), '--workers', str(args.workers)]
            print('Resume command: quadratic-optical run "'+str(inputs)+'" --output "'+str(directory/'results')+'" --config "'+str(directory/'demo.json')+'" --workers '+str(args.workers), flush=True)
            return 0 if args.generate_only else run_batch(p.parse_args(command))
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print('Error: '+str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
