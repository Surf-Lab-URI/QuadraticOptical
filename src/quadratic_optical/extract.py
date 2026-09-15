"""Build a standard image-pair directory from raw PIV frames and campaign surfaces.

The supported workflow reads image pairs and their surface geometry from files
laid out by name. This module writes such a directory from two other sources: the
raw 12-bit PIV frames stored one per file as ``imgPiv``, and the originally
detected surface held in a campaign results file as ``Surfs.surfsPIV``.

It exists as its own step rather than as an image reader inside ``prepare`` for
one reason above the others. The raw frames sit beside a results file full of
supplied velocities, and this package's central promise is that prediction never
reads those. An exported TIFF is unambiguously an image, so the boundary stays
where it can be checked, and the intermediate can be inspected before any fitting
is paid for.

Nothing here estimates or reads velocity. Only ``Surfs.surfsPIV`` is taken from
the campaign file, and the manifest records exactly which leaves were read.
"""
from pathlib import Path
import argparse, hashlib, json, re
import numpy as np
from PIL import Image
from scipy.io import loadmat

from .matio import read_experiment_fields

RAW_SUBDIR = Path('PIVRaw')/'PIV'
RAW_PATTERN = re.compile(r'^(?P<exp>.+)_Piv_(?P<pair>\d+)_(?P<frame>[ab])\.mat$')
SURFACE_FIELD = 'Surfs.surfsPIV'
PAIR_FIELD = 'Surfs.pairNum'
FULL_SCALE = 4095.          # 12-bit sensor range of these frames


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def discover_raw(run_directory, experiment=None):
    """Map pair number to its a/b raw frame files, for one experiment."""
    folder = Path(run_directory)/RAW_SUBDIR
    if not folder.is_dir():
        raise FileNotFoundError('No raw frame directory at '+str(folder))
    found = {}
    names = set()
    for path in sorted(folder.iterdir()):
        match = RAW_PATTERN.match(path.name)
        if not match:
            continue
        names.add(match['exp'])
        if experiment is not None and match['exp'] != experiment:
            continue
        found.setdefault(int(match['pair']), {})[match['frame']] = path
    if experiment is None:
        if len(names) != 1:
            raise ValueError('Raw frames name %d experiments (%s); pass --experiment.'
                             % (len(names), ', '.join(sorted(names)) or 'none'))
        return discover_raw(run_directory, names.pop())
    complete = {n: v for n, v in found.items() if set(v) == {'a', 'b'}}
    incomplete = sorted(set(found) - set(complete))
    return complete, incomplete, experiment


def read_frame(path):
    """One raw frame as a float array, with its stored capture stamp."""
    data = loadmat(str(path), variable_names=['imgPiv', 'ts'])
    if 'imgPiv' not in data:
        raise ValueError(str(path)+' holds no imgPiv array.')
    image = np.asarray(data['imgPiv'])
    if image.ndim != 2:
        raise ValueError(str(path)+' imgPiv is not a single two-dimensional frame.')
    stamp = data.get('ts')
    # ts is a camera clock, not the A-to-B delay; it is recorded, never used for DT.
    return image.astype(float), (float(np.asarray(stamp).ravel()[0]) if stamp is not None and np.size(stamp) else None)


def campaign_surfaces(results, experiment):
    """Originally detected surfaces per frame, keyed by pair number.

    Row 2n of Surfs.surfsPIV is frame a of pair n and row 2n+1 is frame b, which
    this checks against the stored pair labels rather than assuming.
    """
    values, metadata = read_experiment_fields(results, experiment, [SURFACE_FIELD, PAIR_FIELD])
    if SURFACE_FIELD not in values:
        raise ValueError('%s holds no %s for %s.' % (results, SURFACE_FIELD, experiment))
    rows = np.asarray(values[SURFACE_FIELD], float)
    if rows.ndim != 2:
        raise ValueError(SURFACE_FIELD+' is not a frames-by-columns array.')
    labels = np.asarray(values[PAIR_FIELD]).ravel() if PAIR_FIELD in values else None
    surfaces = {}
    for index in range(0, len(rows)-1, 2):
        pair = index//2
        if labels is not None and not (labels[index] == pair and labels[index+1] == pair):
            raise ValueError('Surface rows %d/%d carry pair labels %s/%s, not %d; the '
                             'two-frames-per-pair layout does not hold in this file.'
                             % (index, index+1, labels[index], labels[index+1], pair))
        surfaces[pair] = (rows[index], rows[index+1])
    return surfaces, metadata


def write_frame(path, values, ceiling):
    """Clipped frame as TIFF: 8-bit while it fits, 16-bit above that."""
    clipped = np.clip(values, 0., float(ceiling))
    if ceiling <= 255:
        Image.fromarray(np.rint(clipped).astype(np.uint8), mode='L').save(path)
    else:
        Image.fromarray(np.rint(clipped).astype(np.uint16)).save(path)
    return clipped


def suggested_config(ceiling, dx=None, dt=None, depth_m=.02, exclusion=10.):
    """Configuration matching what extract wrote.

    The surface is the originally detected one, so no historical offset applies.
    Raw frames carry no removed band, so availability is full and the package's
    own surface exclusion is the only near-surface mask. intensity_scale maps the
    retained range onto 0-255, which the contrast floor follows automatically.
    """
    return {'dx_m_per_px': dx, 'dt_s': dt, 'depth_m': depth_m,
            'grid_spacing_px': 8, 'grid_phase_px': 7,
            'surface': {'mode': 'sidecar', 'index_base': 0, 'offset_px': 0.},
            'availability': 'full',
            'intensity_scale': 255./float(ceiling), 'intensity_offset': 0.,
            'surface_exclusion_px': float(exclusion),
            'workers': 1, 'integration_interval_px': 2., 'depth_step_m': .0001,
            'piv_quality': 'finite', 'pairs': {}}


def extract(run_directory, results, output, experiment=None, pairs=None,
            ceiling=255, depth_m=.02, exclusion=10., dx=None, dt=None, overwrite=False,
            progress=print):
    """Write image pairs and surface sidecars for the requested pair numbers."""
    run_directory = Path(run_directory); output = Path(output)
    available, incomplete, experiment = discover_raw(run_directory, experiment)
    if not available:
        raise ValueError('No complete raw a/b frame pairs for '+experiment)
    wanted = sorted(available) if pairs is None else sorted({int(p) for p in pairs})
    missing = [p for p in wanted if p not in available]
    if missing:
        raise ValueError('Raw frames are missing for pairs: '+', '.join(map(str, missing)))
    surfaces, surface_metadata = campaign_surfaces(results, experiment)
    absent = [p for p in wanted if p not in surfaces]
    if absent:
        raise ValueError('%s has no surface rows for pairs: %s'
                         % (SURFACE_FIELD, ', '.join(map(str, absent))))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for pair in wanted:
        name = '%s_%d' % (experiment, pair)
        image_paths = {'A': output/(name+'_imgA.tif'), 'B': output/(name+'_imgB.tif')}
        sidecar = output/(name+'_surface.npz')
        if not overwrite and all(p.exists() for p in list(image_paths.values())+[sidecar]):
            progress('  %s already extracted; skipping' % name)
            continue
        record = {'pair': name, 'pair_number': pair, 'frames': {}}
        shapes = set()
        for frame, key in (('A', 'a'), ('B', 'b')):
            source = available[pair][key]
            values, stamp = read_frame(source)
            shapes.add(values.shape)
            clipped = write_frame(image_paths[frame], values, ceiling)
            record['frames'][frame] = {
                'source': str(source), 'source_sha256': sha256(source),
                'shape': list(values.shape), 'raw_min': float(values.min()),
                'raw_max': float(values.max()),
                'clipped_fraction': float(np.mean(values > ceiling)),
                'camera_stamp_not_used_for_dt': stamp,
                'written': str(image_paths[frame])}
        if len(shapes) != 1:
            raise ValueError(name+': the two raw frames have different dimensions.')
        height, width = shapes.pop()
        surface_a, surface_b = surfaces[pair]
        for label, trace in (('a', surface_a), ('b', surface_b)):
            if trace.shape != (width,):
                raise ValueError('%s: %s row for frame %s has %d columns, the frame has %d.'
                                 % (name, SURFACE_FIELD, label, trace.size, width))
            if not np.isfinite(trace).all():
                raise ValueError('%s: %s row for frame %s is not finite.' % (name, SURFACE_FIELD, label))
        np.savez_compressed(sidecar, surface_a=surface_a, surface_b=surface_b)
        record['surface'] = {'field': SURFACE_FIELD, 'source': str(results),
                             'rows_matlab': [2*pair+1, 2*pair+2],
                             'median_row': [float(np.median(surface_a)), float(np.median(surface_b))],
                             'written': str(sidecar),
                             'convention': 'originally detected surface, zero-based image rows, no offset applied'}
        records.append(record)
        progress('  %-22s clipped %.3f%%/%.3f%%  surface rows %.1f/%.1f'
                 % (name, 100*record['frames']['A']['clipped_fraction'],
                    100*record['frames']['B']['clipped_fraction'],
                    record['surface']['median_row'][0], record['surface']['median_row'][1]))
    config = suggested_config(ceiling, dx, dt, depth_m, exclusion)
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    manifest = {'schema': 1, 'experiment': experiment, 'run_directory': str(run_directory),
                'raw_pattern': str(run_directory/RAW_SUBDIR/(experiment+'_Piv_NNN_a|b.mat')),
                'results_file': str(results), 'results_sha256': sha256(results),
                'campaign_fields_read': [SURFACE_FIELD, PAIR_FIELD],
                'campaign_metadata': surface_metadata,
                'sensor_full_scale': FULL_SCALE, 'clip_ceiling': float(ceiling),
                'intensity_scale_for_config': 255./float(ceiling),
                'bit_depth_written': 8 if ceiling <= 255 else 16,
                'incomplete_raw_pairs': incomplete,
                'pairs_written': [r['pair'] for r in records],
                'velocity_fields_read': [],
                'note': ('Images are raw frames clipped at the ceiling, not the pre-masked '
                         'TIFFs. No surface band is removed; the package surface exclusion is '
                         'the only near-surface mask. Camera ts stamps are recorded but never '
                         'used as DT.'),
                'records': records}
    (output/'extract_manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description='Export raw PIV frames and campaign surfaces '
                                            'into a directory the run command can read.')
    p.add_argument('--run-dir', required=True, help='experiment run directory holding PIVRaw/PIV')
    p.add_argument('--results', required=True, help='campaign results file holding Surfs.surfsPIV')
    p.add_argument('--output', required=True)
    p.add_argument('--experiment', help='defaults to the single experiment named by the raw frames')
    p.add_argument('--pair', action='append', type=int, help='pair number; repeat to select several')
    p.add_argument('--first', type=int);p.add_argument('--last', type=int)
    p.add_argument('--clip', type=float, default=255.,
                   help='retained intensity ceiling; 255 reproduces the pre-masked TIFF intensities '
                        'exactly, higher keeps more of the 12-bit range (default 255)')
    p.add_argument('--depth-m', type=float, default=.02)
    p.add_argument('--surface-exclusion-px', type=float, default=10.)
    p.add_argument('--dx-m-per-px', type=float);p.add_argument('--dt-s', type=float)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args(argv)
    pairs = a.pair
    if a.first is not None or a.last is not None:
        available, _, _ = discover_raw(a.run_dir, a.experiment)
        low = a.first if a.first is not None else min(available)
        high = a.last if a.last is not None else max(available)
        span = [n for n in sorted(available) if low <= n <= high]
        pairs = sorted(set(pairs or [])|set(span))
    manifest = extract(a.run_dir, a.results, a.output, a.experiment, pairs, a.clip,
                       a.depth_m, a.surface_exclusion_px, a.dx_m_per_px, a.dt_s, a.overwrite)
    print('\nWrote %d pairs to %s' % (len(manifest['pairs_written']), a.output))
    print('Config written to %s' % (Path(a.output)/'config.json'))
    return 0
