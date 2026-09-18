"""Comparison against sparse hand-matched particle pairs.

Manual picks are a held-out reference, read only after the prediction is frozen.
They are not detection seeds, optimization constraints, training labels, or
corrections to the field, and nothing here writes into the frozen arrays.

Each source file holds one image pair's matched particles as ``p_orig`` with
shape ``[N, 2, 2]``: particle, then frame (1 = A, 2 = B), then axis (1 = x
column, 2 = y row), in MATLAB one-based original-camera coordinates. Files are
matched to a pair by the ``exp_name`` and ``image_pair_number`` stored inside
them, not by filename, because the picking tool names files with a hyphen where
the image pairs use an underscore.
"""
from pathlib import Path
import hashlib
import numpy as np

from .matio import read_mat_fields, _text
from .core.finalize_fields import ConservativeEvaluator

FIELDS = ['p_orig', 'np', 'xc', 'yc', 'exp_name', 'image_pair_number',
          'surfa_orig', 'surfb_orig', 'altmask_offset']
SHALLOW_PX = 40.


def _scalar(values, key):
    value = values.get(key)
    if value is None:
        return None
    flat = np.asarray(value).ravel()
    return float(flat[0]) if flat.size else None


def read_manual(path):
    """One manual file as zero-based arrays, with its identity and crop bounds."""
    path = Path(path)
    values = read_mat_fields(path, FIELDS, missing='ignore')
    if 'p_orig' not in values:
        raise KeyError(str(path)+' holds no p_orig array of matched particles.')
    picks = np.asarray(values['p_orig'], float)
    if picks.ndim != 3 or picks.shape[1:] != (2, 2):
        raise ValueError(str(path)+' p_orig must have shape [N,2,2]; found '+str(picks.shape))
    count = _scalar(values, 'np')
    if count is not None and int(count) != len(picks):
        raise ValueError('%s: np is %d but p_orig holds %d rows.' % (path, int(count), len(picks)))
    if not np.isfinite(picks).all():
        raise ValueError(str(path)+' p_orig holds nonfinite positions; every stored row must be matched.')
    # One-based original-camera coordinates throughout the picking tool.
    source = picks[:, 0, :]-1.
    target = picks[:, 1, :]-1.
    experiment = _text(values.get('exp_name')) if 'exp_name' in values else None
    number = _scalar(values, 'image_pair_number')
    bounds = {}
    for key in ('xc', 'yc'):
        if key in values:
            edge = np.asarray(values[key], float).ravel()
            if edge.size == 2:
                bounds[key] = [float(edge[0])-1., float(edge[1])-1.]
    surfaces = {key: np.asarray(values[key], float).ravel() for key in ('surfa_orig', 'surfb_orig')
                if key in values}
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'experiment': experiment, 'pair_number': None if number is None else int(number),
            'count': len(picks), 'source_px': source, 'target_px': target,
            'displacement_px': target-source, 'crop_zero_based': bounds,
            'altmask_offset_px': _scalar(values, 'altmask_offset'), 'surfaces': surfaces,
            'coordinate_note': 'p_orig is one-based original-camera; converted to zero-based here',
            'comparison_only': True}


def find_manual(folder, experiment, pair_number):
    """The manual file whose stored identity matches this pair, or None.

    Matching uses the fields inside each file rather than its name. A folder
    holding more than one file for the same pair is an error rather than a guess.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError('Manual pick directory not found: '+str(folder))
    matches = []
    for path in sorted(folder.glob('*.mat')):
        try:
            record = read_manual(path)
        except (KeyError, ValueError):
            continue
        if record['pair_number'] != int(pair_number):
            continue
        if experiment and record['experiment'] and record['experiment'] != experiment:
            continue
        matches.append(record)
    if len(matches) > 1:
        raise ValueError('More than one manual file claims %s pair %d: %s'
                         % (experiment, pair_number, ', '.join(m['path'] for m in matches)))
    return matches[0] if matches else None


def _summary(errors, depths, accepted):
    """Endpoint-disagreement statistics, whole set and screened, shallow and all."""
    def block(mask, label):
        values = errors[mask & np.isfinite(errors)]
        if not values.size:
            return {'subset': label, 'count': 0, 'mean_px': None, 'median_px': None, 'rms_px': None}
        return {'subset': label, 'count': int(values.size), 'mean_px': float(values.mean()),
                'median_px': float(np.median(values)), 'rms_px': float(np.sqrt((values**2).mean()))}
    shallow = np.isfinite(depths) & (depths < SHALLOW_PX)
    everything = np.ones(errors.shape, bool)
    return [block(everything, 'all picks'),
            block(accepted, 'passing vector screen'),
            block(shallow, 'depth < %g px, all' % SHALLOW_PX),
            block(shallow & accepted, 'depth < %g px, passing' % SHALLOW_PX)]


def compare_manual(directory, manual, requested_depth_m=None, acceptance=None):
    """Evaluate the frozen field at each manual source position and score it.

    The endpoint disagreement is the distance between the predicted arrow
    endpoint and the hand-matched one, which is the same as the length of the
    difference between the predicted and manual displacements.
    """
    directory = Path(directory)
    record = manual if isinstance(manual, dict) else read_manual(manual)
    frozen = np.load(directory/'results.npz', allow_pickle=True)
    if requested_depth_m is None:
        requested_depth_m = float(np.asarray(frozen['requested_depth_m']).ravel()[0])
    origin = np.asarray(frozen['origin0'], float).reshape(2) if 'origin0' in frozen.files else np.zeros(2)
    query = record['source_px']-origin
    # The comparison must score the field that was actually produced: scoring a
    # relaxed run against the default rule would report a screen the pair does
    # not use, and the 'passing' subsets would not match its own results.npz.
    evaluator = ConservativeEvaluator(directory, requested_depth_m=requested_depth_m,
                                      acceptance=acceptance)
    field = evaluator.evaluate_conservative(query)
    predicted = np.asarray(field['disp'], float)
    manual_disp = record['displacement_px']
    errors = np.linalg.norm(predicted-manual_disp, axis=1)
    accepted = np.asarray(field['accepted'], bool)
    depths = np.asarray(field['depth'], float)
    dx = float(np.asarray(frozen['DX']).ravel()[0]); dt = float(np.asarray(frozen['DT']).ravel()[0])
    return {'manual': {k: v for k, v in record.items()
                       if k not in ('source_px', 'target_px', 'displacement_px', 'surfaces')},
            'count': record['count'], 'accepted': int(accepted.sum()),
            'source_px': record['source_px'], 'manual_disp_px': manual_disp,
            'predicted_disp_px': predicted, 'accepted_mask': accepted,
            'depth_px': depths, 'endpoint_error_px': errors,
            'statistics': _summary(errors, depths, accepted),
            'DX': dx, 'DT': dt, 'origin_zero_based': origin.tolist(),
            'frozen_sha256': {name: hashlib.sha256((directory/(name+'.npz')).read_bytes()).hexdigest()
                              for name in ('results', 'inputs') if (directory/(name+'.npz')).is_file()},
            'note': ('Held-out hand-matched particles, read after the prediction was frozen. '
                     'Not seeds, constraints, training labels or corrections. Endpoint '
                     'disagreement is the distance between predicted and manual arrow endpoints.')}
