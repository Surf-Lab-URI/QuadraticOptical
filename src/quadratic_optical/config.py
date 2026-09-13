"""Explicit physical calibration and image/surface conventions."""
from copy import deepcopy
from pathlib import Path
import json
import math

DEFAULTS = {
    'dx_m_per_px': None, 'dt_s': None, 'depth_m': .01,
    'grid_spacing_px': 8, 'grid_phase_px': 7,
    'surface': {'mode': 'auto', 'index_base': 1, 'offset_px': 0.},
    'availability': 'full', 'intensity_scale': 1.,
    'intensity_offset': 0., 'workers': 1,
    'integration_interval_px': 2., 'depth_step_m': .0001,
    'piv_quality': 'correlation', 'pairs': {},
}
_SURFACE_KEYS = {'mode', 'index_base', 'offset_px', 'row_a', 'row_b'}


def merge(base, changes):
    if not isinstance(changes, dict):
        raise ValueError('Configuration overrides must be JSON objects.')
    out = deepcopy(base)
    for key, value in changes.items():
        if key not in out:
            raise ValueError('Unknown configuration key: ' + key)
        if key == 'surface':
            if not isinstance(value, dict):
                raise ValueError('surface must be a JSON object.')
            unknown = set(value) - _SURFACE_KEYS
            if unknown:
                raise ValueError('Unknown surface configuration key: ' + ', '.join(sorted(unknown)))
            out[key].update(value)
        elif isinstance(out[key], dict) and isinstance(value, dict) and key != 'pairs':
            out[key] = merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path=None):
    out = deepcopy(DEFAULTS)
    if path is not None:
        with Path(path).open(encoding='utf8') as f:
            user = json.load(f)
        if not isinstance(user, dict):
            raise ValueError('Configuration must be a JSON object.')
        out = merge(out, user)
    return validate(out)


def for_pair(config, name):
    out = deepcopy(config)
    overrides = out.pop('pairs', {}).get(name, {})
    if not isinstance(overrides, dict):
        raise ValueError('Each pair override must be a JSON object.')
    if 'pairs' in overrides:
        raise ValueError('Nested pair overrides are not supported.')
    return validate(merge(out, overrides))


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def validate(c):
    for key in ['dx_m_per_px', 'dt_s', 'depth_m', 'intensity_scale', 'integration_interval_px', 'depth_step_m']:
        value = c[key]
        if value is None and key in ['dx_m_per_px', 'dt_s']:
            continue
        if not _number(value) or value <= 0:
            raise ValueError(key + ' must be finite and positive.')
    if not _number(c['intensity_offset']):
        raise ValueError('intensity_offset must be finite.')
    for key in ['workers', 'grid_spacing_px']:
        value = c[key]
        if not _number(value) or int(value) != value or value < 1:
            raise ValueError(key + ' must be a positive integer.')
        c[key] = int(value)
    phase = c['grid_phase_px']
    if not _number(phase) or int(phase) != phase or not 0 <= phase < c['grid_spacing_px']:
        raise ValueError('grid_phase_px must be an integer in [0, grid_spacing_px).')
    c['grid_phase_px'] = int(phase)
    if not isinstance(c['surface'], dict) or set(c['surface']) - _SURFACE_KEYS:
        raise ValueError('surface must contain only mode, index_base, offset_px, row_a and row_b.')
    surface = c['surface']
    if surface.get('mode') not in ['auto', 'piv_mat', 'sidecar', 'constant', 'nonzero_boundary']:
        raise ValueError('Unknown surface.mode.')
    if isinstance(surface.get('index_base'), bool) or surface.get('index_base') not in [0, 1]:
        raise ValueError('surface.index_base must be 0 or 1.')
    for key in ['offset_px', 'row_a', 'row_b']:
        if key in surface and not _number(surface[key]):
            raise ValueError('surface.' + key + ' must be finite.')
    if c['availability'] not in ['full', 'nonzero_boundary']:
        raise ValueError('availability must be full or nonzero_boundary.')
    if c['piv_quality'] not in ['correlation', 'finite']:
        raise ValueError('piv_quality must be correlation or finite.')
    if 'pairs' in c and (not isinstance(c['pairs'], dict) or any(not isinstance(v, dict) for v in c['pairs'].values())):
        raise ValueError('pairs must map pair names to JSON objects.')
    if c['dx_m_per_px'] is not None and c['depth_m'] < 20 * c['dx_m_per_px']:
        raise ValueError('depth_m must reach at least 20 pixels below the surface for the common-domain integration band.')
    return c
