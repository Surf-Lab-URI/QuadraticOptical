"""Shared source identities and image-only input boundary checks."""
import hashlib
from pathlib import Path


def forbidden_input_keys(keys):
    """Inspect names without loading excluded array payloads."""
    return [k for k in keys if k != 'supplied_velocity_used' and
            (k.lower().startswith(('supplied_', 'classical', 'manual', 'velocity',
                                   'previous_', 'old_', 'prior_', 'tracks')) or
             k.lower() in ('disp', 'displacement', 'delta_x', 'delta_y', 'delta_z',
                           'delta_x1', 'delta_z1', 'u', 'v', 'w', 'ux', 'uy', 'piv', 'flow'))]


def source_hashes(names):
    """Content hashes, independent of installation directory or working dir."""
    directory = Path(__file__).resolve().parent
    result = {}
    for name in names:
        result[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    return result


TRACKING_SOURCES = ('tracking.py', 'fit_local_cached.py', '_provenance.py')
FITTING_SOURCES = TRACKING_SOURCES + ('fit_fields.py', 'ptv_model.py')
