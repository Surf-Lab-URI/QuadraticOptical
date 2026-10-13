"""Add the QO velocity-profile products to a run results .mat file.

Writes into the PIV struct, alongside PIV.U_msv98 / PIV.z_msv98, because these
profiles exist at the PIV pair times:

  PIV.U_QO        [number_of_pair x Nz_QOIR]  best-available QO profile, as measured
  PIV.U_QOIR      [number_of_pair x Nz_QOIR]  the same, lightly smoothed and anchored to IR
  PIV.z_QOIR      [1 x Nz_QOIR]               depth grid, positive-up, z = 0 at the surface
  PIV.U_QOIRPIV   [number_of_pair x Nz_PIV]   U_QOIR mapped onto PIV image rows
  PIV.QO_*                                    provenance

The file is a MATLAB v7.3 (HDF5) file. Before anything is written the current
file is copied to a name carrying its own modification time, matching the
convention already used in these directories, so no version is ever lost.
Existing datasets are checksummed before and after and must be unchanged.

Usage:
  write_qo_fields.py <results.mat> <experiment> <qo_root> [--dry-run]
"""
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

# Our integration profile is sampled every 0.1 mm, so the stored grid says that
# rather than borrowing the finer msv98 grid and implying resolution we do not
# have. Comparing the two needs one interpolation; see PIVFileContents.md.
Z_STEP_MM = 0.1
Z_MAX_MM = 20.0
ACCEPTANCE = 'aggressive'


def matlab_double(group, name, array):
    """Create a MATLAB-readable double dataset. HDF5 stores dims reversed."""
    data = np.asarray(array, dtype=np.float64)
    stored = data.T if data.ndim > 1 else data.reshape(-1, 1)
    if name in group:
        del group[name]
    dataset = group.create_dataset(name, data=stored)
    dataset.attrs['MATLAB_class'] = np.bytes_(b'double')
    dataset.attrs['H5PATH'] = np.bytes_(group.name.encode())
    return dataset


def matlab_char(group, name, text):
    """Create a MATLAB-readable char array, stored as uint16 like exp_name."""
    data = np.array([ord(c) for c in text], dtype=np.uint16).reshape(-1, 1)
    if name in group:
        del group[name]
    dataset = group.create_dataset(name, data=data)
    dataset.attrs['MATLAB_class'] = np.bytes_(b'char')
    dataset.attrs['MATLAB_int_decode'] = np.int32(2)
    dataset.attrs['H5PATH'] = np.bytes_(group.name.encode())
    return dataset


def digest(path, keys):
    """Checksum named datasets so an edit elsewhere in the file is detectable."""
    out = {}
    with h5py.File(path, 'r') as handle:
        for key in keys:
            out[key] = hashlib.sha256(np.ascontiguousarray(handle[key][()]).tobytes()).hexdigest()
    return out


def collect(qo_root, experiment):
    """Read each processed pair's raw and anchored profiles, in mm and m/s."""
    found = {}
    for directory in sorted(Path(qo_root).glob(experiment + '_*')):
        anchored = directory / 'anchored_profile.npz'
        comparison = directory / 'piv_comparison.mat'
        if not anchored.exists() or not comparison.exists():
            continue
        number = int(directory.name.rsplit('_', 1)[1])
        a = np.load(anchored, allow_pickle=True)
        I = loadmat(comparison, squeeze_me=True)['integrals']
        get = lambda k: np.asarray(I[k].item() if I[k].shape == () else I[k], dtype=float)
        found[number] = dict(
            anchored_depth_mm=np.asarray(a['depth_mm'], float),
            anchored_u=np.asarray(a['u_cm_per_s'], float) / 100.,
            raw_depth_mm=get('depth_m') * 1000.,
            raw_u=get('of_original_mean_u_m_per_s'),
            ir=float(a['ir_surface_cm_per_s']) / 100.,
            window_mm=float(a['window_mm']))
    return found


def main():
    results = Path(sys.argv[1])
    experiment = sys.argv[2]
    qo_root = Path(sys.argv[3])
    dry_run = '--dry-run' in sys.argv

    found = collect(qo_root, experiment)
    if not found:
        raise SystemExit('No processed pairs with an anchored profile under ' + str(qo_root))

    with h5py.File(results, 'r') as handle:
        pair_numbers = np.asarray(handle['PIV']['pairNum']).ravel()
        surfaces = np.asarray(handle['Surfs']['surfsPIV'])       # (Nz_PIV, 2*npairs)
        surface_pairs = np.asarray(handle['Surfs']['pairNum']).ravel()
        dx = float(np.asarray(handle['Surfs']['dx']).ravel()[0])
        rows_piv = surfaces.shape[0]
    npairs = pair_numbers.size

    depth_mm = np.arange(0., Z_MAX_MM + .5 * Z_STEP_MM, Z_STEP_MM)
    z_m = -depth_mm / 1000.
    U_QO = np.full((npairs, depth_mm.size), np.nan)
    U_QOIR = np.full((npairs, depth_mm.size), np.nan)
    U_QOIRPIV = np.full((npairs, rows_piv), np.nan)

    written = []
    for number, record in sorted(found.items()):
        hits = np.flatnonzero(pair_numbers == number)
        if hits.size != 1:
            print('  pair %d: not a unique PIV.pairNum, skipped' % number)
            continue
        j = int(hits[0])
        # Each profile is placed only where it has samples; nothing is
        # extended past its own shallowest and deepest point.
        for source_depth, source_u, target in (
                (record['raw_depth_mm'], record['raw_u'], U_QO),
                (record['anchored_depth_mm'], record['anchored_u'], U_QOIR)):
            good = np.isfinite(source_depth) & np.isfinite(source_u)
            if good.sum() < 2:
                continue
            d, u = source_depth[good], source_u[good]
            order = np.argsort(d)
            d, u = d[order], u[order]
            inside = (depth_mm >= d[0]) & (depth_mm <= d[-1])
            target[j, inside] = np.interp(depth_mm[inside], d, u)

        # The anchored profile is referenced to the LOCAL surface, so it is
        # placed at this pair's own mean surface row rather than the fixed
        # pre-flow row U_msv98PIV uses. The two differ by the mean wave
        # elevation, up to about 0.45 mm here.
        own = surfaces[:, np.flatnonzero(surface_pairs == number)]
        if own.size:
            row = float(np.mean(own))                       # MATLAB 1-based row
            # MATLAB column k is python index k-1; depth of column k is
            # (k - row) * dx, zero at the surface and positive downward.
            columns = np.arange(1, rows_piv + 1, dtype=float)
            col_depth_mm = (columns - row) * dx * 1000.
            good = np.isfinite(record['anchored_depth_mm']) & np.isfinite(record['anchored_u'])
            d = record['anchored_depth_mm'][good]
            u = record['anchored_u'][good]
            order = np.argsort(d)
            d, u = d[order], u[order]
            inside = (col_depth_mm >= d[0]) & (col_depth_mm <= d[-1])
            U_QOIRPIV[j, inside] = np.interp(col_depth_mm[inside], d, u)
        written.append(number)

    print('pairs written: %d (%d-%d)   grid: %d depths, 0 to %.1f mm'
          % (len(written), min(written), max(written), depth_mm.size, Z_MAX_MM))
    if dry_run:
        print('dry run: nothing written')
        return

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            cwd='/home/surflab/GitRepos/QuadraticOptical').stdout.strip()
    stamp = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(results.stat().st_mtime))
    backup = results.with_name(results.stem + '_' + stamp + results.suffix)
    if backup.exists():
        raise SystemExit('Backup already exists, refusing to overwrite: ' + str(backup))
    print('backing up current file to %s' % backup.name)
    shutil.copy2(results, backup)
    if backup.stat().st_size != results.stat().st_size:
        raise SystemExit('Backup size differs from the original; aborting before any write.')

    existing = ['PIV/U_msv98', 'PIV/U_msv98PIV', 'PIV/z_msv98', 'PIV/t', 'PIV/pairNum',
                'PIV/IR_idx', 'Surfs/surfsPIV', 'Surfs/eta', 'USurf/usurf0', 'USurf/usurffilt']
    before = digest(results, existing)

    with h5py.File(results, 'r+') as handle:
        piv = handle['PIV']
        matlab_double(piv, 'U_QO', U_QO)
        matlab_double(piv, 'U_QOIR', U_QOIR)
        matlab_double(piv, 'z_QOIR', z_m.reshape(1, -1))
        matlab_double(piv, 'U_QOIRPIV', U_QOIRPIV)
        matlab_double(piv, 'QO_window_mm', np.array([[found[written[0]]['window_mm']]]))
        matlab_double(piv, 'QO_pairs_written', np.asarray(written, float).reshape(1, -1))
        matlab_char(piv, 'QO_commit', commit)
        matlab_char(piv, 'QO_acceptance', ACCEPTANCE)
        matlab_char(piv, 'QO_written', time.strftime('%Y-%m-%d %H:%M:%S %Z'))
        matlab_char(piv, 'QO_note',
                    'U_QO is the QO profile as measured, averaged horizontally over every '
                    'accepted vector at each depth below the local surface. U_QOIR is that '
                    'profile smoothed by a moving average along the curve and joined by a '
                    'straight line to the IR surface velocity at z=0. U_QOIRPIV places '
                    'U_QOIR on PIV image rows using each pair row mean surface, not the '
                    'fixed pre-flow row used by U_msv98PIV. NaN where no profile exists. '
                    'See PIVFileContents.md.')

    after = digest(results, existing)
    changed = [k for k in existing if before[k] != after[k]]
    if changed:
        raise SystemExit('EXISTING DATA CHANGED: ' + ', '.join(changed) + ' -- restore from ' + str(backup))
    print('existing datasets unchanged (%d checked)' % len(existing))
    print('commit recorded: %s' % commit[:12])


if __name__ == '__main__':
    main()
