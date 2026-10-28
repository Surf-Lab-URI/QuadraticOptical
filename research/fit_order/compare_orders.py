"""Figures comparing a pair fitted with quadratic terms against a linear refit.

Writes into the COMPARISON pair directory (the linear one), so its viewer picks
the figures up alongside its own report:

  fitorder_profiles      velocity profiles from both fits, and their difference
  fitorder_delta_u_z     horizontal velocity difference at every display point
  fitorder_delta_w_z     vertical velocity difference at every display point

Differences are always comparison minus reference -- linear minus quadratic for
the intended use -- and are shown only where BOTH fits accepted a vector. The
maps reuse the package's own height-panel drawing, so they match field_u_z
exactly in aspect, datum and surface line.

Usage:
  compare_orders.py <reference pair dir> <comparison pair dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

from quadratic_optical import reporting as rp

REF_COLOUR = rp.OF           # quadratic, as in every other report
CMP_COLOUR = '#222222'       # linear
CLIP_PERCENTILE = 99.        # differences have a long tail; see the module notes


def best_available(directory):
    I = loadmat(Path(directory) / 'piv_comparison.mat', squeeze_me=True)['integrals']
    get = lambda k: np.asarray(I[k].item() if I[k].shape == () else I[k], dtype=float)
    return get('depth_m') * 1000., get('of_original_mean_u_m_per_s') * 100.


def anchored(directory):
    a = np.load(Path(directory) / 'anchored_profile.npz', allow_pickle=True)
    return np.asarray(a['depth_mm'], float), np.asarray(a['u_cm_per_s'], float), float(a['ir_surface_cm_per_s'])


def on_grid(depth, u, grid):
    """Profile on a common depth grid, NaN wherever it has no samples."""
    good = np.isfinite(depth) & np.isfinite(u)
    out = np.full(grid.shape, np.nan)
    if good.sum() < 2:
        return out
    d, v = depth[good], u[good]
    order = np.argsort(d); d, v = d[order], v[order]
    inside = (grid >= d[0]) & (grid <= d[-1])
    out[inside] = np.interp(grid[inside], d, v)
    return out


def profiles_figure(ref, cmp_, out, ref_label, cmp_label):
    rz, ru = best_available(ref); cz, cu = best_available(cmp_)
    if not np.array_equal(rz, cz):
        raise ValueError('The two best-available profiles are on different depth grids.')
    ra_z, ra_u, ir = anchored(ref); ca_z, ca_u, _ = anchored(cmp_)
    grid = np.round(np.arange(0., 20.0001, .1), 4)
    rows = [('As measured: each method on all of its own data', rz, ru, cz, cu),
            ('Smoothed (1 mm, along the curve) and anchored to the IR surface velocity',
             grid, on_grid(ra_z, ra_u, grid), grid, on_grid(ca_z, ca_u, grid))]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10.5),
                            gridspec_kw=dict(width_ratios=[1, 1, .85]))
    stats = {}
    for r, (title, zr, ur, zc, uc) in enumerate(rows):
        delta = uc - ur
        for c, (top, bottom) in enumerate([(0., 20.), (0., 4.)]):
            ax = axs[r, c]
            ax.plot(ur, zr, color=REF_COLOUR, lw=2., label=ref_label)
            ax.plot(uc, zc, color=CMP_COLOUR, lw=1.4, ls=(0, (5, 2)), label=cmp_label)
            ax.scatter([ir], [0.], marker='*', s=200, color='#edbd36', edgecolor='#222',
                       zorder=10, clip_on=False, label='IR surface (smoothed)')
            ax.set_ylim(bottom, top - .04 * (bottom - top))
            inside = (zr <= bottom) & np.isfinite(ur)
            if inside.any():
                lo = min(np.nanmin(ur[inside]), np.nanmin(uc[(zc <= bottom)]), ir)
                hi = max(np.nanmax(ur[inside]), np.nanmax(uc[(zc <= bottom)]), ir)
                pad = .06 * (hi - lo + 1e-9)
                ax.set_xlim(lo - pad, hi + pad)
            ax.grid(alpha=.2); ax.legend(loc='lower right', fontsize=8)
            ax.set_xlabel('Mean horizontal velocity (cm/s)')
            if c == 0:
                ax.set_ylabel('Depth below local surface (mm)')
            ax.set_title(('Full depth' if c == 0 else 'Top 4 mm'), fontsize=10, loc='left')
        ax = axs[r, 2]
        ok = np.isfinite(delta)
        ax.axvline(0., color='0.4', lw=.9)
        ax.plot(delta, zr, color=CMP_COLOUR, lw=1.4)
        ax.fill_betweenx(zr, 0., np.where(ok, delta, 0.), color=CMP_COLOUR, alpha=.12, lw=0)
        ax.set_ylim(20., -.8); ax.grid(alpha=.2)
        span = np.nanmax(np.abs(delta[ok])) if ok.any() else 1.
        ax.set_xlim(-1.15 * span, 1.15 * span)
        ax.set_xlabel('%s minus %s (cm/s)' % (cmp_label.split()[0], ref_label.split()[0]))
        ax.set_title('Difference', fontsize=10, loc='left')
        axs[r, 0].text(0., 1.09, title, transform=axs[r, 0].transAxes, fontsize=11.5,
                       fontweight='bold', va='bottom')
        key = 'as_measured' if r == 0 else 'anchored'
        band = {}
        for lo_mm, hi_mm in ((0, 1), (1, 3), (3, 8), (8, 20)):
            m = ok & (zr >= lo_mm) & (zr < hi_mm)
            band['%g-%g mm' % (lo_mm, hi_mm)] = (round(float(np.nanmax(np.abs(delta[m]))), 4)
                                                 if m.any() else None)
        stats[key] = dict(max_abs_difference_cm_s_by_depth=band,
                          max_abs_difference_cm_s=round(float(span), 4))
    fig.suptitle('%s  ·  velocity profiles, %s vs %s' % (Path(cmp_).name, ref_label, cmp_label),
                 x=.07, ha='left', y=.995, fontsize=13)
    fig.text(.07, .012, 'Same particle tracks and screening in both; the fits differ only in polynomial '
             'order. Difference panels share the full-depth axis.', fontsize=8.5, color='0.35')
    fig.subplots_adjust(left=.07, right=.98, top=.90, bottom=.07, wspace=.24, hspace=.42)
    rp.save(fig, Path(out), 'fitorder_profiles')
    return stats


def delta_maps(ref, cmp_, out, ref_label, cmp_label):
    s = np.load(Path(ref) / 'plot_samples.npz', allow_pickle=True)
    t = np.load(Path(cmp_) / 'plot_samples.npz', allow_pickle=True)
    for key in ('query', 'x_axis_px', 'depth_axis_px'):
        if not np.array_equal(s[key], t[key]):
            raise ValueError('Display grids differ on ' + key + '; differences would be misplaced.')
    x = np.asarray(s['x_axis_px'], float); z = np.asarray(s['depth_axis_px'], float)
    rows, columns = len(z), len(x)
    dx = float(s['DX']); dt = float(s['DT']); to_cm = dx / dt * 100.
    both = (np.asarray(s['accepted']) & np.asarray(t['accepted'])).reshape(rows, columns)
    d = (np.asarray(t['disp']) - np.asarray(s['disp'])).reshape(rows, columns, 2)
    du = np.where(both, d[..., 0] * to_cm, np.nan)
    dw = np.where(both, -d[..., 1] * to_cm, np.nan)          # positive up

    # The same height lattice field_panels builds: each sample keeps its own row.
    coords = (x * dx * 1000., z * dx * 1000.)
    query_y = np.asarray(s['query'])[:, 1].reshape(rows, columns)
    surface_y = query_y[0] - z[0]
    record_path = Path(cmp_) / 'surface_ir_reference.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    datum_row, datum_note = rp._still_water_datum(record)
    if datum_row is None:
        datum_row = float(np.mean(surface_y))
        datum_note = 'the mean surface of this frame pair (row %.2f) because %s' % (datum_row, datum_note)
    else:
        datum_note = 'the still-water level (row %.2f): %s' % (datum_row, datum_note)
    grid = (np.tile(coords[0], (rows, 1)), (datum_row - query_y) * dx * 1000.)
    surface_mm = (datum_row - surface_y) * dx * 1000.

    stats = dict(samples_on_grid=int(rows * columns), accepted_by_both=int(both.sum()),
                 accepted_by_one_only=int(((np.asarray(s['accepted']) ^ np.asarray(t['accepted']))).sum()))
    for name, values, label, what in (('fitorder_delta_u_z', du, 'u', 'horizontal velocity'),
                                      ('fitorder_delta_w_z', dw, 'w', 'vertical velocity, positive up')):
        finite = values[np.isfinite(values)]
        limit = float(np.percentile(np.abs(finite), CLIP_PERCENTILE)) if finite.size else 1.
        title = ('%s   Δ%s = %s − %s   (%s, where both fits accept)   [height z]'
                 % (Path(cmp_).name, label, cmp_label.split()[0], ref_label.split()[0], what))
        clip = rp._field_image_z(Path(out), name, values, 'RdBu_r', -limit, limit, grid, surface_mm,
                                 title, 'Δ%s (cm/s)' % label, datum_note, contours=None)
        stats[name] = dict(colour_limit_cm_s=round(limit, 4),
                           limit_rule='symmetric, %gth percentile of |difference|' % CLIP_PERCENTILE,
                           median_abs_cm_s=round(float(np.median(np.abs(finite))), 4),
                           max_abs_cm_s=round(float(np.max(np.abs(finite))), 4), **clip)
    return stats


def main():
    ref, cmp_ = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    ref_label = sys.argv[3] if len(sys.argv) > 3 else 'quadratic fits'
    cmp_label = sys.argv[4] if len(sys.argv) > 4 else 'linear fits'
    record = dict(reference=str(ref), comparison=str(cmp_), reference_label=ref_label,
                  comparison_label=cmp_label, difference='comparison minus reference',
                  profiles=profiles_figure(ref, cmp_, cmp_, ref_label, cmp_label),
                  maps=delta_maps(ref, cmp_, cmp_, ref_label, cmp_label))
    (cmp_ / 'fitorder_comparison.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
