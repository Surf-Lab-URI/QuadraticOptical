import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('work/mplcache')
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from scipy.io import savemat

out = Path('outputs')
z = np.load('work/final_results.npz')
c = np.load('work/coverage_local_diagnostics.npz')
t = np.load('work/ptv_tracks.npz')
q, d, dep, original = [z[k] for k in ['query', 'disp', 'depth', 'accepted']]
extra = c['support_only_additional']
grid = np.arange(200, 200 + int(z['grid_count']))
near_original = grid[original[grid] & (dep[grid] < 40)]
near_extra = grid[extra[grid] & (dep[grid] < 40)]
assert len(near_original) == 40 and len(near_extra) == 4
assert np.array_equal(np.where(extra[:200])[0], [64])
assert not np.any(extra & original)
sa = z['surface_a']
xs = np.arange(len(sa))
tp, td = t['points'], t['disp']
depth_tracks = tp[:, 1] - np.interp(tp[:, 0], xs, sa)
shallow = t['accepted'] & (depth_tracks < 20)
deeper = t['accepted'] & (depth_tracks >= 20) & (depth_tracks < 40)
assert shallow.sum() == 21 and deeper.sum() == 69
A = np.asarray(Image.open('data/ExpLCL_1_03-123_imgA.tif'))
CYAN, AMBER, INK, MUTED = '#25c4dc', '#ffc05b', '#17263b', '#536170'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

def background(ax, limits=(60, 390, 102, 0)):
    ax.imshow(A, cmap='gray', vmin=35, vmax=180, origin='upper', extent=(-.5, 500.5, 500.5, -.5))
    ax.fill_between(xs, sa, sa + 10, color='#946ca6', alpha=.4, lw=0)
    ax.plot(xs, sa, color='#f4d9ed', lw=1.1)
    ax.set(xlim=limits[:2], ylim=limits[2:], xlabel='x (px)', ylabel='y (px, downward)')
    ax.set_aspect('equal')

def arrows(ax, xy, uv, color, width=.003):
    return ax.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], color=color,
                     angles='xy', scale_units='xy', scale=1, width=width,
                     headwidth=3.8, headlength=4.5, headaxislength=4.1,
                     pivot='tail', minlength=.15)

fig = plt.figure(figsize=(14, 8.1))
gs = fig.add_gridspec(2, 2, left=.065, right=.97, top=.85, bottom=.15,
                      width_ratios=[4.0, 1.15], wspace=.22, hspace=.43)
ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
detail, notes = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
fig.text(.065, .955, 'Near-surface coverage: estimates and particle evidence',
         fontsize=19, weight='bold', color=INK)
fig.text(.065, .913, 'A → B displacement in pixels per pair  ·  true-scale arrows  ·  source depths below 40 px',
         fontsize=11, color=MUTED)
background(ax0)
arrows(ax0, q[near_original], d[near_original], CYAN)
arrows(ax0, q[near_extra], d[near_extra], AMBER, .004)
ax0.scatter(q[near_extra, 0], q[near_extra, 1], s=24, facecolors='none', edgecolors=AMBER, lw=1.1)
ax0.scatter(q[64, 0], q[64, 1], s=95, facecolors='none', edgecolors='white', lw=1.2)
ax0.set_title('Continuous field · 40 previously shown + 4 provisional additions',
              loc='left', fontsize=12, weight='bold', color=INK, pad=10)
ax0.legend(handles=[Line2D([0], [0], color=CYAN, lw=2.5, label='Original screen'),
                    Line2D([0], [0], color=AMBER, lw=2.5, label='Window-support rule relaxed')],
           bbox_to_anchor=(0, -.22), loc='upper left', ncol=2, frameon=False, borderaxespad=0, fontsize=9)

background(detail, (186, 216, 78, 54))
arrows(detail, q[64:65], z['manual_truth'][64:65], 'white', .025)
arrows(detail, q[64:65], d[64:65], AMBER, .012)
detail.set_title('Closest manual pick · 65', loc='left', fontsize=11, weight='bold', pad=10)
detail.set_xticks([190, 200, 210])
detail.set_yticks([60, 70])
detail.set_ylabel('')
detail.text(0, -.35, 'White: manual   Amber: model\nDepth: 16.71 px\nEndpoint difference: 0.355 px',
            transform=detail.transAxes, va='top', fontsize=9, color=MUTED, linespacing=1.6)

background(ax1)
arrows(ax1, tp[deeper], td[deeper], CYAN, .0027)
arrows(ax1, tp[shallow], td[shallow], AMBER, .003)
ax1.scatter(tp[shallow, 0], tp[shallow, 1], s=9, c=AMBER)
ax1.set_title('Sparse automatic tracks · 90 candidates, reaching 14.35 px depth',
              loc='left', fontsize=12, weight='bold', color=INK, pad=10)
ax1.legend(handles=[Line2D([0], [0], color=CYAN, lw=2.5, label='69 tracks at 20–40 px depth'),
                    Line2D([0], [0], color=AMBER, lw=2.5, label='21 tracks below 20 px depth')],
           bbox_to_anchor=(0, -.22), loc='upper left', ncol=2, frameon=False, borderaxespad=0, fontsize=9)
notes.axis('off')
notes.text(0, .97, 'What changed', fontsize=12, weight='bold', color=INK, va='top')
notes.text(0, .81, 'The velocity estimates are\nunchanged. Four grid points\nare shown as candidates after\nreviewing boundary support.\n\nThe sparse tracks already\nexisted; they provide evidence\nat additional locations.\n\nNeither tier establishes\naccuracy closer than the\nnearest manual pick.', color=MUTED, va='top', fontsize=10, linespacing=1.5)
fig.text(.065, .036, 'Purple: first 10 px below the traced surface. No manual validation above 16.71 px depth; gradient coverage is unchanged.',
         fontsize=10, color=MUTED)
for suffix in ['png', 'svg']:
    fig.savefig(out / ('coverage_review.' + suffix), dpi=230, facecolor='white')
plt.close(fig)

with (out / 'coverage_review.csv').open('w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['query_type', 'id_one_based', 'x_px', 'y_px', 'dx_px_per_pair', 'dy_down_px_per_pair',
                 'depth_px', 'display_tier', 'original_screen', 'provisional_support_only_additional',
                 'broader_window_only_additional', 'ncc', 'forward_backward_error_px', 'method_spread_px',
                 'forward_nonfolding_share', 'reverse_nonfolding_share', 'manual_endpoint_difference_px'])
    for i in range(200 + len(grid)):
        tier = 'original_screen' if original[i] else ('provisional_boundary_support' if extra[i] else 'other_flagged')
        wr.writerow(['manual' if i < 200 else 'grid', i + 1 if i < 200 else i - 199,
                     *q[i], *d[i], dep[i], tier, int(original[i]), int(extra[i]),
                     int(c['window_only_additional'][i]), z['ncc'][i], z['fb'][i], z['method_spread'][i],
                     c['forward_nonfolding_share'][i], c['reverse_nonfolding_share'][i],
                     z['manual_error'][i] if i < 200 else ''])

savemat(out / 'coverage_review.mat', {
    'query_xy_zero_based': q[:927], 'displacement_px_per_pair': d[:927],
    'depth_px': dep[:927], 'original_screen': original[:927].astype('uint8'),
    'provisional_support_only_additional': extra[:927].astype('uint8'),
    'broader_window_only_additional': c['window_only_additional'][:927].astype('uint8'),
    'gradient_accepted_unchanged': z['gradient_accepted'][:927].astype('uint8'),
    'manual_displacement_px_per_pair': z['manual_truth'],
    'note': 'Rows1:200 manual positions, rows201:927 original727 grid positions. Same estimates as velocity_results.mat. Provisional support-only additions waive the 85 percent mapped-window support rule, retain at least95 percent nonfolding local contributions in both directions and all original remaining tests. Broader window-only candidates also waive local nonfolding share and are not promoted in the figure. No physical calibration or time interval supplied.'
}, do_compression=True)
print('Coverage figure, 927-row table and MAT file exported. Grid536+4; near-surface40+4; tracks69+21.')
