"""Portable synthetic checks; no experiment files or supplied PIV are read."""
import json
import contextlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter

from quadratic_optical.core import tracking, fit_fields, finalize_fields, integrate_profile
from quadratic_optical.core.fit_local_cached import fit_local
from quadratic_optical.core._provenance import source_hashes, TRACKING_SOURCES


def translated(a, dx, dy, fill=0):
    b = np.full_like(a, fill)
    h, w = a.shape
    x0, x1 = max(0, -dx), min(w, w-dx)
    y0, y1 = max(0, -dy), min(h, h-dy)
    b[y0+dy:y1+dy, x0+dx:x1+dx] = a[y0:y1, x0:x1]
    return b


def particle_images(shape=(160, 192), shift=(6, -3)):
    rng = np.random.RandomState(1907)
    spikes = np.zeros(shape)
    count = int(np.prod(shape)*.035)
    y = rng.randint(0, shape[0], count)
    x = rng.randint(0, shape[1], count)
    np.add.at(spikes, (y, x), rng.uniform(80, 300, count))
    a = gaussian_filter(spikes, .9)
    return a, translated(a, *shift)


def image_inputs(shape=(160, 192), shift=(6, -3)):
    a, b = particle_images(shape, shift)
    yy, xx = np.indices(shape)
    sa = np.full(shape[1], 10.)
    sb = sa+shift[1]
    va = yy >= sa[None]+10
    vb = translated(va, *shift, False)
    px, py = np.meshgrid([60., 80., 100., 120.], [50., 70., 90.])
    return dict(A=fit_fields.normalize(a, va), B=fit_fields.normalize(b, vb),
                rawA=a, rawB=b, va=va, vb=vb,
                availability_a=va, availability_b=vb,
                surface_a=sa, surface_b=sb, points=np.c_[px.ravel(), py.ravel()],
                origin0=np.array([0, 0]), DX=np.array(.0001), DT=np.array(.01),
                requested_max_depth_px=np.array(100.), fitting_max_depth_px=np.array(150.),
                detector_max_depth_px=np.array(150.), image_only=np.array(True),
                supplied_velocity_used=np.array(False))


class CoreTests(unittest.TestCase):
    def test_signed_search_and_common_pixel_formula(self):
        a, _ = particle_images((240, 264))
        va = np.ones(a.shape, bool)
        va[:18] = False
        for dx, dy, center in [(23, -17, (130, 120)), (-29, 19, (130, 120)),
                               (-64, 0, (130, 120)), (0, 64, (130, 120)),
                               (15, 6, (7, 90)), (-23, -8, (256, 120))]:
            b = translated(a, dx, dy)
            vb = translated(va, dx, dy, False)
            score, count, bounds, _, minimum = tracking.translation_ncc(a, b, va, vb, center)
            seeds, scores, common = tracking.peaks(score, count, bounds, 64)
            np.testing.assert_array_equal(seeds[0], [dx, dy])
            self.assertAlmostEqual(scores[0], 1., places=10)
            self.assertGreaterEqual(common[0]+1e-8, minimum)
            cx, cy = center
            x0, x1 = max(0, cx-19), min(a.shape[1], cx+20)
            y0, y1 = max(0, cy-19), min(a.shape[0], cy+20)
            mask = va[y0:y1, x0:x1] & vb[y0+dy:y1+dy, x0+dx:x1+dx]
            aa = a[y0:y1, x0:x1][mask]
            bb = b[y0+dy:y1+dy, x0+dx:x1+dx][mask]
            aa, bb = aa-aa.mean(), bb-bb.mean()
            direct = np.dot(aa, bb)/np.sqrt(np.dot(aa, aa)*np.dot(bb, bb))
            self.assertAlmostEqual(direct, score[dy-bounds[2], dx-bounds[0]], places=10)

    def test_invalid_overlap_and_nonfinite_candidates(self):
        a, _ = particle_images()
        tiny = np.zeros(a.shape, bool)
        tiny[77:82, 93:98] = True
        score, count, bounds, source, required = tracking.translation_ncc(a, a, tiny, tiny, (96, 80))
        self.assertEqual(source, 25)
        self.assertEqual(required, 40)
        self.assertTrue(np.all(score == -2))
        score[0, 0], score[1, 1] = np.nan, np.inf
        self.assertTrue(np.isnan(tracking.peaks(score, count, bounds, 64)[0]).all())
        p = np.zeros((2, 3))
        bad = dict(ncc=np.nan, mindet=1., support_fraction=1.)
        good = dict(ncc=.9, mindet=1., support_fraction=1.)
        self.assertFalse(tracking.admissible(p, bad))
        self.assertEqual(tracking.best_candidate([(p, bad), (p, good)], [0, 1]), (1, True))

    def test_input_guards_inspect_all_field_names(self):
        inp = image_inputs()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'inputs.npz'
            np.savez(path, **inp)
            for load in [tracking.load_inputs, fit_fields.inputs_from, finalize_fields.load_geometry]:
                load(path)
            for forbidden in ['supplied_dx', 'classical_at_points', 'manual_truth', 'flow', 'delta_x1']:
                np.savez(path, **dict(inp, **{forbidden: np.array([123.])}))
                for load in [tracking.load_inputs, fit_fields.inputs_from, finalize_fields.load_geometry]:
                    with self.assertRaises(ValueError):
                        load(path)

    def test_local_translation_and_exact_cache_thread_equivalence(self):
        a, b = particle_images(shift=(-7, 5))
        a = (a-a.mean())/a.std()
        b = (b-b.mean())/b.std()
        va = np.ones(a.shape, bool)
        vb = translated(va, -7, 5, False)
        centers = [(70., 60.), (110., 90.), (6., 90.)]
        gradient, mask = np.gradient(b), vb.astype(float)
        def task(center, cached):
            return fit_local(a, b, va, vb, center, [-6.5, 4.7], r=13, order=2,
                             reg=5e-5, target_gradient=gradient if cached else None,
                             target_mask_float=mask if cached else None)
        serial = [task(center, False) for center in centers]
        with ThreadPoolExecutor(max_workers=2) as pool:
            cached = list(pool.map(lambda center: task(center, True), centers))
        for (p, s), (q, t) in zip(serial, cached):
            np.testing.assert_allclose(p, q, atol=0, rtol=0, equal_nan=True)
            self.assertEqual(set(s), set(t))
            for key in s:
                np.testing.assert_allclose(s[key], t[key], atol=0, rtol=0, equal_nan=True)
        for p, stats in serial[:2]:
            np.testing.assert_allclose(p[:, 0], [-7, 5], atol=.03)
            np.testing.assert_allclose(p[:, 1:3]/13, 0, atol=.004)
            self.assertGreater(stats['ncc'], .999)

    def test_blend_exact_polynomial_and_derivative(self):
        centers = np.array([[0., 0.], [8., 0.], [0., 8.], [8., 8.]])
        r = 13.
        translation = np.array([2., -1.])
        slope = np.array([[.08, -.03], [.02, .05]])
        curvature = np.array([[.002, -.001, .0005], [-.0007, .0012, -.002]])
        def truth(q):
            x, y = q[:, 0], q[:, 1]
            basis = np.c_[.5*x*x, x*y, .5*y*y]
            return translation+q@slope.T+basis@curvature.T
        p = np.zeros((len(centers), 2, 6))
        p[:, :, 0] = truth(centers)
        for k, (x, y) in enumerate(centers):
            p[k, :, 1] = r*(slope[:, 0]+curvature[:, 0]*x+curvature[:, 1]*y)
            p[k, :, 2] = r*(slope[:, 1]+curvature[:, 1]*x+curvature[:, 2]*y)
            p[k, :, 3:] = r*r*curvature
        f = finalize_fields.FastLocalField(dict(points=centers, params=p, radius=r))
        q = np.array([[3.2, 2.1], [6.7, 4.5], [12., 8.3]])
        d, g = f.evaluate(q)
        np.testing.assert_allclose(d, truth(q), atol=2e-15)
        eps = 1e-4
        for axis in range(2):
            offset = np.eye(2)[axis]*eps
            fd = (f.evaluate(q+offset)[0]-f.evaluate(q-offset)[0])/(2*eps)
            np.testing.assert_allclose(g[:, :, axis], fd, atol=2e-10)

    def test_quadratic_deformation_recovery_from_continuous_particles(self):
        # Generate B by inverting a known map and evaluating continuous Gaussian
        # particles, independently of the solver's bilinear image interpolation.
        rng = np.random.RandomState(7)
        n, radius = 111, 19.
        center, translation = np.array([55., 55.]), np.array([8.4, -3.7])
        slope = np.array([[.08, -.06], [.04, -.03]])
        curvature = .5*np.array([[.004, -.002, .0025], [-.002, .0015, -.003]])
        yy, xx = np.indices((n, n))
        xy = np.stack([xx, yy], axis=-1).astype(float)
        particles = rng.uniform(-20, n+20, (450, 2))
        amplitudes = rng.uniform(.5, 1.5, 450)
        def displacement(pos):
            z = pos-center
            q = np.stack([.5*z[..., 0]**2, z[..., 0]*z[..., 1], .5*z[..., 1]**2], -1)
            return translation+z@slope.T+q@curvature.T
        def texture(pos):
            flat = pos.reshape(-1, 2)
            values = np.empty(len(flat))
            for first in range(0, len(flat), 256):
                delta = flat[first:first+256, None]-particles
                values[first:first+256] = np.exp(-np.sum(delta*delta, axis=2)/(2*1.25**2))@amplitudes
            return values.reshape(pos.shape[:-1])
        source = center+(xy-center-translation)@np.linalg.inv(np.eye(2)+slope).T
        for _ in range(16):
            z = source-center
            jac = np.broadcast_to(np.eye(2)+slope, source.shape[:-1]+(2, 2)).copy()
            jac[..., 0] += z[..., 0, None]*curvature[:, 0]+z[..., 1, None]*curvature[:, 1]
            jac[..., 1] += z[..., 0, None]*curvature[:, 1]+z[..., 1, None]*curvature[:, 2]
            residual = source+displacement(source)-xy
            source -= np.linalg.solve(jac, residual[..., None])[..., 0]
        np.testing.assert_allclose(source+displacement(source), xy, atol=1e-9)
        a, b = texture(xy), .88*texture(source)+.12
        a, b = (a-a.mean())/a.std(), (b-b.mean())/b.std()
        mask = np.ones(a.shape, bool)
        p1, _ = fit_local(a, b, mask, mask, center, translation+[1., -.8], r=19, order=1, reg=5e-5)
        p2, stats = fit_local(a, b, mask, mask, center, translation+[1., -.8], r=19,
                             order=2, reg=5e-5, seed_affine=p1)
        np.testing.assert_allclose(p2[:, 0], translation, atol=.04)
        np.testing.assert_allclose(p2[:, 1:3]/radius, slope, atol=.005)
        self.assertGreater(stats['ncc'], .995)
        xx, yy = np.meshgrid(np.linspace(-radius, radius, 11), np.linspace(-radius, radius, 11))
        z = np.c_[xx.ravel(), yy.ravel()]
        q = np.c_[np.ones(len(z)), z/radius, .5*(z[:, 0]/radius)**2,
                  z[:, 0]*z[:, 1]/radius**2, .5*(z[:, 1]/radius)**2]
        error1 = np.linalg.norm(q[:, :3]@p1.T-displacement(z+center))
        error2 = np.linalg.norm(q@p2.T-displacement(z+center))
        self.assertLess(error2, .2*error1)

    def test_blend_nan_neighbors_are_not_silently_removed(self):
        data = dict(points=np.array([[0., 0.], [8., 0.], [30., 0.]]),
                    params=np.zeros((3, 2, 6)), radius=13.)
        data['params'][1] = np.nan
        f = finalize_fields.FastLocalField(data)
        d, _ = f.evaluate(np.array([[0., 0.], [30., 0.], [90., 0.]]))
        self.assertTrue(np.isnan(d[0]).all())
        np.testing.assert_array_equal(d[1], [0, 0])
        self.assertTrue(np.isnan(d[2]).all())

    def test_surface_integration_missing_support_and_common_domain(self):
        checks = integrate_profile.self_test()
        self.assertTrue(checks['passed'], checks)
        x = integrate_profile.horizontal_samples(9, 2.)
        np.testing.assert_allclose(x[-3:], [7.5, 8., 8.5])
        u = np.ones((2, len(x)))
        out = integrate_profile.integrate_arrays(x, [0., .01], u, u.astype(bool),
                                                 np.array([u]*3), .001, 0., .01)
        np.testing.assert_allclose(out['covered_segment_integral_assumed_m2_per_s'], .009)

    def test_resume_identity_tracks_inputs_code_and_settings(self):
        inp = image_inputs()
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            np.savez(directory/'inputs.npz', **inp)
            fake = tracking.ImageOnlyTracker(inp).coarse_task(0)
            with patch.object(tracking.ImageOnlyTracker, 'coarse_task', return_value=fake) as fit:
                self.assertTrue(tracking.run(directory, workers=1, stop_after='coarse'))
                self.assertEqual(fit.call_count, len(inp['points']))
                before = json.loads((directory/'tracking_settings.json').read_text())
                self.assertTrue(tracking.run(directory, workers=2, stop_after='coarse'))
                self.assertEqual(fit.call_count, len(inp['points']))
                self.assertEqual(before['source_sha256'], source_hashes(TRACKING_SOURCES))
                with patch.dict(tracking.SETTINGS, dict(track_fb_max=.9)):
                    with self.assertRaises(ValueError):
                        tracking.run(directory, stop_after='coarse')
                np.savez(directory/'inputs.npz', **dict(inp, DT=np.array(.02)))
                with self.assertRaises(ValueError):
                    tracking.run(directory, stop_after='coarse')

    def test_complete_portable_pipeline_and_fit_resume(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(td)
            np.savez_compressed(directory/'inputs.npz', **image_inputs())
            self.assertTrue(tracking.run(directory, workers=2, chunk_size=64))
            report = fit_fields.run(directory, workers=2, checkpoint_every=16)
            for stage in ['main', 'affine', 'large_window', 'margin14', 'reverse']:
                self.assertTrue(report[stage]['complete'])
            result, summary = finalize_fields.run(directory, requested_depth_m=.01, batch_size=64)
            self.assertEqual(summary['accepted_grid'], 12)
            np.testing.assert_allclose(result['disp'], np.tile([6, -3], (12, 1)), atol=.03)
            profile, _ = integrate_profile.run(directory, requested_depth_m=.01,
                                                interval_px=4, depth_step_m=.002, depth_batch=3)
            self.assertGreater(np.sum(profile['sample_accepted']), 0)
            # The deliberately short synthetic fitting grid cannot support the
            # entire width or every depth; preserve that missing-domain result.
            self.assertTrue(np.isnan(profile['full_width_integral_assumed_m2_per_s']).all())
            with patch.object(fit_fields, 'fit_local', side_effect=AssertionError('Unexpected refit')):
                fit_fields.run(directory, workers=1)
            with patch.dict(fit_fields.SETTINGS, dict(regularization=1e-4)):
                with self.assertRaises(ValueError):
                    fit_fields.run(directory, stages=['reverse'])


if __name__ == '__main__':
    unittest.main()
