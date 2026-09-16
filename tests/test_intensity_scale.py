"""Thresholds in brightness units must follow the intensity conversion.

These exist because a previous fix was inert and nothing failed. The scaling was
written correctly inside the tracker, but the key it needed was missing from
load_inputs' whitelist, so the lookup silently took its default and a 12-bit run
kept using 8-bit thresholds. Any test that only exercised the arithmetic would
have passed. These check the value that actually reaches the code.
"""
import unittest
import numpy as np

from quadratic_optical.core.tracking import (load_inputs, ImageOnlyTracker,
                                             DETECTOR_MIN_PEAK, DETECTOR_BACKGROUND_MAX)
from quadratic_optical.core.fit_fields import inputs_from, base_floor, base_margin, CONTRAST_FLOOR
from quadratic_optical.prepare import contrast_floor, normalize

TWELVE_BIT = 255./1023.


def write_inputs(path, **extra):
    shape = (96, 96)
    rng = np.random.default_rng(3)
    image = rng.random(shape)*40.
    arrays = dict(A=image.copy(), B=image.copy(), rawA=image.copy(), rawB=image.copy(),
                  va=np.ones(shape, bool), vb=np.ones(shape, bool),
                  availability_a=np.ones(shape, bool), availability_b=np.ones(shape, bool),
                  surface_a=np.full(shape[1], 8.), surface_b=np.full(shape[1], 8.),
                  points=np.array([[20., 40.], [30., 50.]]), origin0=np.array([0, 0]),
                  DX=np.array(5e-5), DT=np.array(.01),
                  requested_max_depth_px=np.array(60.), fitting_max_depth_px=np.array(124.),
                  detector_max_depth_px=np.array(149.),
                  image_only=np.array(True), supplied_velocity_used=np.array(False))
    arrays.update(extra)
    np.savez_compressed(path, **arrays)
    return path


class IntensityScaleTests(unittest.TestCase):
    def setUp(self):
        import tempfile, pathlib
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix='scale_'))

    def test_detector_thresholds_follow_the_conversion(self):
        path = write_inputs(self.dir/'twelve.npz', intensity_scale=np.array(TWELVE_BIT))
        tracker = ImageOnlyTracker(load_inputs(path))
        self.assertAlmostEqual(tracker.intensity_scale, TWELVE_BIT, places=12)
        self.assertAlmostEqual(tracker.detector_min_peak, DETECTOR_MIN_PEAK*TWELVE_BIT, places=12)
        self.assertAlmostEqual(tracker.detector_background_max,
                               DETECTOR_BACKGROUND_MAX*TWELVE_BIT, places=12)

    def test_defaults_when_the_conversion_predates_the_key(self):
        tracker = ImageOnlyTracker(load_inputs(write_inputs(self.dir/'old.npz')))
        self.assertEqual(tracker.intensity_scale, 1.)
        self.assertEqual(tracker.detector_min_peak, DETECTOR_MIN_PEAK)
        self.assertEqual(tracker.detector_background_max, DETECTOR_BACKGROUND_MAX)

    def test_detector_floor_survives_the_whitelist(self):
        path = write_inputs(self.dir/'floor.npz', detector_min_depth_px=np.array(7.))
        self.assertEqual(ImageOnlyTracker(load_inputs(path)).detector_min_depth_px, 7.)

    def test_fitting_reads_the_stored_floor_and_margin(self):
        path = write_inputs(self.dir/'fit.npz', contrast_floor=np.array(contrast_floor(TWELVE_BIT)),
                            surface_exclusion_px=np.array(4.))
        loaded = inputs_from(path)
        self.assertAlmostEqual(base_floor(loaded), CONTRAST_FLOOR*TWELVE_BIT**2, places=12)
        self.assertEqual(base_margin(loaded), 4.)

    def test_fitting_defaults_without_those_keys(self):
        loaded = inputs_from(write_inputs(self.dir/'plain.npz'))
        self.assertEqual(base_floor(loaded), CONTRAST_FLOOR)
        self.assertEqual(base_margin(loaded), 10.)

    def test_normalization_is_invariant_under_a_rescale(self):
        rng = np.random.default_rng(11)
        image = rng.random((80, 80))*200.
        valid = rng.random((80, 80)) > .2
        full = normalize(image, valid, contrast_floor(1.))
        quarter = normalize(image*TWELVE_BIT, valid, contrast_floor(TWELVE_BIT))
        self.assertTrue(np.allclose(full, quarter, atol=1e-10))

    def test_an_unscaled_floor_would_not_be_invariant(self):
        # Guards the guard: if contrast_floor were ever made a constant again,
        # the test above must be able to fail.
        rng = np.random.default_rng(11)
        image = rng.random((80, 80))*200.
        valid = rng.random((80, 80)) > .2
        self.assertFalse(np.allclose(normalize(image, valid, CONTRAST_FLOOR),
                                     normalize(image*TWELVE_BIT, valid, CONTRAST_FLOOR), atol=1e-10))


if __name__ == '__main__':
    unittest.main()


class ReportingGridReachTests(unittest.TestCase):
    """A lowered floor must reach the saved grid, not just the acceptance test.

    finalize selects which fitting nodes to evaluate before acceptance runs, so a
    hard-coded floor there silently discards every shallow node however the
    acceptance threshold is configured. That is what happened: a sweep down to
    3 px produced a fitting grid reaching 3.03 px and a reported grid that still
    stopped at 12, with nothing raising.
    """
    def test_selection_uses_the_configured_floor(self):
        import inspect
        from quadratic_optical.core import finalize_fields
        source = inspect.getsource(finalize_fields.run)
        self.assertIn('depth >= evaluator.min_depth', source,
                      'the reporting grid must select on the configured floor')
        self.assertNotIn('depth >= 12', source,
                         'a hard-coded reporting floor makes min_depth_px inert')
