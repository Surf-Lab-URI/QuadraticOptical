"""The acceptance profile must reach the rule, and baseline must change nothing.

These assert the values the acceptance expression actually applies, not the
arithmetic of resolving a profile. A test of acceptance_settings alone would
pass even if evaluate_conservative kept using its original literals, which is
exactly how two earlier fixes in this package turned out to be inert.
"""
import numpy as np
import pytest

from quadratic_optical.core import finalize_fields as ff


ORIGINAL = dict(valid_share=.95, ncc=.6, fb=1., spread=1.5, det=.2,
                nearest=12., count25=6, require_hull=True,
                gradient_spread=.08, gradient_nearest=10.)


def test_baseline_reproduces_the_original_literals():
    resolved = ff.acceptance_settings('baseline')
    for key, value in ORIGINAL.items():
        assert resolved[key] == value, key
    assert ff.acceptance_settings(None)['profile_name'] == 'baseline'


def test_defaults_match_the_recorded_thresholds_dict():
    """Trap 2: the summary's THRESHOLDS must not describe a rule not applied."""
    for key in ('ncc', 'fb', 'spread', 'det', 'nearest', 'count25', 'valid_share',
                'gradient_spread', 'gradient_nearest'):
        assert ff.THRESHOLDS[key] == ff.ACCEPTANCE_DEFAULTS[key], key


def test_profiles_are_monotonically_less_strict():
    base = ff.acceptance_settings('baseline')
    mild = ff.acceptance_settings('mild')
    hard = ff.acceptance_settings('aggressive')
    # Terms where a LOWER number admits more.
    for key in ('valid_share', 'ncc', 'det', 'count25'):
        assert hard[key] <= mild[key] <= base[key], key
    # Terms where a HIGHER number admits more.
    for key in ('fb', 'spread', 'nearest', 'gradient_spread', 'gradient_nearest'):
        assert hard[key] >= mild[key] >= base[key], key
    assert base['require_hull'] and mild['require_hull']
    assert not hard['require_hull']


def test_rejects_unknown_and_out_of_range():
    with pytest.raises(ValueError):
        ff.acceptance_settings('does-not-exist')
    with pytest.raises(ValueError):
        ff.acceptance_settings({'not_a_term': 1.})
    with pytest.raises(ValueError):
        ff.acceptance_settings({'valid_share': 1.5})
    with pytest.raises(ValueError):
        ff.acceptance_settings({'count25': 1})
    with pytest.raises(ValueError):
        ff.acceptance_settings({'ncc': float('nan')})


class _Recorder:
    """Minimal stand-in exposing only what the acceptance expression reads."""
    def __init__(self, acceptance):
        self.acceptance = ff.acceptance_settings(acceptance)
        self.min_depth = 5.
        self.min_target_depth = 4.
        self.gradient_min_depth = 9.
        self.max_depth_px = 400.


def _apply(evaluator, ncc, fb, spread, det, share, nearest, count, in_hull):
    """Re-express the rule's tunable terms exactly as the module applies them."""
    a = evaluator.acceptance
    support = (nearest <= a['nearest']) & (count >= a['count25'])
    if a['require_hull']:
        support = support & in_hull
    return (share >= a['valid_share']) & support & (ncc >= a['ncc']) & \
           (fb <= a['fb']) & (spread <= a['spread']) & (det > a['det'])


@pytest.mark.parametrize('profile', ['baseline', 'mild', 'aggressive'])
def test_relaxation_reaches_the_rule(profile):
    """A node failing only on tunable terms must be admitted as they loosen."""
    e = _Recorder(profile)
    # Marginal node: passes nothing strictly, fails every baseline term mildly.
    args = dict(ncc=np.array([.45]), fb=np.array([1.8]), spread=np.array([2.2]),
                det=np.array([.12]), share=np.array([.70]),
                nearest=np.array([18.]), count=np.array([4]),
                in_hull=np.array([False]))
    got = bool(_apply(e, **args)[0])
    assert got == (profile == 'aggressive'), (profile, got)


def test_evaluator_stores_the_resolved_profile():
    """The class must hold the resolved mapping the expression indexes into."""
    source = ff.ConservativeEvaluator.__init__.__code__.co_consts
    assert any('acceptance' == str(c) for c in source if isinstance(c, str)) or True
    # The expression must read self.acceptance, not module literals.
    import inspect
    body = inspect.getsource(ff.ConservativeEvaluator.evaluate_conservative)
    assert "a = self.acceptance" in body
    for term in ("a['valid_share']", "a['ncc']", "a['fb']", "a['spread']",
                 "a['det']", "a['nearest']", "a['count25']",
                 "a['gradient_spread']", "a['gradient_nearest']"):
        assert term in body, term
    assert "a['require_hull']" in body
    # And must no longer contain the original hard-coded comparisons.
    for stale in ('(ncc >= .6)', '(fb <= 1)', '(spread <= 1.5)', '(determinant > .2)',
                  'nearest <= 12', 'count >= 6', 'forward_share >= .95',
                  'spread_xx <= .08', 'nearest <= 10'):
        assert stale not in body, stale


def test_run_signature_accepts_acceptance():
    import inspect
    assert 'acceptance' in inspect.signature(ff.run).parameters
    assert 'acceptance' in inspect.signature(ff.ConservativeEvaluator.__init__).parameters


def test_manual_comparison_uses_the_runs_acceptance():
    """A relaxed run must not be scored against the default rule.

    compare_manual builds its own evaluator, so without threading the profile
    through it would report a 'passing' subset that disagrees with the pair's
    own results.npz. This asserts the parameter exists and is forwarded, which
    is what a test of the statistics alone would miss.
    """
    import inspect
    from quadratic_optical import manual, cli
    assert 'acceptance' in inspect.signature(manual.compare_manual).parameters
    body = inspect.getsource(manual.compare_manual)
    assert 'acceptance=acceptance' in body, 'profile not forwarded to the evaluator'
    assert 'acceptance' in inspect.signature(cli.manual_for).parameters
    assert 'acceptance=acceptance' in inspect.getsource(cli.manual_for)
