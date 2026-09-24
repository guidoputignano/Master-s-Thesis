"""The pre-registered predictions: limits, onset ordering, and the comparison against them."""
import numpy as np
import pandas as pd
import pytest

from analysis import experiment_predictions as ep


@pytest.fixture(scope='module')
def pred():
    return ep.predict(top=2.0, ramp_min=30.0, n=800)


def row(pred, share, onset, tau, t):
    d = pred[np.isclose(pred.senescent_share, share) & (pred.onset == onset)
             & np.isclose(pred.shear_pa, tau) & np.isclose(pred.time_h, t)]
    assert len(d) == 1
    return d.iloc[0]


def test_plateau_at_the_reference_shear_and_the_mixture(pred):
    r = row(pred, 0.0, 'step', 1.4, 24)
    assert abs(r.angle - 20.0) < 0.1 and abs(r.ar - 2.3) < 0.01          # e^-8 of the change left
    r30 = row(pred, 0.3, 'step', 1.4, 24)
    assert abs(r30.angle - (0.7 * 20.0 + 0.3 * 45.0)) < 0.1              # cell-weighted mean
    s = row(pred, 0.3, 'static', 0.0, 6)
    assert s.angle == pytest.approx(45.0) and s.ar == pytest.approx(0.7 * 1.9 + 0.3 * 2.0)


def test_ramp_lags_the_step_early_and_catches_up(pred):
    for share in (0.0, 0.3):
        step1, ramp1 = row(pred, share, 'step', 1.4, 1), row(pred, share, 'ramp', 1.4, 1)
        assert ramp1.angle > step1.angle + 1.0                            # later start: less aligned at 1 h
        step24, ramp24 = row(pred, share, 'step', 1.4, 24), row(pred, share, 'ramp', 1.4, 24)
        assert abs(ramp24.angle - step24.angle) < 0.2


def test_bands_contain_the_nominal_and_widen_with_the_literature_constant(pred):
    for _, r in pred[pred.onset != 'static'].iterrows():
        assert r.angle_model_lo - 1e-9 <= r.angle <= r.angle_model_hi + 1e-9
        assert r.angle_literature_hi >= r.angle_model_hi - 1e-9
    r6 = row(pred, 0.0, 'step', 1.4, 6)
    assert r6.angle_literature_hi - r6.angle_literature_lo > r6.angle_model_hi - r6.angle_model_lo


def test_higher_shear_aligns_more_but_not_below_the_floor(pred):
    lo, hi = row(pred, 0.0, 'step', 1.4, 24), row(pred, 0.0, 'step', 2.0, 24)
    assert hi.angle < lo.angle
    assert pred.angle_model_lo.min() >= ep.PARAMS['theta_floor_deg'][0] - 1e-9


def test_compare_accepts_model_data_and_flags_a_shifted_condition(pred):
    rng = np.random.default_rng(1)
    rows = []
    for _, r in pred.iterrows():
        for f in range(10):
            rows.append(dict(senescent_share=r.senescent_share, onset=r.onset, shear_pa=r.shear_pa,
                             time_h=r.time_h, angle_deg=r.angle + rng.normal(0, 3.0),
                             aspect_ratio=r.ar + rng.normal(0, 0.05),
                             gap_pct=0.5 * (r.gap_lo + r.gap_hi)))
    m = pd.DataFrame(rows)
    table, kin = ep.compare(m, top=2.0)
    assert (table.angle_verdict == 'inside model band').mean() > 0.9
    assert (table.gap_verdict == 'inside model band').all()
    # adaptation constant recovered from the time course
    k = kin[(kin.senescent_share == 0.0) & (kin.onset == 'step') & np.isclose(kin.shear_pa, 1.4)].iloc[0]
    assert 1.5 < k.T_h < 6.0 and abs(k.plateau_deg - 20.0) < 3.0
    shifted = m.copy()
    sel = (shifted.onset == 'step') & np.isclose(shifted.shear_pa, 1.4) & (shifted.time_h == 24) \
        & (shifted.senescent_share == 0.0)
    shifted.loc[sel, 'angle_deg'] += 15.0
    t2, _ = ep.compare(shifted, top=2.0)
    v = t2[(t2.onset == 'step') & np.isclose(t2.shear_pa, 1.4) & (t2.time_h == 24) & (t2.senescent_share == 0.0)]
    assert v.angle_verdict.iloc[0] == 'above'


def test_compare_names_the_hypothesis_the_data_follow():
    alt = ep.predict(top=4.0, n=400, hypotheses=('long_confluent', 'edge_2Pa'))
    rng = np.random.default_rng(2)
    for hyp in ('long_confluent', 'edge_2Pa'):
        d = alt[alt.hypothesis == hyp]
        rows = [dict(senescent_share=r.senescent_share, onset=r.onset, shear_pa=r.shear_pa, time_h=r.time_h,
                     angle_deg=r.angle + rng.normal(0, 2.0), aspect_ratio=r.ar)
                for _, r in d.iterrows() for _ in range(8)]
        table, _ = ep.compare(pd.DataFrame(rows), top=4.0)
        late = table[np.isclose(table.shear_pa, 4.0) & (table.time_h == 24)]
        assert (late.closest_hypothesis == hyp).all()


def test_contrast_is_the_paired_difference_and_separates_the_hypotheses():
    con = ep.contrast(top=4.0, n=400)
    m = con[(con.hypothesis == 'model') & (con.senescent_share == 0.0) & (con.onset == 'step') & (con.time_h == 24)].iloc[0]
    pred = ep.predict(top=4.0, n=10)
    hi, lo = row(pred, 0.0, 'step', 4.0, 24), row(pred, 0.0, 'step', 1.4, 24)
    assert m.d_angle == pytest.approx(hi.angle - lo.angle, abs=1e-9)
    assert m.d_angle_model_lo <= m.d_angle <= m.d_angle_model_hi
    assert m.d_angle < 0 and m.d_ar > 0                                     # more shear: more aligned, longer
    get = lambda h: con[(con.hypothesis == h) & (con.senescent_share == 0.0) & (con.onset == 'step')
                        & (con.time_h == 24)].d_angle.iloc[0]
    assert get('edge_2Pa') > 30.0                                            # perpendicular at 4 Pa
    assert get('long_confluent') == pytest.approx(36.5 - 38.0, abs=0.2)     # Robotti's 16 h points
    # senescent cells dilute the contrast by their share
    s30 = con[(con.hypothesis == 'model') & (con.senescent_share == 0.3) & (con.onset == 'step') & (con.time_h == 24)].iloc[0]
    assert s30.d_angle == pytest.approx(0.7 * m.d_angle, abs=1e-6)


def test_fields_needed_scales_with_the_inverse_square_of_the_difference():
    assert ep.fields_needed(10.0) == 8                     # 2 (2.80 x 7.1 / 10)^2 = 7.9
    assert ep.fields_needed(5.0) == 32
    assert ep.fields_needed(0.0) == float('inf')
