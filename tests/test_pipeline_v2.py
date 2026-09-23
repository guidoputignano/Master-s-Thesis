"""Synthetic checks for Imaging/pipeline_v2 (features, senescence mixture, gaps) and the
verdict session (known answers)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('skimage')
pytest.importorskip('tifffile')
PIL = pytest.importorskip('PIL')

_IMAGING = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Imaging')
sys.path.insert(0, os.path.join(_IMAGING, 'Validation'))
sys.path.insert(0, os.path.join(_IMAGING, 'pipeline_v2'))
import features as ft  # noqa: E402
import senescence as sn  # noqa: E402
import verdict_session as vs  # noqa: E402

KEY20 = '0Pa_A1_19dec21_20xA_L2RA_FlatA_seq001'
KEY40 = '1.4Pa_A1_20dec21_40x_L2RA_FlatA_seq002'


def test_pixel_size_by_magnification():
    assert ft.pixel_um(KEY20) == 0.429 and ft.pixel_um(KEY40) == 0.2145


def test_cell_features_units_shape_nuclei_and_gaps():
    cells = np.zeros((60, 60), np.int32)
    cells[10:20, 10:30] = 1                       # 10 x 20 px, long axis horizontal
    cells[30:40, 30:40] = 2
    cells[0:5, 50:60] = 3                         # touches the border
    nuclei = np.zeros_like(cells)
    nuclei[12:18, 12:16] = 1                      # inside cell 1
    nuclei[12:18, 22:26] = 2                      # inside cell 1 -> two nuclei
    nuclei[34:38, 38:44] = 3                      # 8 of 24 px in cell 2: not assigned
    holes = np.zeros(cells.shape, bool)
    holes[41:45, 30:40] = True                    # right under cell 2
    df = ft.cell_features(cells, KEY20, nuclei, holes).set_index('label')
    assert df.loc[1, 'area_um2'] == pytest.approx(200 * 0.429 ** 2)
    assert df.loc[1, 'aspect_ratio'] == pytest.approx(2.0, rel=0.05)
    assert df.loc[1, 'misalign_deg'] == pytest.approx(0.0, abs=1e-6)      # along the x axis (flow)
    assert df.loc[1, 'n_nuclei'] == 2 and df.loc[2, 'n_nuclei'] == 0
    assert df.loc[1, 'nuc_area_um2'] == pytest.approx(48 * 0.429 ** 2)
    assert bool(df.loc[3, 'touches_border']) and not bool(df.loc[1, 'touches_border'])
    assert bool(df.loc[2, 'hole_adjacent']) and not bool(df.loc[1, 'hole_adjacent'])


def _mixture(n, frac, ratio, sd=0.35, seed=0):
    rng = np.random.default_rng(seed)
    big = rng.random(n) < frac
    return np.exp(np.where(big, np.log(400 * ratio), np.log(400)) + rng.normal(0, sd, n))


def test_mixture_recovers_fraction_and_ratio():
    x = np.log(_mixture(3000, 0.30, 2.3))
    free = sn.fit(x)
    assert free['frac_enlarged'] == pytest.approx(0.30, abs=0.05)
    assert free['median_ratio'] == pytest.approx(2.3, rel=0.1)
    assert free['bic1'] - free['bic2'] > 10                               # two components preferred
    fixed = sn.fit(x, fixed_log_ratio=np.log(sn.CHALA_RATIO))
    assert fixed['median_ratio'] == pytest.approx(sn.CHALA_RATIO)
    assert fixed['frac_enlarged'] == pytest.approx(0.30, abs=0.05)
    p = sn.posterior(np.log([200.0, 400.0, 900.0, 2000.0]), free['params'])
    assert np.all(np.diff(p) > 0) and p[0] < 0.05 and p[-1] > 0.95


def test_equal_variance_resists_skew_only_solution():
    # One skewed population (log-area with a heavy right tail), no second population:
    # the free-variance fit can call most cells "enlarged" at a small ratio; the
    # shifted-population model must not report a large enlarged fraction at a large ratio.
    rng = np.random.default_rng(5)
    x = np.log(400) + rng.gamma(4.0, 0.12, 4000) - 0.48
    eq = sn.fit(x)
    assert eq['frac_enlarged'] < 0.3 or eq['median_ratio'] < 1.6
    fr = sn.fit(x, equal_var=False)
    assert fr['sd_large'] > fr['sd_small'] or fr['frac_enlarged'] < 0.3


def test_mixture_single_population_prefers_one_component():
    x = np.log(_mixture(3000, 0.0, 1.0, seed=3))
    f = sn.fit(x)
    assert f['bic1'] - f['bic2'] < 0


def test_summarise_bootstraps_by_field():
    rows = []
    for field in range(8):
        a = _mixture(300, 0.25, 2.3, seed=field)
        rows.append(pd.DataFrame({'key': f'f{field}', 'condition': 'c', 'area_um2': a}))
    s = sn.summarise(pd.concat(rows), n_boot=40).iloc[0]
    assert s.n_fields == 8 and s.frac_lo < s.frac_enlarged < s.frac_hi and s.frac_hi - s.frac_lo < 0.2
    assert s.frac_enlarged == pytest.approx(0.25, abs=0.05) and s.ratio_lo < 2.3 < s.ratio_hi


def test_gaps_from_cells_drops_lines_keeps_holes():
    an = pytest.importorskip('analyze')
    cells = np.ones((80, 80), np.int32)
    cells[:, 40:] = 2
    cells[:, 40] = 0                              # 1 px unlabelled line between two cells
    cells[10:20, 10:20] = 0                       # 10 x 10 px = 18.4 um^2 at 20x: a gap
    cells[60:63, 10:13] = 0                       # 3 x 3 px = 1.7 um^2: below the minimum
    gaps = an.gaps_from_cells(cells, 0.429)
    assert gaps[10:20, 10:20].mean() > 0.8 and not gaps[:, 40].any() and not gaps[60:63, 10:13].any()


def test_nematic_order():
    an = pytest.importorskip('analyze')
    d = pd.DataFrame({'touches_border': False, 'aspect_ratio': 2.0, 'axial_deg': [10, 10, 10, 10, 10, 10.0]})
    s, axis = an.nematic(d)
    assert s == pytest.approx(1.0) and axis == pytest.approx(10.0)
    d = pd.DataFrame({'touches_border': False, 'aspect_ratio': 2.0, 'axial_deg': [0, 45, 90, 135, 0, 90.0]})
    assert an.nematic(d)[0] < 0.4


def test_verdict_session_embed_code_and_score(tmp_path):
    from PIL import Image
    items = [dict(id=f'x{i}', question='One cell?', image=Image.new('RGB', (8, 8), (i, 0, 0)),
                  block='cell', method='v1' if i < 2 else 'v2') for i in range(4)]
    page = vs.write_session(items, str(tmp_path), title='t', seed=1, embed=True)
    html = open(page).read()
    assert html.count('data:image/jpeg;base64,') == 4 and 'v1' not in html and 'v2' not in html
    key = pd.read_csv(tmp_path / 'key.csv')
    assert list(key.verdict_id) == ['V0001', 'V0002', 'V0003', 'V0004']
    answers = ''.join('Y' if m == 'v2' else 'N' for m in key.method)
    t = vs.score(str(tmp_path), by=['method'], code=f'VS4:{answers}').set_index('method')
    assert t.loc['v1', 'yes_rate'] == 0 and t.loc['v2', 'yes_rate'] == 1
    assert list(vs.decode('VS3:Y-U').answer) == ['yes', '', 'unsure']
    with pytest.raises(ValueError):
        vs.decode('VS3:YY')
    with pytest.raises(ValueError):
        vs.score(str(tmp_path), by=['method'], code='VS3:YYY')


def test_dark_gaps_separates_gaps_from_missed_cells():
    an = pytest.importorskip('analyze')
    cells = np.zeros((120, 120), np.int32)
    nuclei = np.zeros_like(cells)
    cad = np.full(cells.shape, 100.0)                 # cytoplasm haze everywhere
    k = 0
    for y in range(0, 120, 30):
        for x in range(0, 120, 30):
            k += 1
            cells[y + 1:y + 29, x + 1:x + 29] = k
            cad[y + 1:y + 29, x + 1:x + 29] = 80 + 4 * k          # cell-to-cell variation of the haze
            nuclei[y + 12:y + 18, x + 12:x + 18] = k
    cells[cells == 6] = 0                             # a real gap: no cell, bare substrate (dark)
    nuclei[nuclei == 6] = 0
    cad[31:59, 31:59] = 10.0
    cells[cells == 11] = 0                            # a missed cell: cytoplasm level, nucleus inside
    gaps, thr = an.dark_gaps(cells, nuclei, cad, 0.429)
    assert 10.0 < thr < 90.0
    assert gaps[35:55, 35:55].all()                   # the dark gap is found
    assert not gaps[61:89, 61:89].any()               # the missed cell is not a gap
    assert an.orphan_fraction(cells, nuclei) == pytest.approx(1 / 15)


def test_build_verdicts_stratified_precision(tmp_path, capsys):
    bv = pytest.importorskip('build_verdicts')
    rows = []
    for method, p_unmatched in (('v1', 0.0), ('v2', 1.0)):
        for stratum, n_pop, yes in (('matched', 900, [1, 1, 1, 1]), ('unmatched', 100, [p_unmatched] * 4)):
            for y in yes:
                rows.append(dict(block='cell', method=method, folder='Static-x20', stratum=stratum,
                                 stratum_n=n_pop, area_um2=400.0, truth=bool(y)))
    rows += [dict(block='gap', method='v2', folder='Static-x20', stratum='component', stratum_n=np.nan,
                  area_um2=a, truth=t) for a, t in ((300.0, True), (100.0, False))]
    key = pd.DataFrame(rows)
    key.insert(0, 'verdict_id', [f'V{i + 1:04d}' for i in range(len(key))])
    key.to_csv(tmp_path / 'key.csv', index=False)
    code = 'VS%d:%s' % (len(key), ''.join('Y' if t else 'N' for t in key.truth))
    bv.main(['score', '--session', str(tmp_path), '--code', code])
    out = capsys.readouterr().out
    assert 'v1: 0.90' in out and 'v2: 1.00' in out          # 0.9 * 1 + 0.1 * 0 and 0.9 * 1 + 0.1 * 1
    assert 'v2: 0.75 of 400 um^2' in out
