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
    shown = html.split('const ITEMS=', 1)[1].split('const KEYZ=', 1)[0]      # what the page displays
    assert html.count('data:image/jpeg;base64,') == 4 and 'v1' not in shown and 'v2' not in shown
    key = pd.read_csv(tmp_path / 'key.csv')
    assert list(key.verdict_id) == ['V0001', 'V0002', 'V0003', 'V0004']
    answers = ''.join('Y' if m == 'v2' else 'N' for m in key.method)
    t = vs.score(str(tmp_path), by=['method'], code=f'VS4:{answers}').set_index('method')
    assert t.loc['v1', 'yes_rate'] == 0 and t.loc['v2', 'yes_rate'] == 1
    t2 = vs.score(page, by=['method'], code=f'VS4:{answers}').set_index('method')   # key read from the page
    assert t2.equals(t)
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
    assert 'v2: 0.75' in out                                # uniform gap sample: 300 of 400 um^2 confirmed
    d = key.assign(yes=key.truth)
    est = {m: e for m, e, lo, hi, _ in bv.cell_precision(key, d)}
    assert 0.8 < est['v1'] < 0.9 < est['v2']                # 0.9 * p(matched) + 0.1 * p(unmatched), Jeffreys


def test_refine_v2_markers_growth_and_orphans():
    rf = pytest.importorskip('refine_v2')
    cells = np.zeros((90, 90), np.int32)
    nuclei = np.zeros_like(cells)
    cells[5:25, 5:25] = 1                                   # cut short: its territory runs to x = 44
    nuclei[10:18, 10:18] = 1
    cells[5:30, 50:80] = 2                                  # large, nucleus-free: kept
    cells[40:44, 40:44] = 3                                 # small nucleus-free fragment: dropped
    nuclei[60:68, 10:18] = 2                                # nucleus without a cell: new marker
    m, info = rf.markers(cells, nuclei)
    assert info == dict(nucleated=1, nucleus_free_kept=1, fragments_dropped=1, orphan_markers=1)
    assert (m == 3).sum() == 0 and (m > 2).sum() == 64
    tophat = np.zeros(cells.shape)
    tophat[:, 45] = 100.0                                   # a junction between the left and right cells
    tophat[35, :45] = 100.0                                 # and one below cell 1
    raw = np.full(cells.shape, 100.0)
    grown, gaps, sens, info = rf.refine(cells, nuclei, tophat, raw, nuclei * 50.0, 0.429)
    assert not gaps.any() and info['covered_out'] == 1.0
    c1 = grown[12, 12]
    assert (grown[5:30, 5:44] == c1).mean() > 0.95          # grew to the junctions
    assert grown[42, 42] != 0 and grown[64, 14] not in (0, c1)   # fragment absorbed, orphan has a cell


def test_nuclear_signal_catches_missed_dim_nucleus():
    an = pytest.importorskip('analyze')
    nuc_img = np.full((60, 60), 10.0)
    detected = np.zeros((60, 60), np.int32)
    detected[5:15, 5:15] = 1
    nuc_img[5:15, 5:15] = 200.0
    nuc_img[40:50, 40:50] = 80.0                            # dim nucleus the model missed
    sig = an.nuclear_signal(nuc_img, detected)
    assert sig[40:50, 40:50].all() and not sig[25:35, 25:35].any()
    cells = np.zeros((60, 60), np.int32)
    cells[0:30, 0:30] = 1
    cad = np.full((60, 60), 100.0)
    cad[32:58, 32:58] = 5.0                                 # dark region around the missed nucleus
    without, _ = an.dark_gaps(cells, detected, cad, 0.429)
    with_img, _ = an.dark_gaps(cells, detected, cad, 0.429, nuc_img=nuc_img)
    assert without[45, 45] and not with_img[45, 45]
    assert an.area_filter(without, 0.429, 1e6).sum() == 0


def test_round2_exclusion_uses_masks():
    b2 = pytest.importorskip('build_verdicts2')
    lab = np.zeros((100, 100), np.int32)
    lab[10:20, 10:20] = 1                                   # overlaps the round-1 zone
    lab[10:20, 40:50] = 2                                   # 6.4 um (15 px) beyond the zone edge: excluded
    lab[70:90, 70:90] = 3                                   # far away: kept
    zone = np.zeros(lab.shape, bool)
    zone[5:25, 5:25] = True                                 # a round-1 object
    zones = {KEY20: np.asarray(__import__('scipy').ndimage.distance_transform_edt(~zone)) * 0.429 <= b2.EXCLUDE_UM}
    assert b2.excluded_labels(zones, KEY20, lab) == {1, 2}
    assert b2.excluded_labels(zones, 'other', lab) == set()


def test_nucleated_split_ignores_fragments():
    an = pytest.importorskip('analyze')
    c1 = np.zeros((40, 80), np.int32)
    c1[:, :40] = 1                                          # v1 cell with two nuclei
    c1[:, 40:] = 2
    n1 = np.zeros_like(c1)
    n1[10:15, 5:10], n1[10:15, 25:30], n1[10:15, 50:55] = 1, 2, 3
    c2 = np.zeros_like(c1)
    c2[:, :20], c2[:, 20:40] = 1, 2                         # v2 splits cell 1 into two nucleated cells
    c2[:, 40:78], c2[:, 78:] = 3, 4                         # and cuts a nucleus-free sliver off cell 2
    n2 = n1.copy()
    split, merge = an.nucleated_split_merge(c1, n1, c2, n2)
    assert split == 1 and merge == 0
    g, p, inter, ga, pa = __import__('seg_eval').overlap(c1, c2)
    assert __import__('seg_eval').split_merge(inter, ga, pa)[0].sum() == 2      # the plain count includes the sliver


def test_minority_constraint_and_small_cell_tail():
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(np.log(500), 0.4, 5000), rng.normal(np.log(120), 0.4, 150)])   # small-cell tail
    f = sn.fit(x)
    assert f['frac_enlarged'] <= 0.5                        # the enlarged label never goes to the majority
    assert f['bic1'] - f['bic2_any'] > 10 > f['bic1'] - f['bic2']      # a tail is supported, an enlarged minority is not
    free = sn.fit(x, max_frac=1.0)
    assert free['frac_enlarged'] > 0.9                      # unconstrained: "enlarged" = the main population


def test_scoring_post_stratified_and_pps():
    bv = pytest.importorskip('build_verdicts')
    rows = []
    for folder, n_pop, yes in (('A', 900, [1, 1, 1, 1]), ('B', 100, [0, 0, 0, 0])):
        rows += [dict(block='cell', method='m', folder=folder, stratum='matched', stratum_n=n_pop, yes=bool(y)) for y in yes]
    rows.append(dict(block='cell', method='m', folder='C', stratum='matched', stratum_n=5000, yes=np.nan))   # no answer
    key = pd.DataFrame(rows)
    d = key.dropna(subset=['yes']).assign(yes=lambda t: t.yes.astype(bool))
    (_, est, lo, hi, used), = bv.cell_precision(key, d, by=('folder', 'stratum'))
    assert used == 2 and 0.8 < est < 0.9 and lo < est < hi              # 0.9 * p_A + 0.1 * p_B, C dropped
    gaps = pd.DataFrame([dict(block='gap', method='m', folder='A', sampling='pps', mult=3, folder_area=1e5, yes=True,
                              area_um2=1e4),
                         dict(block='gap', method='m', folder='B', sampling='pps', mult=3, folder_area=400.0, yes=False,
                              area_um2=100.0)])
    (_, est, lo, hi, n), = bv.gap_area_precision(gaps, gaps)
    assert est > 0.8 and n == 6                              # area share, not the 0.5 item share


def test_cell_precision_is_design_weighted():
    bv = pytest.importorskip('build_verdicts')
    rows = [dict(block='cell', method='m', folder='A', stratum='s', stratum_n=900, yes=y) for y in (1, 1, 0, 0)]
    rows += [dict(block='cell', method='m', folder='B', stratum='s', stratum_n=100, yes=1) for _ in range(4)]
    key = pd.DataFrame(rows)
    d = key.assign(yes=key.yes.astype(bool))
    (_, est, lo, hi, used), = bv.cell_precision(key, d)                 # truth: 0.9 * 0.5 + 0.1 * 1 = 0.55
    assert used == 2 and 0.5 < est < 0.68 and lo < 0.55 < hi             # partial pooling: mild pull to 0.75
    (_, jeff, _, _, _), = bv.cell_precision(key, d, prior='jeffreys')
    assert jeff == pytest.approx(0.55, abs=0.03)
    (_, pooled, _, _, _), = bv.cell_precision(key, d, by=('stratum',))  # equal-allocation pooling is biased
    assert pooled > est + 0.1
