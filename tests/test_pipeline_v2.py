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
    assert used == 2 and 0.75 < est < 0.9 and lo < est < hi             # 0.9 * p_A + 0.1 * p_B (median ~0.80), C dropped
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


def _junction_field(rng, blur_px=0.0, shape=(200, 200), step=40):
    """Bright 2-px lines on a grid (junctions) over a dim background, plus noise."""
    from scipy import ndimage
    img = np.full(shape, 100.0)
    img[::step, :] = img[1::step, :] = 400.0
    img[:, ::step] = img[:, 1::step] = 400.0
    if blur_px:
        img = ndimage.gaussian_filter(img, blur_px)
    return img + rng.normal(0, 10, shape)


def test_quality_ridge_snr_drops_with_blur_and_z_flags_outlier():
    qm = pytest.importorskip('quality')
    rng = np.random.default_rng(3)
    sharp = qm.ridge_snr(_junction_field(rng), 0.429)
    blurred = qm.ridge_snr(_junction_field(rng, blur_px=4.0), 0.429)
    assert blurred < 0.5 * sharp
    assert qm.noise_sigma(rng.normal(0, 10, (300, 300))) == pytest.approx(10, rel=0.05)
    z = qm.robust_z([10, 11, 9, 10.5, 9.5, 10.2, 3.0])
    assert z[-1] < -3 and (np.abs(z[:-1]) < 2).all()


def test_quality_nuclear_contrast_drops_with_blur():
    qm = pytest.importorskip('quality')
    from scipy import ndimage
    yy, xx = np.mgrid[:120, :120]
    nuclei = np.zeros((120, 120), np.int32)
    for i, (cy, cx) in enumerate(((30, 30), (30, 90), (90, 30), (90, 90)), 1):
        nuclei[(yy - cy) ** 2 + (xx - cx) ** 2 <= 12 ** 2] = i
    img = np.where(nuclei > 0, 1000.0, 100.0)
    sharp = qm.nuclear_contrast(img, nuclei, 0.429)
    hazy = qm.nuclear_contrast(ndimage.gaussian_filter(img, 5.0), nuclei, 0.429)
    assert sharp == pytest.approx(9.0, rel=0.01) and hazy < 0.6 * sharp


def test_quality_repeated_fields_and_assess():
    qm = pytest.importorskip('quality')
    rng = np.random.default_rng(5)
    from scipy import ndimage
    base = ndimage.gaussian_filter(rng.random((300, 300)), 3)
    a = base[:256, :256]
    b = base[40:296, 20:276] + rng.normal(0, 0.002, (256, 256))      # same area, shifted: overlap ~0.75
    c = ndimage.gaussian_filter(rng.random((256, 256)), 3)           # a different field
    pairs = qm.repeated_fields({'f1': a, 'f2': b, 'f3': c}, size=256)
    assert [(p[0], p[1]) for p in pairs] == [('f1', 'f2')] and pairs[0][2] > 0.7 and pairs[0][3] > 0.9
    rows = [dict(key=f'k{i}', ridge_snr=s, nuclear_contrast=5.0) for i, s in enumerate([10, 11, 9, 10, 10.5, 2.0])]
    q = qm.assess(rows, [('k0', 'k1', 1.0, 0.99)]).set_index('key')
    assert q.low_quality.tolist() == [False] * 5 + [True]
    assert q.duplicate_of['k0'] == 'k1' and q.duplicate_of['k1'] == ''     # the lower score is dropped
    assert q.exclude.tolist() == [True, False, False, False, False, True]


def test_scoring_restrict_drops_fields_and_rescales_population():
    bv = pytest.importorskip('build_verdicts')
    key = pd.DataFrame([dict(verdict_id='V1', block='cell', method='m', folder='A', stratum='s', stratum_n=100, key='a1'),
                        dict(verdict_id='V2', block='cell', method='m', folder='A', stratum='s', stratum_n=100, key='a2'),
                        dict(verdict_id='V3', block='gap', method='m', folder='A', stratum='component', key='a2')])
    d = key.assign(yes=True)
    strata = pd.DataFrame([dict(key='a1', folder='A', method='m', stratum='s', n=75),
                           dict(key='a2', folder='A', method='m', stratum='s', n=25)])
    k2, d2 = bv.restrict(key, d, {'a2'}, strata)
    assert d2.verdict_id.tolist() == ['V1']
    assert k2.loc[k2.block == 'cell', 'stratum_n'].tolist() == [75.0, 75.0]


def test_grow_gaps_extends_seeds_and_fills_specks():
    an = pytest.importorskip('analyze')
    seeds = np.zeros((60, 60), bool)
    seeds[20:26, 20:26] = True
    cand = np.zeros_like(seeds)
    cand[18:40, 18:40] = True                     # the dark region around the seed
    cand[29, 29] = False                          # a 1-px speck inside it
    cand[50:55, 50:55] = True                     # dark but never reaches the seed threshold
    nuclear = np.zeros_like(seeds)
    g = an.grow_gaps(seeds, cand, nuclear, 0.429)
    assert g[18:40, 18:40].all() and not g[50:55, 50:55].any()
    nuclear[29, 29] = True                        # a speck with nuclear signal stays open
    assert not an.grow_gaps(seeds, cand, nuclear, 0.429)[29, 29]


def test_render_keeps_full_crop_at_border():
    bv = pytest.importorskip('build_verdicts')
    rgb = np.random.default_rng(0).random((200, 200, 3))
    mask = np.zeros((200, 200), bool)
    mask[0:6, 90:100] = True                      # object on the top edge
    im = np.asarray(bv.render(rgb, mask, 0.429, size=100))
    left = im[:, :100]
    assert (left.reshape(-1, 3) != 24).any(axis=1).mean() > 0.95      # no padding band in the crop


def test_export_flags_objective_that_does_not_match_the_file_name(tmp_path, monkeypatch, capsys):
    import types
    ex = pytest.importorskip('export_from_drive')

    class _F:
        def __init__(self, path):
            self.path = path
            self.sizes = {'Z': 13, 'C': 3, 'Y': 1024, 'X': 1024}
            mic = types.SimpleNamespace(objectiveName='Plan Apo 20x DIC M N2', objectiveMagnification=20.0,
                                        objectiveNumericalAperture=0.75)
            self.metadata = types.SimpleNamespace(channels=[types.SimpleNamespace(microscope=mic)])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def voxel_size(self):
            return types.SimpleNamespace(x=0.429, y=0.429, z=0.7)

    monkeypatch.setitem(sys.modules, 'nd2', types.SimpleNamespace(ND2File=_F))
    d = tmp_path / 'Renamed Data' / 'A1'
    d.mkdir(parents=True)
    for name in ('1.4Pa_A1_20dec21_40x_L2RA_FlatA_seq006.nd2', '1.4Pa_A1_20dec21_20xA_L2RA_FlatA_seq015.nd2'):
        (d / name).write_bytes(b'')
    found, have = ex.pixel_sizes(str(tmp_path))
    assert have and 'does not match' in found['A1_40x']['note'] and 'note' not in found['A1_20x']
    assert 'does not match' in capsys.readouterr().err


def test_export_nd2_finds_the_nikon_dapi_channel():
    """A1's files name DAPI 'WF 395' and other Nikon files '395 Confocal'; neither says DAPI."""
    en = pytest.importorskip('export_nd2')
    assert all(en.DAPI_RE.search(n) for n in ('WF 395', '395 Confocal', 'DAPI', 'Hoechst 405'))
    assert not any(en.DAPI_RE.search(n) for n in ('WF 470', 'WF 555', '555 Confocal', 'Mono'))


def test_export_nd2_metadata_planes_and_dapi_sum(tmp_path, monkeypatch):
    """Fake .nd2 files: A1 only, corrected 40x pixel, sharpest plane, DAPI sum, zips, --check."""
    import types
    import zipfile
    import tifffile
    en = pytest.importorskip('export_nd2')
    ft_px = ft.PIXEL_UM
    assert en.CORRECT_PX_UM == {'20x': ft_px['20x'], '40x': ft_px['40x']}
    rng = np.random.default_rng(5)
    sharp = rng.integers(100, 4000, (3, 64, 64)).astype(np.uint16)          # (C, Y, X) in focus
    from scipy import ndimage as ndi
    stack = np.stack([sharp if z == 3 else ndi.gaussian_filter(sharp.astype(float), (0, 3 + abs(z - 3), 3 + abs(z - 3))).astype(np.uint16)
                      for z in range(6)])                                    # (Z, C, Y, X), plane 3 sharp

    class _F:
        def __init__(self, path):
            self.path = path
            self.sizes = {'Z': 6, 'C': 3, 'Y': 64, 'X': 64}
            self.dtype = np.uint16
            mic = types.SimpleNamespace(objectiveName='Plan Apo 20x', objectiveMagnification=20.0,
                                        objectiveNumericalAperture=0.75)
            self.metadata = types.SimpleNamespace(channels=[
                types.SimpleNamespace(microscope=mic, channel=types.SimpleNamespace(name=n))
                for n in ('Cy5', 'DAPI', 'GFP')])
            self.text_info = {'capturing': 'Andor iXon 888\r\nExposure: 200 ms'}
            self.loop_indices = tuple({'Z': z} for z in range(6))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def voxel_size(self):
            return types.SimpleNamespace(x=0.429, y=0.429, z=0.7)

        def frame_metadata(self, i):
            pos = types.SimpleNamespace(stagePositionUm=(1000.0, -250.0, 3000.0 + 0.7 * i))
            t = types.SimpleNamespace(absoluteJulianDayNumber=2459568.9)      # 20 Dec 2021, 09:36 UTC
            return types.SimpleNamespace(channels=[types.SimpleNamespace(position=pos, time=t)])

        def asarray(self):
            return stack

    monkeypatch.setitem(sys.modules, 'nd2', types.SimpleNamespace(ND2File=_F))
    d = tmp_path / 'Renamed Data' / 'A1'
    d.mkdir(parents=True)
    for name in ('1.4Pa_A1_20dec21_40x_L2RA_FlatA_seq006.nd2', '0Pa_A1_19dec21_20xA_L2RA_FlatA_seq001.nd2',
                 '0Pa_U_19dec21_20xA_L2RA_FlatA_seq001.nd2'):                  # the last is flow3: skipped
        (d / name).write_bytes(b'')
    rows, _, _ = en.export(str(tmp_path), str(tmp_path / 'out'), check=True)
    assert [r['key'] for r in rows] == ['0Pa_A1_19dec21_20xA_L2RA_FlatA_seq001', '1.4Pa_A1_20dec21_40x_L2RA_FlatA_seq006']
    assert not (tmp_path / 'out').exists()                                    # --check writes nothing
    r20, r40 = rows
    assert r20['px_um'] == 0.429 and r20['px_note'] == ''
    assert r40['px_um'] == 0.2145 and 'records 20x' in r40['px_note']
    assert r40['exposure_ms'] == 200.0 and r40['dapi_channel'] == 1 and r40['stage_x_um'] == 1000.0
    assert abs(r40['z_last_um'] - r40['z_first_um'] - 3.5) < 1e-9 and r40['acquired'] == '2021-12-20T09:36:00'

    rows, planes, stacks = en.export(str(tmp_path), str(tmp_path / 'out'), split_mb=0.05, stacks=True)
    assert all(r['best_z'] == '3|3|3' for r in rows) and len(planes) > 1   # split at 0.05 MB
    names = {n: z for z in planes for n in zipfile.ZipFile(z).namelist()}
    member = 'nd2_planes/1.4Pa_A1_40x/1.4Pa_A1_20dec21_40x_L2RA_FlatA_seq006_dapi_sum.ome.tif'
    assert member in names and 'nd2_planes/nd2_metadata.csv' in names
    with zipfile.ZipFile(names[member]) as z:
        z.extract(member, tmp_path / 'x')
    with tifffile.TiffFile(tmp_path / 'x' / member) as tf:
        np.testing.assert_array_equal(tf.asarray(), stack[:, 1].astype(np.uint32).sum(0))
        assert 'PhysicalSizeX="0.2145"' in tf.ome_metadata
    focus = pd.read_csv(tmp_path / 'out' / 'nd2_focus.csv')
    assert len(focus) == 2 * 3 * 6 and set(focus[focus.best == 1].z) == {3}
    assert any('nd2_stacks/0Pa_A1_20x/' in n for z in stacks for n in zipfile.ZipFile(z).namelist())

def test_golgi_polarity_direction_border_cells_and_offset():
    po = pytest.importorskip('polarity')
    rng = np.random.default_rng(2)
    cells = np.zeros((200, 200), np.int32)
    nuclei = np.zeros_like(cells)
    golgi = rng.uniform(1, 5, cells.shape)
    yy, xx = np.mgrid[:200, :200]
    lab = 0
    for i in range(5):
        for j in range(5):                        # 5 x 5 cells of 40 px; the outer ring touches the border
            lab += 1
            cy, cx = 40 * i + 20, 40 * j + 20
            cells[40 * i:40 * i + 40, 40 * j:40 * j + 40] = lab
            nuclei[(yy - cy) ** 2 + (xx - cx) ** 2 <= 6 ** 2] = lab
            golgi[(yy - cy) ** 2 + (xx - cx + 10) ** 2 <= 3 ** 2] = 100.0     # 10 px left of the nucleus
    v = po.golgi_vectors(cells, nuclei, golgi, 0.5)
    assert len(v) == 9                            # interior cells only
    assert np.allclose(v.dx_um, -5.0, atol=0.3) and np.allclose(v.dy_um, 0.0, atol=0.3)
    n, r, ang, p = po.mean_direction(v.dx_um, v.dy_um)
    assert r > 0.99 and abs(ang - 180) < 2 and p < 1e-3
    n, r, ang, p = po.mean_direction(*np.split(rng.normal(0, 1, 800), 2))    # no preferred side
    assert r < 0.15 and p > 0.01
    # a channel offset seen on the static slide is removed before the flow slide is summarised
    phi = rng.uniform(0, 2 * np.pi, 600)
    static = pd.DataFrame(dict(dx_um=4 * np.cos(phi) - 1.0, dy_um=4 * np.sin(phi)))
    flow = pd.DataFrame(dict(dx_um=4 * np.cos(phi[:300]) - 1.0, dy_um=4 * np.sin(phi[:300])))
    flow.loc[:149, 'dx_um'] = -4.0 - 1.0
    flow.loc[:149, 'dy_um'] = 0.0
    cells_df = pd.concat([static.assign(condition='0Pa_A1_20x'), flow.assign(condition='1.4Pa_A1_20x')])
    s = po.summarise(cells_df.assign(date='19dec21')).set_index('condition')
    assert s.loc['0Pa_A1_20x', 'R'] > 0.1 and s.loc['0Pa_A1_20x', 'R_corrected'] < 0.5 * s.loc['0Pa_A1_20x', 'R']
    assert s.loc['1.4Pa_A1_20x', 'R_corrected'] > 0.4
    assert abs(s.loc['1.4Pa_A1_20x', 'direction_corrected_deg'] - 180) < 10


def test_nuclear_mixture_recovers_fraction_and_antimode():
    nm = pytest.importorskip('nuclear_mixture')
    bs = pytest.importorskip('by_shear')
    rng = np.random.default_rng(4)
    x = np.log(np.concatenate([rng.lognormal(np.log(67), 0.2, 2100), rng.lognormal(np.log(125), 0.2, 900)]))
    f = nm.fit(x)
    assert abs(f['frac_enlarged'] - 0.30) < 0.03 and abs(f['ratio'] - 125 / 67) < 0.08
    assert f['bic1'] - f['bic2'] > 50
    t = bs.antimode(f['params'])
    assert 67 < t < 125
    one = nm.fit(np.log(rng.lognormal(np.log(67), 0.2, 3000)))
    assert one['bic1'] - one['bic2'] < 10                       # one population: no support for two


def test_by_shear_pools_folders_by_shear_stress():
    bs = pytest.importorskip('by_shear')
    df = pd.DataFrame(dict(condition=['0Pa_A1_20x', '0Pa_A1_40x', '1.4Pa_A1_20x', '1.4Pa_A1_40x'],
                           key=['0Pa_A1_19dec21_20xA_x_seq001', '0Pa_A1_19dec21_40x_x_seq001',
                                '1.4Pa_A1_20dec21_20xA_x_seq001', '1.4Pa_A1_20dec21_40x_x_seq001']))
    s, f = bs.Grouping('shear'), bs.Grouping('folder')
    assert list(s.label(df).grp) == ['Static', 'Static', '1.4 Pa', '1.4 Pa']
    assert list(f.label(df).grp) == list(df.condition)
    assert list(s.label(df).date) == ['19dec21', '19dec21', '20dec21', '20dec21']
    assert s.static_of('1.4 Pa') == 'Static' and f.static_of('1.4Pa_A1_40x') == '0Pa_A1_40x'


def test_gap_contact_null_detects_gaps_placed_next_to_enlarged_cells():
    gc = pytest.importorskip('gap_contact')
    cells = np.zeros((120, 120), np.int32)
    lab = 0
    for i in range(6):
        for j in range(6):
            lab += 1
            cells[20 * i:20 * i + 20, 20 * j:20 * j + 20] = lab
    labels = np.arange(1, lab + 1)
    enlarged = np.isin(labels, np.random.default_rng(1).choice(labels, 6, replace=False))
    gaps = np.zeros(cells.shape, bool)
    for l in labels[enlarged]:                                   # a small gap inside each enlarged cell
        yy, xx = np.nonzero(cells == l)
        cy, cx = int(yy.mean()), int(xx.mean())
        gaps[cy - 2:cy + 2, cx - 2:cx + 2] = True
    obs, null, ne, nn = gc.field_null(cells, gaps, labels, enlarged, 0.5, 200, np.random.default_rng(0))
    assert ne == 6 and nn == 30 and obs == (6, 0)
    share_null = null[:, 0] / np.maximum(null.sum(1), 1)
    assert (np.sum(share_null >= 1.0) + 1) / 201 < 0.05          # the observed share (1.0) is rare by geometry
    assert np.median(share_null) < 0.4


def test_nd2_dna_index_puts_doubled_dna_at_4n():
    nl = pytest.importorskip('nd2_link')
    rng = np.random.default_rng(5)
    area = np.r_[rng.normal(65, 5, 200), rng.normal(130, 10, 60)]
    dna = np.r_[rng.normal(1.0, 0.08, 200), rng.normal(2.0, 0.12, 60)] * 5e4
    d = pd.DataFrame(dict(key='k', label=np.arange(260), area_um2=area, dna_raw=dna))
    out = nl.dna_index(d)
    assert abs(out[~out.enlarged].dna.median() - 1.0) < 0.05
    assert (out[out.enlarged].cls == '4N').mean() > 0.9 and (out[~out.enlarged].cls == '2N').mean() > 0.9
    # the summed-DAPI step: a nucleus twice as bright integrates to twice the signal
    nuclei = np.zeros((60, 60), np.int32)
    yy, xx = np.mgrid[:60, :60]
    nuclei[(yy - 15) ** 2 + (xx - 15) ** 2 <= 36] = 1
    nuclei[(yy - 45) ** 2 + (xx - 45) ** 2 <= 36] = 2
    stack = np.full((5, 60, 60), 10.0)
    stack[:, nuclei == 1] += 100
    stack[:, nuclei == 2] += 200
    s = nl.dna_sums(stack, nuclei, 1.0).set_index('label')
    assert abs(s.dna_raw[2] / s.dna_raw[1] - 2.0) < 0.05


def test_cell_features_neighbours_on_a_grid():
    cf = pytest.importorskip('cell_features')
    cells = np.zeros((30, 30), np.int32)
    lab = 0
    for i in range(3):
        for j in range(3):
            lab += 1
            cells[10 * i:10 * i + 10, 10 * j:10 * j + 10] = lab
    nb = cf.neighbours(cells)
    assert nb[5] == 8 and nb[1] == 3 and nb[2] == 5              # centre, corner, edge (8-connectivity)


def test_round3_scores_gap_area_by_multiplicity_and_outlines_touching_nuclei():
    import build_verdicts3 as bv3
    key = pd.DataFrame([
        dict(verdict_id='V0001', block='gap', shear='Static', mult=3, shear_area=100.0, stratum_n=np.nan),
        dict(verdict_id='V0002', block='gap', shear='Static', mult=1, shear_area=100.0, stratum_n=np.nan),
        dict(verdict_id='V0003', block='gap', shear='1.4 Pa', mult=2, shear_area=300.0, stratum_n=np.nan),
        dict(verdict_id='V0004', block='multinucleated', shear='Static', mult=np.nan, shear_area=np.nan, stratum_n=10),
        dict(verdict_id='V0005', block='multinucleated', shear='1.4 Pa', mult=np.nan, shear_area=np.nan, stratum_n=30),
        dict(verdict_id='V0006', block='multinucleated', shear='1.4 Pa', mult=np.nan, shear_area=np.nan, stratum_n=30)])
    ans = pd.DataFrame({'id': key.verdict_id, 'answer': ['yes', 'no', 'yes', 'yes', 'no', 'unsure']})
    t = bv3.score_table(key, ans, n_draw=20_000).set_index(['block', 'shear'])
    # static: 3 of 4 draws yes; flow: 2 of 2; pooled by area 1:3
    assert abs(t.loc[('gap', 'Static'), 'n'] - 4) < 1e-9 and t.loc[('gap', '1.4 Pa'), 'n'] == 2
    assert 0.6 < t.loc[('gap', 'Static'), 'estimate'] < 0.8
    assert t.loc[('gap', '1.4 Pa'), 'estimate'] > t.loc[('gap', 'Static'), 'estimate']
    assert t.loc[('multinucleated', 'Static'), 'estimate'] == 1.0
    assert t.loc[('multinucleated', '1.4 Pa'), 'n'] == 1              # the unsure answer is left out
    assert abs(t.loc[('multinucleated', 'pooled by count'), 'estimate'] - 0.25) < 1e-9
    # two touching nuclei keep the boundary between them
    rgb = np.zeros((80, 80, 3))
    cell = np.zeros((80, 80), bool)
    cell[20:60, 20:60] = True
    nuc = np.zeros((80, 80), np.int32)
    nuc[30:50, 25:40] = 1
    nuc[30:50, 40:55] = 2
    im = np.array(bv3.render_multi(rgb, cell, nuc, 0.429, size=160))
    right = im[:, 168:]
    red = (right[..., 0] == 255) & (right[..., 1] == 40)
    ys, xs = np.nonzero(red)
    mid = (xs.min() + xs.max()) // 2
    assert red[(ys.min() + ys.max()) // 2, mid - 3:mid + 4].any()   # a red line between the nuclei


def test_nucleus_rule_removes_gaps_around_a_nucleus_and_fills_small_voids():
    import analyze as an
    um = 0.429
    gaps = np.zeros((120, 120), bool)
    nuclei = np.zeros((120, 120), np.int32)
    gaps[5:45, 5:45] = True
    gaps[15:35, 15:35] = False                    # ring around a nucleus: a faint cell
    nuclei[18:32, 18:32] = 1
    gaps[60:100, 5:45] = True
    gaps[78:84, 22:28] = False                    # 36 px = 6.6 um^2 void, no nucleus: filled
    gaps[60:110, 60:110] = True
    gaps[70:100, 70:100] = False                  # 900 px = 166 um^2 void, no nucleus: kept
    out = an.nucleus_rule(gaps, nuclei, nuclei > 0, um)
    assert not out[5:45, 5:45].any()              # the ring is gone
    assert out[78:84, 22:28].all()                # the small void is filled
    assert not out[70:100, 70:100].any() and out[60:70, 60:110].all()
    # a gap that only touches a nucleus from outside is kept
    g2 = np.zeros((60, 60), bool)
    g2[10:30, 10:30] = True
    n2 = np.zeros((60, 60), np.int32)
    n2[30:40, 10:20] = 1
    assert (an.nucleus_rule(g2, n2, n2 > 0, um) == g2).all()
