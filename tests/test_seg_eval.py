"""Synthetic checks for Imaging/Validation/seg_eval.py (known answers)."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('skimage')
tifffile = pytest.importorskip('tifffile')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'Imaging', 'Validation'))
import seg_eval as se  # noqa: E402
from skimage.draw import ellipse  # noqa: E402


def grid(n=4, size=20, gap=0, shape=(100, 100)):
    """n x n squares of side `size` separated by `gap` px, labelled 1..n*n."""
    lab = np.zeros(shape, np.int32)
    k = 0
    for i in range(n):
        for j in range(n):
            k += 1
            y, x = 5 + i * (size + gap), 5 + j * (size + gap)
            lab[y:y + size, x:x + size] = k
    return lab


def test_sample_key_and_condition():
    names = {
        'denoised_0Pa_A1_19dec21_20xA_L2RA_FlatA_seq001_Cadherins_regional_segmented.tif':
            ('0Pa_A1_19dec21_20xA_L2RA_FlatA_seq001', '0Pa_A1_20x'),
        '1.4Pa_A1_20dec21_40x_L2RA_FlatA_seq002_cell_mask_merged_conservative.tif':
            ('1.4Pa_A1_20dec21_40x_L2RA_FlatA_seq002', '1.4Pa_A1_40x'),
        'denoised_0Pa_U_05mar19_20x_L2RA_Flat_seq008_Nuclei_regional_tophat_mask.tif':
            ('0Pa_U_05mar19_20x_L2RA_Flat_seq008', '0Pa_U_20x'),
        '8Pa-1.4Pa_U_03apr19_40x_R2L_Flat_seq004.nd2':
            ('8Pa-1.4Pa_U_03apr19_40x_R2L_Flat_seq004', '8Pa-1.4Pa_U_40x'),
    }
    for name, (key, cond) in names.items():
        assert se.sample_key(name) == key
        assert se.condition_of(key) == cond
    assert se.sample_key('notes.txt') is None


def test_px_um_lookup():
    assert se.parse_px_um(None) is None
    assert se.parse_px_um('0.65')('0Pa_A1_19dec21_20xA_L2RA_FlatA_seq001') == 0.65
    f = se.parse_px_um('20x=0.65,40x=0.325')
    assert f('1.4Pa_A1_20dec21_40x_L2RA_FlatA_seq002') == 0.325
    with pytest.raises(KeyError):
        se.parse_px_um('20x=0.65')('1.4Pa_A1_20dec21_40x_L2RA_FlatA_seq002')


def test_perfect_match_with_permuted_labels():
    gt = grid()
    perm = np.concatenate([[0], np.random.default_rng(0).permutation(np.arange(1, 17)) + 100])
    s, pairs = se.evaluate_cells(gt, perm[gt])
    assert s['f1@50'] == 1 and s['f1@75'] == 1 and s['ap50_95'] == 1
    assert s['n_split_gt'] == 0 and s['n_merge_pred'] == 0
    assert len(pairs) == 16 and np.allclose(pairs.iou, 1)
    assert s['gt_sum_area'] == s['pred_sum_area']


def test_merge_of_three_cells():
    gt = grid()
    pred = gt.copy()
    pred[np.isin(gt, [2, 3])] = 1          # cells 1-3 (first row) become one object
    s, _ = se.evaluate_cells(gt, pred)
    assert s['n_merge_pred'] == 1 and s['n_gt_in_merges'] == 3
    assert (s['tp@50'], s['fp@50'], s['fn@50']) == (13, 1, 3)
    assert s['pred_n_interior'] == 14 and s['gt_n_interior'] == 16


def test_split_into_three():
    gt = grid()
    pred = gt.copy()
    y, x = np.nonzero(gt == 6)
    pred[y[(x - x.min()) >= 7], x[(x - x.min()) >= 7]] = 50
    pred[y[(x - x.min()) >= 14], x[(x - x.min()) >= 14]] = 51
    s, _ = se.evaluate_cells(gt, pred)
    assert s['n_split_gt'] == 1 and s['n_merge_pred'] == 0
    assert (s['tp@50'], s['fp@50'], s['fn@50']) == (15, 3, 1)


def test_boundary_shift_lowers_high_iou_thresholds_only():
    gt = grid(gap=4)                         # background between cells so growth is possible
    from skimage.segmentation import expand_labels
    pred = expand_labels(gt, 1)              # 20x20 -> 22x22: IoU = 400/484 = 0.826
    s, pairs = se.evaluate_cells(gt, pred)
    assert s['f1@50'] == 1 and s['f1@75'] == 1
    assert np.allclose(pairs.iou, 400 / 484, atol=0.01)
    assert 0.3 < s['ap50_95'] < 0.8
    assert s['pred_sum_area'] / s['gt_sum_area'] == pytest.approx(484 / 400, rel=0.02)


def test_roi_scoping_counts_only_in_scope_false_positives():
    full = grid()
    roi = (0, 0, 50, 50)                     # contains the centroids of cells 1, 2, 5, 6
    gt = np.where(np.isin(full, [1, 2, 5, 6]), full, 0)
    pred = full.copy()
    pred[(pred == 16)] = 0                   # a miss outside the ROI must not matter
    s, _ = se.evaluate_cells(gt, pred, roi=roi)
    assert (s['n_gt'], s['n_pred']) == (4, 4)
    assert (s['tp@50'], s['fp@50'], s['fn@50']) == (4, 0, 0)
    pred2 = pred.copy()
    pred2[2:4, 2:4] = 99                     # spurious object inside the ROI
    s2, _ = se.evaluate_cells(gt, pred2, roi=roi)
    assert s2['fp@50'] == 1


def _ellipse(shape, cy, cx, rot_deg, label=1, a=6, b=18):
    img = np.zeros(shape, np.int32)
    rr, cc = ellipse(cy, cx, a, b, shape=shape, rotation=math.radians(rot_deg))
    img[rr, cc] = label
    return img


def test_orientation_axial_angles_and_wraparound():
    horiz = se.morphology(_ellipse((80, 80), 40, 40, 0)).iloc[0]
    vert = se.morphology(_ellipse((80, 80), 40, 40, 90)).iloc[0]
    assert horiz.misalign_deg == pytest.approx(0, abs=2)
    assert vert.misalign_deg == pytest.approx(90, abs=2)
    assert horiz.aspect_ratio == pytest.approx(3, rel=0.1)
    plus = se.morphology(_ellipse((80, 80), 40, 40, 20)).iloc[0]
    minus = se.morphology(_ellipse((80, 80), 40, 40, -20)).iloc[0]
    assert plus.misalign_deg == pytest.approx(20, abs=2) and minus.misalign_deg == pytest.approx(20, abs=2)
    assert se.axial_diff(plus.axial_deg, minus.axial_deg) == pytest.approx(40, abs=3)
    assert se.axial_diff(179, 1) == pytest.approx(2)


def test_multinucleation_counts_from_points_and_masks():
    gt = grid()
    pred = gt.copy()
    pts = np.array([[15, 15], [10, 10], [15, 35], [35, 15]])     # two clicks in cell 1
    nuc = np.zeros_like(gt)
    nuc[8:12, 8:12], nuc[18:22, 18:22], nuc[13:17, 33:37] = 1, 2, 3  # two nuclei in cell 1
    s, pairs = se.evaluate_cells(gt, pred, gt_points=pts, pred_nuclei=nuc)
    assert s['gt_n_multinucleated'] == 1 and s['pred_n_multinucleated'] == 1
    assert s['gt_n_no_nucleus'] == 13
    assert pairs.set_index('gt_label').loc[1, 'gt_n_nuclei'] == 2


def test_holes_metrics():
    gt = np.zeros((100, 100), bool)
    gt[10:30, 10:30] = True                  # 400 px gap
    pred = np.zeros_like(gt)
    pred[10:30, 20:40] = True                # half overlapping
    pred[80, 80] = True                      # 1-px speck
    s = se.evaluate_holes(gt, pred)
    assert s['gt_area_frac'] == pytest.approx(0.04)
    assert s['pred_area_frac'] == pytest.approx(0.0401)
    assert s['dice'] == pytest.approx(2 * 200 / (400 + 401))
    assert (s['obj_tp'], s['obj_fp'], s['obj_fn']) == (1, 1, 0)   # IoU 1/3 >= 0.3
    f = se.evaluate_holes(gt, pred, min_area_px=5)
    assert (f['obj_tp'], f['obj_fp']) == (1, 0) and f['pred_area_frac'] == pytest.approx(0.04)
    e = se.evaluate_holes(np.zeros((10, 10)), np.zeros((10, 10)))
    assert e['dice'] == 1 and not e['gt_present'] and not e['pred_present']
    r = se.evaluate_holes(gt, pred, roi=(0, 0, 50, 50))
    assert r['n_px'] == 2500 and r['gt_area_frac'] == pytest.approx(400 / 2500)


def test_nuclei_points():
    pred = np.zeros((60, 60), np.int32)
    pred[5:15, 5:15], pred[25:35, 5:15], pred[45:55, 45:55] = 1, 2, 3
    pts = [[8, 8], [12, 12],                 # two clicks in nucleus 1 (two nuclei merged)
           [24 - 2, 10],                     # 2 px above nucleus 2: within tolerance
           [40, 30]]                         # nothing there: miss
    s = se.evaluate_nuclei_points(pts, pred, tol_px=3)
    assert (s['tp'], s['fp'], s['fn']) == (2, 1, 2)
    assert s['n_pred_with_2plus_clicks'] == 1
    s0 = se.evaluate_nuclei_points(pts, pred, tol_px=0)
    assert s0['tp'] == 1


def test_voronoi_baseline():
    seeds = np.zeros((20, 40), np.int32)
    seeds[10, 5], seeds[10, 35] = 1, 2
    v = se.voronoi_from_seeds(seeds)
    assert (v[:, :19] == 1).all() and (v[:, 21:] == 2).all()
    mask = np.ones_like(seeds, bool)
    mask[:, 30:] = False
    assert (se.voronoi_from_seeds(seeds, mask)[:, 30:] == 0).all()


def test_cluster_bootstrap_interval():
    rows = pd.DataFrame({'x': np.r_[np.ones(10), np.zeros(10)]})
    est, lo, hi = se.cluster_bootstrap(rows, lambda d: d.x.mean(), n_boot=500)
    assert est == 0.5 and lo < 0.5 < hi


def test_cli_end_to_end(tmp_path):
    key = '1.4Pa_A1_20dec21_20xA_L2RA_FlatA_seq003'
    gt_dir, pred_dir, out = tmp_path / 'gt', tmp_path / 'pred', tmp_path / 'out'
    gt_dir.mkdir()
    pred_dir.mkdir()
    gt = grid()
    pred = gt.copy()
    pred[np.isin(gt, [2, 3])] = 1
    tifffile.imwrite(gt_dir / f'{key}__r00_cells.tif', gt.astype(np.uint16))
    tifffile.imwrite(pred_dir / f'{key}_cell_mask_merged_conservative.tif', pred.astype(np.uint32))
    tifffile.imwrite(pred_dir / f'{key}_cell_mask_other_variant.tif', pred.astype(np.uint32))
    man = tmp_path / 'manifest.csv'
    pd.DataFrame([dict(roi_id=f'{key}__r00', task='cells', key=key, y0=0, x0=0, h=100, w=100),
                  dict(roi_id=f'{key}__r0', task='cells', key=key, y0=0, x0=0, h=10, w=10)]).to_csv(man, index=False)
    with pytest.raises(SystemExit):          # two candidate predictions: refuse to guess
        se.main(['cells', '--gt', str(gt_dir), '--pred', str(pred_dir), '--manifest', str(man),
                 '--out', str(out), '--n-boot', '50'])
    se.main(['cells', '--gt', str(gt_dir), '--pred', str(pred_dir), '--pred-glob', '*merged_conservative.tif',
             '--manifest', str(man), '--px-um', '0.65', '--out', str(out), '--n-boot', '50'])
    per = pd.read_csv(out / 'cells_per_item.csv')
    assert len(per) == 1 and per.roi_id[0] == f'{key}__r00'   # '__r0' must not claim '__r00_cells.tif'
    assert per['n_gt_in_merges'][0] == 3
    text = (out / 'cells_summary.md').read_text()
    assert '1.4Pa_A1_20x' in text and 'um^2' in text


def test_leaderboard_ranks_methods(tmp_path):
    key = '0Pa_A1_19dec21_20xA_L2RA_FlatA_seq001'
    gt = grid()
    merged = gt.copy()
    merged[np.isin(gt, [2, 3])] = 1
    for d in ('gt', 'good', 'bad'):
        (tmp_path / d).mkdir()
    tifffile.imwrite(tmp_path / 'gt' / f'{key}_cells.tif', gt.astype(np.uint16))
    tifffile.imwrite(tmp_path / 'good' / f'{key}_mask.tif', gt.astype(np.uint16))
    tifffile.imwrite(tmp_path / 'bad' / f'{key}_mask.tif', merged.astype(np.uint16))
    out = tmp_path / 'out'
    se.main(['cells', '--gt', str(tmp_path / 'gt'), '--method', f"good={tmp_path / 'good'}",
             '--method', f"bad={tmp_path / 'bad'}::*_mask.tif", '--out', str(out), '--n-boot', '20'])
    board = (out / 'cells_leaderboard.md').read_text()
    good = next(l for l in board.splitlines() if l.startswith('| good'))
    bad = next(l for l in board.splitlines() if l.startswith('| bad'))
    assert good.split('|')[3].strip().startswith('1.000')
    assert float(bad.split('|')[3].split()[0]) < 1
    assert (out / 'good' / 'cells_summary.md').exists() and (out / 'bad' / 'cells_per_item.csv').exists()


def test_diagnostics_flags_gap_enrichment(tmp_path):
    import diagnostics as dg
    key = '0Pa_A1_19dec21_20xA_L2RA_FlatA_seq001'
    cells = grid(gap=2)
    holes = np.zeros_like(cells)
    holes[5:27, 50:70] = 1                  # gap next to cells 3 and 4 (first row, right)
    cells[holes > 0] = 0
    nuclei = np.zeros_like(cells)
    for i, (y, x) in enumerate([(10, 10), (14, 14)]):   # cell 1 gets two nuclei
        nuclei[y:y + 2, x:x + 2] = i + 1
    d, m = dg.image_diagnostics(cells, holes, nuclei)
    assert d['cells_on_holes_frac'] == 0 and 0 < d['frac_hole_adjacent'] < 1
    assert d['frac_2plus_nuclei'] == pytest.approx(1 / len(m))
    for sub, arr in (('c', cells), ('h', holes), ('n', nuclei)):
        (tmp_path / sub).mkdir()
        tifffile.imwrite(tmp_path / sub / f'{key}_x.tif', arr.astype(np.uint16))
    labels = np.unique(cells[cells > 0])
    adjacent = set(m.index[m.hole_adjacent])
    csv = pd.DataFrame({'cell_id_unique': [f'{key}_{l}' for l in labels],
                        'cell_type': ['Senescent' if l in adjacent else 'Non-senescent' for l in labels],
                        'rule_based_classification_granular': ['Rule_Sen_VeryLarge' if l in adjacent
                                                               else 'Rule_NonSenescent' for l in labels],
                        'cell_area': [9000.0 if l in adjacent else 2000.0 for l in labels],
                        'nuclei_count': [1] * len(labels), 'nucleus_to_cell_area_ratio': [0.2] * len(labels)})
    csv.to_csv(tmp_path / 'classes.csv', index=False)
    out = tmp_path / 'out'
    dg.main(['--cells', str(tmp_path / 'c'), '--holes', str(tmp_path / 'h'), '--nuclei', str(tmp_path / 'n'),
             '--classes', str(tmp_path / 'classes.csv'), '--px-um', '0.65', '--very-large-um2', '3000',
             '--out', str(out)])
    text = (out / 'diagnostics.md').read_text()
    assert '100.0% (n=' in text and '0.0% (n=' in text      # all calls next to the gap, none elsewhere
    assert 'Sensitivity of the senescent fraction' in text and 'Rule_Sen_VeryLarge' in text


def test_audit_gallery_build_and_score(tmp_path):
    import audit_gallery as ag
    key = '1.4Pa_A1_20dec21_20xA_L2RA_FlatA_seq003'
    cells = grid(gap=2)
    rng = np.random.default_rng(1)
    membrane = rng.normal(100, 10, cells.shape)
    membrane[cells == 0] = 400                      # bright junctions between cells
    for sub, arr in (('c', cells), ('m', membrane)):
        (tmp_path / sub).mkdir()
        tifffile.imwrite(tmp_path / sub / f'{key}_x.tif', arr.astype(np.float32 if sub == 'm' else np.uint16))
    labels = np.arange(1, 17)
    rule = ['Rule_Sen_VeryLarge' if l <= 4 else 'Rule_NonSenescent' for l in labels]
    pd.DataFrame({'cell_id_unique': [f'{key}_{l}' for l in labels], 'rule_based_classification_granular': rule,
                  'cell_type': ['Senescent' if r != 'Rule_NonSenescent' else 'Non-senescent' for r in rule]}
                 ).to_csv(tmp_path / 'cls.csv', index=False)
    out = tmp_path / 'audit'
    ag.main(['build', '--cells', str(tmp_path / 'c'), '--membrane', str(tmp_path / 'm'),
             '--classes', str(tmp_path / 'cls.csv'), '--per-stratum', '4', '--out', str(out)])
    key_df = pd.read_csv(out / 'audit_key.csv')
    assert len(key_df) == 8 and len(list((out / 'crops').glob('*.png'))) == 8
    html = (out / 'index.html').read_text()
    assert 'Rule_Sen' not in html and 'Senescent' not in html      # auditor stays blind to the class
    # Expert: only half of the "very large" calls are real senescent single cells; the rest are gap leakage.
    sen_ids = key_df[key_df.stratum == 'Rule_Sen_VeryLarge'].audit_id.tolist()
    ver = [dict(audit_id=a, seg='correct', pheno='normal') for a in key_df.audit_id if a not in sen_ids]
    ver += [dict(audit_id=a, seg='correct' if k < 2 else 'leaks', pheno='senescent_like') for k, a in enumerate(sen_ids)]
    pd.DataFrame(ver).to_csv(tmp_path / 'v.csv', index=False)
    ag.main(['score', '--audit', str(out), '--verdicts', str(tmp_path / 'v.csv')])
    text = (out / 'audit_summary.md').read_text()
    # 4 VeryLarge cells, half real -> 2 senescent of (2 + 12) correct cells = 14.3 %, reported 25 %
    assert '14.3%' in text and 'Reported by the pipeline for the same strata: 25.0%' in text
    assert ag.wilson(5, 10)[1] < 0.5 < ag.wilson(5, 10)[2]


def test_sample_rois_stratifies_and_blinds(tmp_path):
    import sample_rois as sr
    img_dir, hole_dir = tmp_path / 'img', tmp_path / 'holes'
    img_dir.mkdir()
    hole_dir.mkdir()
    for i in range(1, 7):
        key = f'0Pa_A1_19dec21_20xA_L2RA_FlatA_seq{i:03d}'
        tifffile.imwrite(img_dir / f'{key}_Cadherins.tif', np.zeros((512, 512), np.uint16))
        holes = np.zeros((512, 512), np.uint8)
        if i % 2:
            holes[100:300, 100:300] = 1          # odd fields have a big gap, even fields none
        tifffile.imwrite(hole_dir / f'{key}_segmented.tif', holes)
    out = tmp_path / 'manifest.csv'
    sr.main(['--images', str(img_dir), '--holes', str(hole_dir), '--rois-per-condition', '4',
             '--fields-per-condition', '2', '--size', '128', '--out', str(out)])
    man = pd.read_csv(out)
    cells, fields = man[man.task == 'cells'], man[man.task == 'holes']
    assert sorted(cells.stratum.value_counts().to_dict().items()) == [('confluent', 2), ('gap', 2)]
    assert (cells[cells.stratum == 'gap'].pred_hole_frac >= 0.03).all()
    assert (cells[cells.stratum == 'confluent'].pred_hole_frac <= 0.005).all()
    assert cells.key.is_unique                   # at most one ROI per field
    assert (fields.pred_hole_frac == 0).any()    # a pipeline-empty field is always included
    assert man.roi_id.is_unique and man.blind_id.is_unique
    assert ((cells.y0 >= 32) & (cells.y0 + 128 <= 512 - 32)).all()


def test_export_from_drive(tmp_path, monkeypatch):
    import types
    import zipfile
    import export_from_drive as ex
    root = tmp_path / 'Thesis'
    k20, k40 = '0Pa_A1_19dec21_20xA_L2RA_FlatA_seq001', '1.4Pa_A1_20dec21_40x_L2RA_FlatA_seq002'
    files = {
        f'Segmented/Static-x20/Cell_merged_conservative/{k20}_cell_mask_merged_conservative.tif': grid(),
        f'Segmented/Static-x20/Holes/denoised_{k20}_Cadherins_regional_segmented.tif': np.zeros((100, 100), np.uint8),
        f'Segmented/Static-x20/Holes/denoised_{k20}_Cadherins_regional (1)_segmented.tif': np.zeros((100, 100), np.uint8),
        f'Segmented/1.4Pa-x40/Holes/denoised_{k40}_Cadherins_regional_segmented.tif': np.zeros((100, 100), np.uint8),
        f'Segmented/1.4Pa-x40/Holes/denoised_{k40}_Cadherins_regional_segmented_dilated.tif': np.zeros((100, 100), np.uint8),
        f'Projected/Static-x20/Cadherins/tophat/denoised_{k20}_Cadherins_regional_tophat.tif':
            np.random.default_rng(0).integers(0, 60000, (300, 300)).astype(np.uint16),
    }
    for rel, arr in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(root / rel, arr)
    (root / 'Analysis/Static-x20/Senescence_Results').mkdir(parents=True)
    (root / 'Analysis/Static-x20/Senescence_Results/cell_classification_rule_based_full.csv').write_text('a\n1\n')
    # calibration: the converted TIFF has none (as written by 2Tiff.ipynb), one OME-TIFF has it, and an .nd2 has it
    (root / 'TIF_Converted/1.4Pa-A-1').mkdir(parents=True)
    tifffile.imwrite(root / f'TIF_Converted/1.4Pa-A-1/{k20}.tif', np.zeros((2, 8, 8), np.uint16))
    tifffile.imwrite(root / f'TIF_Converted/1.4Pa-A-1/{k40}.ome.tif', np.zeros((8, 8), np.uint16), ome=True,
                     metadata={'PhysicalSizeX': 0.1625, 'PhysicalSizeY': 0.1625, 'axes': 'YX'})
    (root / 'Renamed Data/Static-A-1').mkdir(parents=True)
    (root / f'Renamed Data/Static-A-1/{k20}.nd2').write_bytes(b'')

    class FakeND2:
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def voxel_size(self): return types.SimpleNamespace(x=0.325, y=0.325, z=0.7)
        sizes = {'Z': 21, 'C': 3, 'Y': 1024, 'X': 1024}
    monkeypatch.setitem(sys.modules, 'nd2', types.SimpleNamespace(ND2File=FakeND2))

    sizes, have = ex.pixel_sizes(str(root))
    assert have and sizes['A1_20x']['px_x_um'] == 0.325 and sizes['A1_40x']['px_x_um'] == pytest.approx(0.1625)
    out = tmp_path / 'export'
    ex.main(['--root', str(root), '--out', str(out), '--part', 'all', '--conditions', 'Static-x20', '1.4Pa-x40'])
    report = zipfile.ZipFile(out / 'common_all.zip').read('metadata/export_report.md').decode()
    row = next(l for l in report.splitlines() if 'Static-x20/Holes ' in l)
    assert row.split('|')[5].strip() == '1'                       # the "(1)" copy is flagged
    row40 = next(l for l in report.splitlines() if '1.4Pa-x40/Holes ' in l)
    assert row40.split('|')[5].strip() == ''                      # plain + dilated are not duplicates
    assert 'missing/empty' in report and 'A1_20x | 0.3250' in report
    names = zipfile.ZipFile(out / 'Static-x20_all.zip').namelist()
    assert f'Segmented/Static-x20/Cell_merged_conservative/{k20}_cell_mask_merged_conservative.tif' in names
    assert 'Analysis/Static-x20/Senescence_Results/cell_classification_rule_based_full.csv' in names
    split = tmp_path / 'split'
    ex.main(['--root', str(root), '--out', str(split), '--part', 'all', '--conditions', 'Static-x20',
             '--split-mb', '0.15'])
    parts = sorted(p.name for p in split.glob('Static-x20_all_part*.zip'))
    assert len(parts) >= 2 and not (split / 'Static-x20_all.zip').exists()
    assert sum(len(zipfile.ZipFile(split / p).namelist()) for p in parts) == len(names)
