# Validating the segmentation (cells, nuclei, holes)

Tools and protocol to measure how well the imaging pipeline segments whole
cells, nuclei and intercellular gaps ("holes"), and how much those errors move
the numbers that feed the model: cell area, aspect ratio, orientation,
multinucleated and senescent fractions, and gap area fraction.

> **Data handling.** This repository is public and the images are
> access-restricted (see `data/Costanza/README.md`). Never commit images,
> masks, annotations or per-cell tables here. Keep them on Drive or in a
> private repository.

## Which Google Drive folders are needed

This layout is not documented anywhere else; it was reconstructed from the
paths hard-coded in the notebooks. All paths are under
`MyDrive/knowledge/University/Master/Thesis/`, for each condition `<c>` in
Static-x20, Static-x40, 1.4Pa-x20 and 1.4Pa-x40 (the A1 experiment). The
notebooks also processed flow3-x20 (series U). It is not used, because its
experimental conditions are unknown.

**Masks and tables** (small; enough for the diagnostics):
- `Segmented/<c>/Cell_merged_conservative` (what every reported number uses) and `Segmented/<c>/Cell`
- `Segmented/<c>/Nuclei`, plus `Nuclei_filtered` for the two 1.4 Pa conditions
- Seeds: `Segmented/<c>/Seed` (Static-x20, Static-x40), `Seed_or` (1.4Pa-x20), `Seed_gol` (1.4Pa-x40)
- `Segmented/<c>/Holes`, plus `Holes_masks` for the two 1.4 Pa conditions
- `Analysis/<c>/Senescence_Results`
- Once: `Analysis/combined_cell_data_adjusted.csv`, `Analysis/descriptive_stats_by_pressure_cell_type_adjusted.csv`, `Analysis/Holes/`

**Images** (for the blind audit and annotation):
- `Projected/<c>/Cadherins/tophat` and `Projected/<c>/Cadherins/background`
- `Projected/<c>/Nuclei/tophat` (Static-x20, 1.4Pa-x20) or `Projected/<c>/Nuclei/background` (the other two)
- `Projected/<c>/Golgi/tophat`

**Pixel size:** one `.nd2` per magnification from `Renamed Data/` (A1 at 20x
and at 40x). The files in `TIF_Converted/` were written without metadata and
cannot provide it.

Not needed: `TIF_Converted/`, `denoised*/`, the segmented membrane and Golgi
masks (`Segmented/<c>/Membrane*`, `Segmented/<c>/Cadherins*`,
`Segmented/<c>/Golgi`), the `*_images/` and `*_vis*/` figure folders, and the
trial and temp folders.

`export_from_drive.py` collects exactly these files, one zip per condition
(split below GitHub's 100 MB limit). It reads the pixel sizes from the
`.nd2` headers and reports any folder that is missing or contains duplicated
fields. In Colab:

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone -q -b claude/gallant-goodall-vqzzxx https://github.com/guidoputignano/Master-s-Thesis
!pip -q install nd2
!python Master-s-Thesis/Imaging/Validation/export_from_drive.py --check        # report only
!python Master-s-Thesis/Imaging/Validation/export_from_drive.py --part masks   # zips into Thesis/validation_export/
```

`export_nd2.py` goes back to the original A1 `.nd2` files for what the projected
TIFFs lost. Per field it writes one metadata row (objective, recorded and corrected
pixel size, z step, channels, exposure, stage X/Y/Z, acquisition time), the sharpest
plane of each channel and the DAPI sum over z (DNA content), with `--stacks` also the
full z-stack. Pixel sizes are the stage calibration (0.650 µm at 20x, 0.325 µm at 40x), with the
recorded values kept. In Colab:

```python
!python Master-s-Thesis/Imaging/Validation/export_nd2.py --check    # list files and metadata only
!python Master-s-Thesis/Imaging/Validation/export_nd2.py            # zips into Thesis/nd2_export/
```

## The plan: three tiers, cheapest first

| Tier | What it answers | Human time | Tool |
|---|---|---|---|
| 0. Freeze | One configuration for every condition; nothing tuned per image | 1–2 h | notebooks |
| 1. Diagnostics | Internal inconsistencies, degenerate objects, whether senescent calls pile up next to gaps and field borders | none | `diagnostics.py` |
| 2. Blind audit | Of the cells the pipeline outputs, how many are correct single cells (per senescence rule), and the error-corrected senescent fraction | 1–2 h expert | `audit_gallery.py` |
| 3. Small ground truth | Recall, boundary accuracy, error on the reported means, and one leaderboard comparing every method | ~1 day | `sample_rois.py`, `seg_eval.py` |

**Tier 0 comes first**, otherwise the ground truth validates a moving target.
Today the hole threshold is gated per folder and overridden per image, the
seed-merge distance differs per condition (2–25 px), only the 1.4 Pa
conditions exclude holes from the watershed, and the senescence gates differ
per condition. Fix one setting everywhere (or state per condition *a priori*,
and why), re-run, then measure. If you tune on annotated tiles, keep a
separate set of tiles for reporting.

## Tier 1: diagnostics on the existing masks (no annotation)

```bash
python Imaging/Validation/diagnostics.py \
    --cells   Segmented/*/Cell_merged_conservative --cells-glob '*merged_conservative.tif' \
    --nuclei  Segmented/*/Nuclei --nuclei-glob '*_filtered_mask.tif' \
    --holes   Segmented/Static-x20/Holes Segmented/1.4Pa-x20/Holes_masks ... --holes-glob '*_segmented.tif' \
    --classes Analysis/*/Senescence_Results/cell_classification_rule_based_full.csv \
    --px-um '20x=0.65,40x=0.325' --very-large-um2 <one physical gate> --out diag/
```

Read `diag/diagnostics.md`. Red flags include cells overlapping holes, a
higher senescent rate next to gaps or on the border than in the interior,
circularity > 1, cells without nuclei, and a senescent fraction that changes a
lot under the harmonised gates. Each is a finding to fix or to disclose.

## Tier 2: blind error audit (fastest honest numbers)

```bash
python Imaging/Validation/audit_gallery.py build \
    --cells Segmented/Static-x20/Cell_merged_conservative Segmented/1.4Pa-x20/Cell_merged_conservative \
    --membrane Projected/Static-x20/Cadherins/tophat Projected/1.4Pa-x20/Cadherins/tophat \
    --nuclei-img Projected/Static-x20/Nuclei/tophat Projected/1.4Pa-x20/Nuclei/tophat \
    --classes Analysis/Static-x20/Senescence_Results/cell_classification_rule_based_full.csv ... \
    --by-condition --per-stratum 30 --out audit/
# open audit/index.html, judge with the keyboard (1-6 segmentation, q/w/e/r phenotype), Download CSV
python Imaging/Validation/audit_gallery.py score --audit audit --verdicts audit_verdicts.csv
```

The page never shows the predicted class. With 30–50 cells per stratum, the
95% CI on a proportion is about ±13–17 points. That is enough to show whether,
say, the "very large" rule mostly catches real senescent cells or gap leakage.
Ideally the auditor is a second person (e.g. the data owner). Judge a random
20% twice to report intra-rater agreement.

## Tier 3: small, stratified ground truth

1. **Choose regions before looking** (seeded, stratified by condition and by
   gap versus confluent regions; at most one ROI per field; includes fields
   the pipeline called hole-free):

   ```bash
   python Imaging/Validation/sample_rois.py --images Projected/*/Cadherins/tophat \
       --holes <hole mask folders> --holes-glob '*_segmented.tif' \
       --rois-per-condition 4 --fields-per-condition 4 --size 400 --out manifest.csv
   ```

2. **Annotate in napari**, in `blind_id` order. For each `cells` row, open the
   whole field (VE-cadherin, nuclei and Golgi channels) and draw the ROI
   rectangle (`y0, x0, h, w`). Then add:
   - a Labels layer: every cell **whose centroid lies inside the ROI**, drawn
     *completely*, even where it extends past the rectangle. Save as
     `<roi_id>_cells.tif` (full-field size, uint16).
   - a Points layer: one click per nucleus in those cells. Save as
     `<roi_id>_nuclei.csv` (napari's points CSV).

   For each `holes` row, paint gaps on the whole field and save as
   `<roi_id>_holes.tif`.

   *Speed:* correcting a pre-segmentation is 3–5× faster than drawing from
   scratch, but it biases the ground truth towards whatever you start from.
   Start from a **different** method (e.g. Cellpose, or the Voronoi
   baseline), never from the pipeline being scored. Draw at least two ROIs
   from scratch to check the bias.

3. **Annotation rules** (write them down; apply them identically everywhere):
   - A cell boundary follows the centre of the VE-cadherin line. Where the
     line is broken, close it by the shortest plausible path.
   - A gap is a region with no cell: no VE-cadherin, no cytoplasmic
     background, no nucleus. It must be larger than a fixed minimum (for
     example 10 µm²; decide before annotating).
   - Multinucleated means ≥ 2 nuclei inside one continuous VE-cadherin
     contour. Dividing cells (mitotic figures) and dead or very bright
     objects are left unlabelled and ignored.
   - Border cells are drawn as visible. The scripts exclude cells touching
     the field border from the morphometric comparison.

4. **Score every method on the same ground truth** (one leaderboard):

   ```bash
   python Imaging/Validation/seg_eval.py cells --gt gt/ --manifest manifest.csv --px-um '20x=..,40x=..' \
       --gt-nuclei gt/ --pred-nuclei Segmented/Static-x20/Nuclei --pred-nuclei-glob '*_filtered_mask.tif' \
       --method "pipeline=Segmented/Static-x20/Cell_merged_conservative::*merged_conservative.tif" \
       --method "pipeline_no_enclave_merge=Segmented/Static-x20/Cell::*_cell_mask.tif" \
       --method "cellpose_cyto3=<dir>" --method "voronoi=<dir>" --out results/cells
   python Imaging/Validation/seg_eval.py holes  --gt gt/ --manifest manifest.csv --px-um ... \
       --min-area-um2 10 --method "valley=<dir>::*_segmented.tif" \
       --method "valley_dilated=<dir>::*_segmented_dilated.tif" --method "otsu=<dir>" --out results/holes
   python Imaging/Validation/seg_eval.py nuclei --gt gt/ --manifest manifest.csv \
       --method "cellpose_cyto3_filtered=Segmented/Static-x20/Nuclei::*_filtered_mask.tif" \
       --method "cellpose_nuclei=<dir>" --out results/nuclei
   python Imaging/Validation/seg_eval.py voronoi --nuclei Segmented/Static-x20/Nuclei \
       --nuclei-glob '*_filtered_mask.tif' --out baselines/voronoi   # nearest-nucleus null model
   ```

   Methods worth including: the pipeline as reported, each of its stages
   switched off (no seed merging, no enclave merge, no hole exclusion), a
   generalist deep model on the two-channel image (Cellpose `cyto3` or
   `cpsam`, with nuclei as the second channel), and the Voronoi null model.
   Any method that beats Voronoi only marginally is not using the junction
   channel. The same table also supports (or refutes) the paper's statement
   that deep models "did not reach the accuracy required".

### How much to annotate

| Item | Suggested amount | Why |
|---|---|---|
| Cell ROIs | 4 per condition × 400×400 px (x20), ≈ 50–70 cells each; ≈ 250 cells per condition | F1 CI of about ±4–5 points per condition; the reported-mean errors become interpretable |
| Nucleus clicks | same ROIs | 2–3 s per nucleus; gives nuclear recall and the multinucleation truth |
| Hole fields | 4 whole fields per condition, including ≥ 1 the pipeline called empty | gap fraction is a per-field quantity; this also checks the "no holes" decisions |
| Second annotator | 2–3 ROIs and 2 fields | the pipeline–human agreement is only interpretable next to human–human agreement |

## What the scripts report

- **Cells:** precision/recall/F1 at IoU 0.5 and 0.75, AP averaged over IoU
  0.5–0.95, panoptic quality, and the fraction of GT cells split or merged.
  The key table is the **reported quantities**: mean area, aspect ratio,
  misalignment and multinucleated fraction over *all* in-scope interior
  cells (merges and spurious objects included, as in the analysis
  notebooks), next to the same statistics from the ground truth. There are
  also Bland–Altman and concordance statistics for matched cells.
- **Holes:** gap area fraction (GT vs predicted, mean absolute error in
  percentage points), pooled pixel Dice, object F1 above the minimum size,
  and agreement on whether a field has any gap at all.
- **Nuclei:** precision/recall/F1 from clicks; objects hit by ≥ 2 clicks are
  merged nuclei.
- All CIs are cluster-bootstrap intervals that resample fields/ROIs, not
  cells, because cells in one field are not independent.

## Units

Area thresholds and µm² values require the true pixel size, and for A1 the `.nd2` headers do not
give it.

- **What the headers say.** Every A1 file records 0.429 µm: Andor iXon 888, 13 µm pixels,
  Plan Apo 20x/0.75, and the Ti zoom changer at 1.5 (× 1.01). The "40x" files record the same
  state, although their images are sampled twice as finely. The recorded optics were stale.
- **The true pixel.** `stage_calibration.py` measures it from two overlapping fields (stage
  displacement over image shift): 0.650 µm at 20x, 0.325 µm at 40x.
  `Imaging/pipeline_v2/features.py` and `calibration.csv` use these values;
  `export_from_drive.py` and `export_nd2.py` flag the recorded values.
- **Older values.** Earlier code used 0.325 µm (`Analysis/Cell_density.ipynb`: densities four
  times too high), 0.429 µm (`Deconv.ipynb`) and 650/1024 ≈ 0.635 µm (the paper's field of view,
  2 % off).
- **No default.** The scripts deliberately have none: pass `--px-um '20x=0.65,40x=0.325'` for A1.

## Files

- `seg_eval.py`: metrics library and CLI (`cells`, `holes`, `nuclei`, `voronoi`, multi-method leaderboard)
- `diagnostics.py`: ground-truth-free checks on existing outputs
- `audit_gallery.py`: blind sampled audit (`build` writes crops and a local verdict page; `score` summarises)
- `sample_rois.py`: seeded, stratified manifest of what to annotate
- `export_from_drive.py`: collects the needed Drive files, reads pixel sizes, flags gaps
- `export_nd2.py`: `.nd2` metadata with stage positions, sharpest planes and DAPI sums (A1 only)
- Tests: `tests/test_seg_eval.py` (synthetic cases with known answers)

Requirements: `numpy scipy scikit-image pandas tifffile pillow`.
