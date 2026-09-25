# Imaging pipeline v2 (alternative to the original pipeline)

A second segmentation and senescence analysis that runs next to the original
notebooks (called **v1** here), with the same inputs and the same per-cell
features. The two can be compared field by field.

There are two variants:

- **v2** (`segment_v2.py`) is Cellpose alone.
- **v2.1** (`refine_v2.py`) is v2 corrected after the first blind review.
  Cellpose often outlines only the part of a cell around the nucleus when
  junctions are faint, so v2.1 grows each Cellpose cell over the junction
  landscape to its junctions. It drops nucleus-free fragments and gives every
  nucleus without a cell its own marker.

> **Junction stain.** The junction channel (folder `Cadherins`) is
> **β-catenin**, not VE-cadherin: Nafsika Chala's notes that came with the
> `.nd2` files list DAPI, β-catenin (mouse) and a Golgi marker (rabbit). The
> junction channel is called "junction" below.

> **Data handling.** This repository is public and the images are
> access-restricted (see `data/Costanza/README.md`). Never commit images,
> masks, verdict sessions or per-cell tables here. Write them to a private
> location (for example the private data repository).

## What changes relative to v1

| Step | v1 (notebooks) | v2 (this folder) |
|---|---|---|
| Nuclei | Cellpose `cyto3` on the nuclear channel (d = 30 px), intensity/area filter | Cellpose `nuclei`, d = 17.0 µm (26 px at 20x, 52 px at 40x); nuclear top-hat at 20x, projection without top-hat at 40x (the top-hat hollows 40x nuclei) |
| Whole cells | one watershed region per nuclear seed on the junction gradient; seeds merged by a corridor test with per-condition distances (2–25 px) | Cellpose `cyto3` on the junction channel + nuclei, d = 38.9 µm (60 px at 20x, 120 px at 40x) |
| Holes / gaps | histogram-valley threshold, per-folder gating and manual overrides; watershed barrier only at 1.4 Pa | darker than all but 1 % of the field's cell interiors (junction channel without top-hat, σ = 1.5 µm), outside nucleated cells, opened with r = 1.30 µm (2 px at 20x, 4 px at 40x), >= 23 µm², at most 5 % nuclear pixels; v2.1 grows each gap over connected pixels that pass at the 5th percentile; uncovered area reported separately |
| Parameters | per condition, some per field | one setting for every condition, in micrometres |
| Units | pixels (x40 areas divided by 4) | µm, 0.650 µm/px at 20x and 0.325 µm/px at 40x (stage calibration; `calibration.csv` per file) |
| Border cells | kept in all statistics | flagged; excluded from morphology and senescence estimates |
| Senescence | ordered gates ("Example value" thresholds, per condition) and k-means (k = 2) | shifted-population log-normal mixture of cell area per condition, with BIC and field-level bootstrap (below) |
| Flow alignment | mean misalignment angle | also the nematic order parameter S = \|⟨exp(2iθ)⟩\| per field (0 = random, 1 = aligned) |

**Calibration: 0.650 µm/px at 20x, 0.325 µm/px at 40x.**

- **What the files record.** Every A1 `.nd2` file records 0.429 µm/px: Andor iXon 888 EMCCD
  (13 µm pixels, 1024 × 1024), Plan Apo 20x/0.75, and the Nikon Ti zoom changer at position
  1.5 (× a 1.01 relay), so 13 / (20 × 1.515) µm. The 24 "40x" files record the same state,
  although they were taken with a twice stronger objective. The recorded optics were stale.
- **What the stage says.** Six pairs of overlapping field positions, 103–668 µm apart on the
  stage, give 0.649–0.653 µm/px in all three channels, the camera pixel through the 20x objective
  alone (`../Validation/stage_calibration.py`). For example, 1.4 Pa 19dec21 seq013 and seq016 lie
  102.8 µm apart on the stage and 158 px apart in the images. Five pairs lie 537–668 µm apart,
  farther than a field would be wide at 0.429 µm/px (439 µm), and still share a strip of cells.
- **Cross-checks.** The same check returns the recorded pixel size of 2019 files of the same
  microscope and camera, taken at zoom position 1.0. Untreated HUVEC nuclei imaged there have the
  A1 nuclear size in pixels.
- **40x.** Those nuclei are four times larger in pixels, so the pixel is 0.325 µm. The one
  overlapping 40x pair gives 0.328 µm on the stage (1 % more, 2 % in area); 0.325 is kept.
- **The imaged field is 666 × 666 µm.** The acquisitions are widefield z-stacks (spinning disk
  out of the light path; 13 × 0.7 µm at 20x, 21 × 0.4 µm at 40x).
- **Constants.** They were set, and their outputs reviewed, at the recorded 0.429 µm. Each µm
  constant is its original value × `features.SCALE` (1.515; areas × 2.296), so every pixel
  operation and mask is unchanged (checked on three fields: identical nuclei, cells and gaps).
- **Per file.** `calibration.csv` holds the per-file record (`nd2_link.py calibration`).
- **Legacy notebooks.** `Deconv.ipynb` used 0.429 µm with the zoom, and
  `Analysis/*/Cell_density.ipynb` used 0.325 µm at 20x, so their densities are four times too
  high.

**Gaps.** β-catenin marks junctions, so a gap and a cell interior look
alike in the top-hat image. The difference is the diffuse cytoplasmic signal,
which bare substrate lacks. The gap rule therefore uses the projection
without top-hat and a per-field threshold taken from the cells themselves.
Uncovered pixels that are at cytoplasm level, or contain a nucleus, are
missed cells, not gaps.

**Gap extent (v2.1, after the second review).** The reviewer saw gaps
extending beyond their outline. A gap now starts from the 1st-percentile
seeds, which are unchanged, and grows over connected pixels that pass the same
test at the 5th percentile. Enclosed specks under 23 µm² without nuclear
signal are filled. `refine_v2.py --no-grow` keeps the seeds only, which are
the masks the second review showed; they are also stored as value 8 in
`_v2_gaps_sens.tif`.

**Field quality and repeated fields** (`quality.py`). Neither check uses a
segmentation.

- *Junction clarity.* Bright ridges are measured at the junction scale on the
  junction top-hat (Hessian, σ = 0.91 µm). The score is their 95th
  percentile divided by the noise (MAD of the Laplacian residual). A field is
  low quality when its log score is more than 3 robust SDs below the median
  of its condition and magnification. Defocus and haze lower the score.
  Comparing within a condition keeps flow-induced junction changes out of the
  rule.
- *Repeated fields.* Two fields of one condition that image the same area
  (phase correlation: overlap >= 50 %, r >= 0.9) would count the same cells
  twice. The one with the higher score is kept.

`analyze.py --exclude-fields quality.csv` leaves out both kinds.
`build_verdicts.py score --exclude-fields quality.csv --strata strata.csv`
leaves out answers from low-quality fields only, because repeated fields are
clear images. It rescales each condition × stratum population to the fields
that remain; `build_verdicts.py strata` writes those counts.

**Flow direction** (`polarity.py`). The A1 file names carry "L2RA": left to
right was assumed when the files were renamed, not recorded. Under laminar
flow endothelial cells move the Golgi upstream of the nucleus, and Chala et
al. (2021) found this in senescent monolayers too, although those did not
align. `polarity.py` measures the nucleus-to-Golgi vector of every interior
cell and reports, per slide and per field, the mean resultant length R
(0 = no preferred side), its direction and the Rayleigh p-value. A channel
offset would add the same vector to every cell, so the statistics are
repeated after subtracting the mean vector of the static slide of the same
experiment and magnification.

**Nuclei folder in v1.** `Segmented/<c>/Nuclei` is the filtered subset of
`Nuclei_raw` and the source of the seeds in Static-x20, Static-x40 and
1.4Pa-x20. In 1.4Pa-x40, `Nuclei` is a copy of `Nuclei_raw` and the seeds
came from `Nuclei_filtered`, so `analyze.py` uses `Nuclei_filtered` there.

## Senescence estimate

Chala et al. (2021, Nano Lett. 21:4911) report TNF-α-treated HUVECs 2.27×
larger than controls on average (5335 vs 2354 µm²), with log-normal area
distributions. The A1 slides mix 70 % control and 30 % TNF-α-treated cells.
The absolute thresholds of that paper (e.g. > 5000 µm²) do not transfer,
because cell areas depend on the segmentation, but the scale-free prediction does. The
log cell area of interior cells should be a mixture of two components: the
larger one should have a median about 2.3× the smaller one's and hold at most
about 30 % of the cells.

"Larger by a factor" means the same distribution shape shifted in log area,
so the two components share one variance. The constraint is needed. With free
variances, the likelihood is often highest for a narrow core plus a broad
component (weight 0.7–0.8, ratio 1.2–1.9), which describes skewness rather
than a second population. `senescence.py` fits the shifted-population model
by EM from several starting points, either with a free ratio or with the
ratio fixed at 2.27. With a shared variance the ratio of medians equals the
ratio of means, which is what Chala reports. The enlarged component must also
be the minority (<= 50 % of cells). Without that constraint the best fit under
flow can be a small-cell tail with the "enlarged" label on the majority. The
best unconstrained fit is still reported (`delta_bic_any`). The model
compares one versus two components by BIC and bootstraps whole fields for the
intervals. Each cell
gets a posterior probability of belonging to the enlarged component.
Hand-set gates and k-means are no longer needed; k-means always returns two
clusters, whether or not two populations exist. The mixture weight is the
estimate of the fraction. The count of cells with posterior > 0.5 is a
classification and is biased when the components overlap.

An enlarged "cell" is only meaningful if it is one cell. The verdict session
therefore asks the expert, blind to the method, whether enlarged calls are
single cells.

## Running it

```bash
pip install cellpose==3.1.1.1 tifffile scikit-image pandas scipy pillow
# 1. segmentation (CPU: ~30 s per 20x field; --gpu if available)
python segment_v2.py --root /path/to/data-mt --out /private/v2_masks
# 1b. v2.1: grow the Cellpose cells to their junctions (seconds per field)
python refine_v2.py --root /path/to/data-mt --v2 /private/v2_masks --out /private/v2r_masks
# 1c. field quality and repeated fields
python quality.py --root /path/to/data-mt --v2 /private/v2r_masks --out /private/quality.csv
# 2. per-cell features for v1 and v2 (or v2.1: --v2 /private/v2r_masks), agreement, gaps, alignment, mixtures
python analyze.py --root /path/to/data-mt --v2 /private/v2_masks --out /private/v2_analysis \
    --exclude-fields /private/quality.csv
# 2b. flow direction from the nucleus-to-Golgi vectors (v1 cells, Cellpose nuclei)
python polarity.py --root /path/to/data-mt --v2 /private/v2r_masks --out /private/polarity \
    --exclude-fields /private/quality.csv
# 2c. per-cell senescence features (Cellpose nuclei, Golgi, junction signal) for v1 and v2.1 cells
python cell_features.py --root /path/to/data-mt --masks /private/v2r_masks --analysis /private/v2r_analysis \
    --quality /private/quality.csv --out /private/features
# 2d. the .nd2 files: metadata and stage positions, field key -> file, DNA content per nucleus
python nd2_link.py meta --repo /path/to/data-mt --out /private/nd2
python nd2_link.py map  --repo /path/to/data-mt --out /private/nd2
python nd2_link.py dna  --repo /path/to/data-mt --masks /private/v2r_masks --quality /private/quality.csv --out /private/nd2
# 2e. do gaps sit next to enlarged-nucleus cells more than their size explains? (shifted-mask null)
python gap_contact.py --root /path/to/data-mt --masks /private/v2r_masks \
    --features /private/features/features_v1.csv --out /private/gap_contact
# 2f. every result per shear stress (static, 1.4 Pa), the tables of IMAGING_VALIDATION
python by_shear.py --analysis /private/v2r_analysis --analysis-v2 /private/v2_analysis \
    --features /private/features --polarity /private/polarity/polarity_cells.csv \
    --dna /private/nd2/dna_index.csv --quality /private/quality.csv --root /path/to/data-mt --out /private/by_shear
# 3. blind yes/no session (one self-contained HTML file) and its scoring
python build_verdicts.py build --root /path/to/data-mt --v2 /private/v2_masks \
    --analysis /private/v2_analysis --out /private/verdicts
python build_verdicts.py score --session /private/verdicts/index.html --code 'VS147:YYN...'
# 3b. the same, without low-quality fields (population weights rescaled)
python build_verdicts.py strata --root /path/to/data-mt --v2 /private/v2_masks \
    --analysis /private/v2_analysis --tag v2 --out /private/strata.csv
python build_verdicts.py score --session /private/verdicts/index.html --verdicts answers.csv \
    --exclude-fields /private/quality.csv --strata /private/strata.csv
```

The data root follows the layout of the data repository:
`Projection/<c>/{Cadherins,Nuclei}/{tophat,background}`, `Segmented/<c>/...` and
`Senescence/<c>/Senescence_Results`, with `<c>` in Static-x20, Static-x40,
1.4Pa-x20 and 1.4Pa-x40: the A1 experiment. flow3-x20 is not used, because
its experimental conditions are unknown.

`by_shear.py` pools the clear fields of each shear stress; every field is
measured in µm at its own pixel size, so pooling needs no rescaling.
`--group folder` keeps the four folders apart and reproduces the per-folder
tables, as a check. Its nuclear-size estimate uses `nuclear_mixture.py`, the
same shifted log-normal model as `senescence.py` on the largest Cellpose
nucleus of each interior cell.

`analyze.py` writes `cells_v1.csv`, `cells_v2.csv` and `cells_posterior.csv`
(per cell), `fields.csv` (density, gap fraction, nematic order), `agreement.csv`
(v1 cells matched by v2 at IoU >= 0.5, splits, merges),
`senescence_mixture.csv` and `summary.md`. The summary includes the comparison
with Chala et al. (aspect ratio, misalignment, area ratio, density) and the
agreement between the reported rule-based calls and the mixture.

## Verdict session

`build_verdicts.py` samples three blocks, blind to the method:

- **cell**: interior cells of each method, stratified by whether the other
  method finds the same cell (IoU >= 0.5). The question is "exactly one whole
  cell?". The stratified estimate (Jeffreys posterior per stratum, weighted
  by the stratum's population share) is the cell precision.
- **enlarged**: v1 cells reported as senescent, and v2 cells in the enlarged
  component. The question is the same.
- **gap**: gap components of each method. The question is "a gap in the
  monolayer?". Scoring also weights the answers by component area. In round
  2 the components are drawn with replacement in proportion to area, and
  conditions are combined by gap area.

`build_verdicts2.py` builds round 2 (v1 vs v2.1) on objects never shown
before, with anything within 15 µm of a round-1 object excluded:

- cells, stratified as above;
- multinucleated cells;
- v2.1 enlarged cells;
- gaps sampled in proportion to their area.

Each crop has three views: junctions, "haze" (the junction channel without top-hat,
where cytoplasm is grey and bare substrate black) and Golgi.

`build_verdicts3.py` builds round 3 on the clear fields, per shear stress, for the
two masks no earlier round showed, with anything within 15 µm of a round-1 or
round-2 object excluded:

- the grown v2.1 gaps, drawn in proportion to their area: is all of the outlined
  area bare substrate? An overshoot into cytoplasm counts as "no";
- multinucleated v1 cells, each nucleus outlined in red: are they that many
  separate, whole nuclei, with no other nucleus in the cell?

```bash
python build_verdicts3.py build --root /path/to/data-mt --v2r /private/v2r2_masks \
    --analysis /private/v2r2_analysis_clean \
    --previous /private/verdicts/key.csv /private/v2_masks /private/v2_analysis \
    --previous /private/verdicts2/key.csv /private/v2r_masks /private/v2r_analysis \
    --out /private/verdicts3
python build_verdicts3.py score --session /private/verdicts3/index.html --code 'VS38:YYN...'
```

The page (`index.html`, crops embedded) is answered with Y/N/U. The answers
come back as a short code (`VS<n>:YYNU...`) or a CSV. `verdict_session.py`
in `../Validation` holds the page and the generic scoring.

Tests: `tests/test_pipeline_v2.py` (synthetic data with known answers).
