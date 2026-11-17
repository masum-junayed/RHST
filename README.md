# Reliability-aware Multimodal Histology + Spatial Transcriptomics

Research-grade PyTorch **reference implementation** corresponding to the proposed architecture:

`224x224 ST-aligned H&E patches → frozen pretrained UNI features`

`spatial gene counts → trainable gene MLP`

`bidirectional cross-attention → residual fusion → shared VAE → sample-level uncertainty`

`preliminary tumor relevance → uncertainty-conditioned residual spatial refinement`

`refined relevance → tumor-aware pooling → cancer grade classifier`

Loss = grading cross-entropy + λ_rec reconstruction MSE + β KL + λ_sp morphology-weighted neighbor smoothness.

**Status:** This is independently written from the methodology provided by the author, not original experiment code and not a verified reproduction of manuscript numbers. You must run it on genuine labeled data. No dataset, trained UNI weights, experimental outputs, or paper results are bundled. The HEST-1k collection as a whole does **not** supply universal slide-level grade labels; create a documented labeled cohort with a consistent grading system. HER2ST/CAMEO/Visium HD require separate label/task mapping. Do not claim all datasets share the same clinical grades.

## Structure

- `src/reliability_st/model.py`: fused feature model (UNI feature input, gene MLP, cross-attention, single VAE, spatial relevance, pooling, classification)
- `src/reliability_st/losses.py`: all four objectives and morphology-weighted kNN smoothness
- `src/reliability_st/data.py`: spot-aligned NPZ loader, patient-level grouped cross-validation, padded section batching
- `src/reliability_st/engine.py`: training, fold selection, held-out testing, result saving
- `scripts/prepare_spot_patches.py`: coordinate-centered cropping compatible with CLAM-style workflow
- `scripts/extract_uni_features.py`: UNI extraction from existing patches, optional gated HF access
- `scripts/train.py`, `scripts/test.py`, `scripts/infer.py`: workflows
- `scripts/plot_relevance.py`: plot **actual** spot relevance values (not fabricated ground truth)
- `scripts/check_data.py`: inspect real labeled cohort
- `tests/`: tiny synthetic fixture for functionality tests only

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements.txt
```

On Linux, install `libopenslide` via your OS package manager if reading SVS/large WSIs.

## Data contract

Your `data/manifest.csv` has **one row per spatial tissue section**:

```csv
sample_id,patient_id,cohort,grade_label,npz_path
section001,patient001,HEST-1k,2,features/section001.npz
section002,patient002,HEST-1k,3,features/section002.npz
```

The above rows show **schema only**; these IDs and grades are examples, **not actual HEST records**. `grade_label` must be a verified tissue/sample grade. Samples from a patient MUST share the same `patient_id`. Use a single grading ontology for a given comparison. Files can live at absolute paths or relative to the manifest directory.

Each `.npz` must include:

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `image_features` | `(N, D_uni)` | float32 | Frozen UNI embeddings, exactly one per ST spot |
| `genes` | `(N, G)` | float32 | **Nonnegative raw gene counts**, identical gene columns/order across all samples |
| `coords` | `(N, 2)` | float32 | `(x,y)` histology **pixel** coordinates matched to genes/patches |

Sample labels are **section-level** here. If ground truth is spot-level (e.g. Gleason annotation per spot), group/aggregate labels only with a medically justified protocol or change task/classifier—do not silently turn spot categories into slide-level cancer grades.

`max_spots` caps O(N²) attention cost. Training randomly selects a reproducible subset each epoch; validation/testing use deterministic subsets. For inference on an entire slide, provide a slide with a feasible spot count, or implement a tested spatial tiling/aggregation protocol; inference does **not** automatically cap spots.

### Convert registered patch coordinates and gene counts to UNI NPZ

**Stage 1** Prepare real spot metadata as CSV with columns:

`sample_id,patient_id,cohort,grade_label,slide_path,x,y,gene_BRCA1,gene_TP53,...`

These gene columns must be the same for all sections. Source coordinates must already be mapped into the full-resolution slide coordinate system. Use CLAM to define valid tissue regions if you used its segmentation; ensure coordinate conversion is correct. Our cropping script follows CLAM-compatible 224×224 preprocessing but is **not an exact reimplementation of CLAM tissue segmentation**.

```bash
python scripts/prepare_spot_patches.py \
  --spots-csv data/raw_spots.csv \
  --source-size 224 --size 224 \
  --out-csv data/patch_spots.csv
```

**Stage 2** Accept UNI's model license and authenticate to Hugging Face as authorized. A common route is `huggingface-cli login` or `hf auth login`. The extraction script loads `hf-hub:MahmoodLab/UNI` using timm. Model loading requirements may change with upstream timm/UNI; consult the official UNI model card. If you already extracted frozen UNI embeddings, **skip this step**.

```bash
python scripts/extract_uni_features.py \
  --spots-csv data/patch_spots.csv \
  --outdir data/features --batch-size 128
```

Output is `data/features/*.npz`, `data/manifest.csv`, and `data/genes.txt`. **128 refers to the image-feature extraction batch, NOT the section-level training batch.** The model uses `sample_batch_size: 2` by default; this is illustrative and adjustable.

Validate:

```bash
python scripts/check_data.py --manifest data/manifest.csv
```

## Train

Edit `configs/default.yaml` for `manifest`, cohort, number of folds, labels and hardware constraints. Current values (`d_model=512`, `n_heads=8`, `latent_dim=256`, initial LR=1e-4, StepLR) are author-discussed choices, **not independently confirmed experiment settings**.

```bash
python scripts/train.py --config configs/default.yaml --device auto
```

Each output fold saves `splits.csv`, `best.pt`, `test_metrics.json`, `test_predictions.json`. `cv_summary.json` contains actual fold means and sample standard deviations. GroupKFold + grouped validation ensure patient IDs never cross train/validation/test sets. **Cohort overlap must also be checked manually across HEST and named source datasets.** With small numbers of patients, some folds may be missing grade classes and AUC becomes undefined; the code reports `NaN` rather than fabricating AUC.

## Evaluate a held-out labeled manifest

```bash
python scripts/test.py --checkpoint outputs/default/fold_0/best.pt \
  --manifest data/external_test_manifest.csv \
  --config configs/default.yaml --output external_results.json
```

External labels **must** conform to the same class names and gene order and use the same encoder feature dimensionality. A fold model trained with a different grading ontology is **not** directly comparable to tumor-vs-normal labels.

## Inference (no labels needed)

```bash
python scripts/infer.py --checkpoint outputs/default/fold_0/best.pt \
  --npz data/features/section001.npz --out tumor_map.npz
python scripts/plot_relevance.py --prediction tumor_map.npz --out relevance.png
```

When a registered downsampled background is used, rescale spot coordinates to the displayed image correctly first. The figure is a **weakly supervised relevance map**, not a pathologist mask or calibrated pixel uncertainty. `uncertainty` is one scalar per section (mean VAE posterior variance), **not** a spatial heatmap.

