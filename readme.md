# Lesion-aware semantic transmission + uncertainty-guided learning for mammography

Research code for a simulation study on CBIS-DDSM:

* **RQ1** Does lesion-prioritized (semantic) transmission preserve diagnosis under packet loss better than normal transmission at equal bandwidth?
* **RQ2** Is model uncertainty a reliable signal for retransmission and for referral to a radiologist?
* **RQ3** Does uncertainty-based active learning reduce the number of expert labels needed?

```
mammogram -> SENDER: lesion localizer -> block priorities -> packets -> lossy channel
          -> RECEIVER: reconstruct -> localize lesion -> crop -> classify + uncertainty
                         uncertain -> request retransmission of lesion blocks -> still uncertain -> refer to radiologist
```

## 0. Data source
Notebook 00 downloads the Kaggle copy of CBIS-DDSM (`awsaf49/cbis-ddsm-breast-cancer-image-dataset`, a JPEG conversion) with `kagglehub`. **You do not put the dataset on Drive.** The raw data lives on Colab's temporary disk; only the small preprocessed cache is saved to Drive. JPEG is 8-bit and compressed, so state in your paper that you used a JPEG-converted copy. (The code also reads original DICOM folders if you ever use the TCIA download; see `src/data.py`.)

## 1. Put it on Google Drive
```
MyDrive/
  mammo_semcom/           <- THIS folder (upload it as is): src/, notebooks/, tests/, README.md
  mammo_semcom_work/      <- created automatically (cache, models, results, figures)
```
If your folders have other names, change `CODE_DIR`, `cbis_root`, `work_dir` in the first code cell of each notebook (same cell in all six).

## 2. Run the notebooks in order (Colab, Runtime > Change runtime type > T4 GPU, then Run all)
| Notebook | What it does | Output (in mammo_semcom_work/) |
|---|---|---|
| 00_data_preparation | finds files, builds labels, preprocesses, patient-level split, shows lesion outlines for a visual check | cache/ |
| 01_train_models | trains lesion localizer and classifier ensembles (clean + loss-robust), baseline AUC with CI | models/, results/baseline.csv |
| 02_transmission_experiments | RQ1: uniform / burst / semantic(pred) / semantic(oracle), two receivers, bootstrap stats, ablations | results/transmission_*.csv, figures/ |
| 03_uncertainty_and_adaptive_loop | RQ2: MC dropout vs ensemble, calibration, risk-coverage, retransmit-or-refer loop | results/uncertainty_*.csv, adaptive_loop.csv |
| 04_active_learning | RQ3: random / entropy / BALD / core-set, several seeds, learning-curve area | results/active_learning_*.csv |
| 05_summary_and_export | prints key tables, zips results and figures | mammo_semcom_results.zip |

Each notebook is independent after 00-01 and resumable (cached files and trained models are reused). Do a **dry run first** with `max_images = 150` in the config cell.

## 3. Key settings (src/config.py, overridable in the notebook's Cfg(...) call)
`img_h x img_w` working resolution (multiples of 32; default 768x512; raise if RAM allows), `block` packet size (32), `loss_rates`, `reps`, `burst_len`, `recon` ('mean' | 'zero' | 'inpaint'), epochs, `ensemble_size`, active-learning budget and strategies.

## 4. Design decisions worth stating in a paper
* Split by **patient**; images with mixed-label abnormalities are excluded from image-level evaluation.
* Equal bandwidth: every scheme loses the same number of blocks; background-only (all black) blocks are never transmitted. The priority tag costs ~2 bits/block.
* Importance comes from a **predicted** lesion map (realistic); the true-mask version is reported only as an upper bound.
* AUC (threshold-free) is the headline metric; at a fixed 0.5 threshold the gray filled-in blocks push predictions towards "malignant" and distort sensitivity/specificity.
* Patient-level bootstrap CIs and paired tests; report seeds and folds. `data.grouped_kfold` gives patient-level k-fold if you want cross-validated final numbers.

## 5. What has and has not been verified
Verified here (tests/): CSV/image discovery and matching by series UID for both a DICOM layout and a mock of the Kaggle JPEG layout (including swapped ROI/cropped columns, two-image series, MONOCHROME1, JPEG compression noise in masks, missing files, trailing newlines). The mock of the Kaggle layout is based on my recollection of it, not on the real download, breast crop / mirroring / mask alignment, packet-loss models and reconstruction, metrics and bootstrap, experiment and adaptive-loop logic (with a stand-in receiver), all figure functions.
**Not executed by the author:** the PyTorch parts (models, training loops, MC dropout, ensembles, active-learning loop) because PyTorch was not available in the authoring environment. They were written to be conventional and statically checked, but expect to fix small issues on the first run: copy the full error text and debug from there.

## 6. Limitations (put them in your paper)
Simulated channel and radiologist; digitized-film dataset from a limited population (add external validation, e.g. VinDr-Mammo, if you can get access); working resolution below native (small calcifications are partly lost); single dataset. Research prototype, not a medical device.

## 7. Data citation
Lee RS, Gimenez F, Hoogi A, Miyake KK, Gorovoy M, Rubin DL. A curated mammography data set for use in computer-aided detection and diagnosis research. Sci Data 4:170177 (2017). CBIS-DDSM on TCIA, doi:10.7937/K9/TCIA.2016.7O02S9CY (license CC BY 3.0). Check the dataset page for the exact citation text.
