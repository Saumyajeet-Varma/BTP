# Endsem-imp-pipeline-02.py — Documentation

This document describes **`Endsem_prep/Endsem-imp-pipeline-02.py`**: the same **two-stage** Car-Hacking pipeline as **`Endsem-imp-pipeline-01.py`**, extended with **synthetic probe signals** so you can **sanity-check Stage 2** (decoder-based and optional uniform fakes) without claiming they are real CAN logs.

For shared concepts (data loading, Stage 1 backbone, AAE training), see **`Endsem-imp-pipeline-01.md`**; pipeline-02 differs in **score calibration**, **`zero_day` diagnostics**, and probe QA.

---

## 1. Purpose

**File:** `Endsem_prep/Endsem-imp-pipeline-02.py`

**Pipeline (same high-level story as 01):**

1. **Stage 1** — Multi-class classifier (CANForge-style + focal loss): **Normal**, **DoS**, **Fuzzy**, **Gear**, **RPM**.
2. **Stage 2** — AAE trained **only on true Normal** rows from the training split; anomaly **score** and a **percentile threshold** on calibration normals that pass Stage 1.
3. **Hybrid rule** — If Stage 1 predicts an attack class, that label wins. If Stage 1 predicts **Normal**, Stage 2 scores the sample; if **score > threshold**, output **`zero_day`**.

**What pipeline-02 adds:**

- **`FakeSignalGenerator`** — Uses the trained **`Decoder`** only: sample **`z ~ N(0, σ² I)`**, output **`x = Decoder(z)`** in **`[0,1]^d`** (QA proxy in feature space, **not** logged CAN).
- **`uniform_fake_features`** — Optional **Uniform(0,1)** vectors for comparison.
- **Stage 2 scoring options** — Stats fitted on **AAE training normals** by default; **`max`** aggregation over recon/latent z-scores (more sensitive than a pure blend); default percentile aligned with the notebook-style **97** (less conservative than **99.5**).
- **Diagnostics** — Explains how many test points are even **eligible** for **`zero_day`** and reports **`Hybrid rows labeled zero_day`**.

**Style:** Section banners, optional Colab mount, linear execution, matplotlib/seaborn at the end.

---

## 2. Why `zero_day` Often Looks “Broken” (It Usually Is Not a Code Bug)

Understanding this avoids chasing false bugs:

1. **`zero_day` is not a dataset class.** Car-Hacking only has **Normal** + four **known** attacks. There is no ground-truth “unknown” label to score against.

2. **`zero_day` only fires on the Stage 2 path:** Stage 1 must predict **Normal**. If Stage 1 correctly predicts **DoS / Fuzzy / Gear / RPM**, the hybrid label is that attack — **never** **`zero_day`**.

3. **If Stage 1 recall on attacks is high**, very few **true attacks** are **false negatives** (predicted Normal). Then **almost nobody** is eligible for **`zero_day`**, even if Stage 2 is perfect.

4. **Even among Stage 1 false negatives**, the AAE must assign a **score above threshold**. Attacks that look like normal traffic in the **16-D feature space** can reconstruct well → low score → stay **Normal** in the hybrid.

5. **A very high percentile (e.g. 99.5)** sets a **high** threshold → only the most extreme scores flag → fewer **`zero_day`** and fewer false alarms on benign traffic.

Pipeline-02 prints a block **“STAGE 2 — zero_day eligibility”** that reports: how many test rows have Stage 1 = Normal, how many of those are **true attacks** (FNs), what fraction of those FNs exceed the threshold, and how many hybrid rows end as **`zero_day`**.

---

## 3. Execution Flow (Top to Bottom)

```
Optional Colab mount
    ↓
Imports, seeds, hyperparameters (Stage 2 + fake probes), device
    ↓
Classes + helpers (incl. compute_recon_lat_stats, combined_scores)
    ↓
if data_path missing → print error and stop
else → Stage 1 → Stage 2 → score stats → threshold → synthetic QA → eligibility prints
    → hybrid labels → metrics → plots
```

---

## 4. Configuration Blocks

### 4.1 Colab vs local

Same as pipeline-01: tries **`google.colab.drive`**, otherwise **`data_path = r"9) Car-Hacking Dataset"`**.

### 4.2 Reproducibility

**`RANDOM_STATE = 42`** for NumPy and PyTorch; **`uniform_fake_features`** uses the same seed when no RNG is passed.

### 4.3 Stage 2 tunables (pipeline-02 defaults differ from 01)

| Name | Default (02) | Role |
|------|----------------|------|
| `STAGE2_NORMAL_PERCENTILE` | **`97.0`** | Threshold = this percentile of **calibration** pipeline-normal scores (`true Normal` ∧ Stage 1 `Normal`). Lower → lower threshold → more flags (**more `zero_day`**, more benign FPs). |
| `STAGE2_SCORE_STATS` | **`"train_normal"`** | **`train_normal`**: `recon_mean/std`, `lat_mean/std` from **AAE training normals (`Xn_tr`)**. **`cal_normal`**: stats from the same **cal** pipeline-normal rows (closer to pipeline-01’s older behavior). Training-normal stats usually spread z-scores so **non-normal** points can separate better. |
| `STAGE2_SCORE_AGG` | **`"max"`** | **`max`**: `score = max(rs, ls)` with recon/latent absolute z-scores `rs`, `ls`. **`blend`**: `(1 - LATENT_WEIGHT) * rs + LATENT_WEIGHT * ls`. **`max`** fires if **either** cue is large (helps when one component dominates). |

Other Stage 2 knobs (`AAE_*`, `LATENT_DIM`, `LATENT_WEIGHT`) match the shared setup; **`LATENT_WEIGHT`** affects **`blend`** only, not **`max`**.

### 4.4 Tunables — synthetic probes only

| Name | Default | Role |
|------|---------|------|
| `NUM_FAKE_SAMPLES` | `8000` | Decoder / uniform probe count |
| `FAKE_LATENT_STD` | `2.5` | **`z`** scale before **`Decoder(z)`** |
| `INCLUDE_UNIFORM_FAKE` | `True` | Also score uniform **`[0,1]^d`** fakes |

---

## 5. Scoring Implementation

### 5.1 `compute_recon_lat_stats(enc, dec, X_np)`

Runs the frozen **encoder/decoder** on **`X_np`** in batches; returns **`recon_mean`, `recon_std`, `lat_mean`, `lat_std`** for reconstruction MSE and **`||z||`**.

### 5.2 `combined_scores(...)`

Batched forward passes (avoids huge GPU tensors). Per sample:

- **`rs`** = \|recon MSE − `recon_mean`\| / `recon_std`
- **`ls`** = \|‖**z**‖ − `lat_mean`\| / `lat_std`

Then **`max(rs, ls)`** or weighted **`blend`** per **`STAGE2_SCORE_AGG`**.

### 5.3 Threshold

**`val_scores`** = scores on **`X_cal`** with **`true Normal` ∧ Stage 1 Normal`** (same operational path as 01).  

**`threshold`** = **`np.percentile(val_scores, STAGE2_NORMAL_PERCENTILE)`**.

---

## 6. Synthetic QA

Same idea as before: decoder fakes, optional uniform fakes, balanced **real_normal_cal** vs **decoder_fake** table, extra histogram. All use the **same** stats and aggregation as real traffic.

---

## 7. Test-Time Hybrid Labels

1. Stage 1 ≠ Normal → named attack from **`LabelEncoder`**.
2. Stage 1 = Normal and **score > threshold** → **`zero_day`**.
3. Else → **`Normal`**.

---

## 8. Metrics and Plots

- Stage 1 report + confusion; synthetic probe summary; Stage 2 gated table; hybrid report; binary hybrid stats.
- Plots: Stage 1 loss, Stage 1 CM, Stage 2 AE curve, test Normal vs Attack score histogram, **cal vs decoder vs uniform** histogram, gated CM (if **`n_s2 > 0`**), hybrid CM.

---

## 9. Tuning Guide If You Need More `zero_day`

Trade-off: more **`zero_day`** usually means **more false alarms** on true normals that Stage 1 passes through.

1. **Lower** **`STAGE2_NORMAL_PERCENTILE`** (e.g. **95** … **90**).
2. Keep **`STAGE2_SCORE_AGG = "max"`** or try **`blend`** with **`LATENT_WEIGHT`** if one branch is noisy.
3. **`STAGE2_SCORE_STATS = "train_normal"`** (default) often helps separation vs **`cal_normal`**.
4. **Remember the bottleneck:** if **Stage 1 FN count on test is 0**, no true attack can become **`zero_day`** regardless of Stage 2.

---

## 10. Comparison to `Endsem-imp-pipeline-01.py`

| Aspect | Pipeline-01 | Pipeline-02 |
|--------|-------------|-------------|
| Threshold percentile | Default **99.5** in script | Default **97** + documented rationale |
| Score stats | From **cal** pipeline normals | Default **train normals (`Xn_tr`)** |
| Score aggregation | Weighted blend only | **`max`** or **`blend`** |
| `combined_scores` | Single GPU tensor | Batched |
| Synthetic probes | No | Yes |
| `zero_day` explanation | — | **Eligibility block** + header comment |

---

## 11. How to Run

```bash
cd Endsem_prep   # or project root where data_path resolves
python Endsem-imp-pipeline-02.py
```

Ensure the **Car-Hacking** dataset is visible at **`data_path`**.

---

## 12. File Reference (Quick Map)

| Region | Content |
|--------|---------|
| Header + tunables | Colab, paths, **`STAGE2_*`**, fake probes, `device` |
| Models | Stage 1, AAE, **`FakeSignalGenerator`** |
| Helpers | **`load_full_dataframe`**, **`compute_recon_lat_stats`**, **`combined_scores`**, trainers |
| Main | Train → threshold → QA → **eligibility** → hybrid → plots |

---

*Companion documentation for `Endsem_prep/Endsem-imp-pipeline-02.py`; pair with `Endsem-imp-pipeline-01.md` for the shared baseline.*
