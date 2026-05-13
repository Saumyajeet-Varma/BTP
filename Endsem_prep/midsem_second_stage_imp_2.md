# `midsem_second_stage_imp_2.py` — Stage 2 with F-beta tuning and a Stage-1 softmax gate

This script **extends** `midsem_second_stage_imp.py`: same data path, windowing, **Stage 1** (5-class CNN–LSTM), bottleneck **AE** with **combined score** \(\max(z_\text{mse}, z_\ell)\), **GAN** training, and hybrid **test + GAN `zero_day`** evaluation. The change is **how Stage 2 chooses an operating point** and **how it combines AE evidence with Stage-1 confidence**.

---

## 1. What stays the same (see `midsem_second_stage_imp.md`)

- Train/val/test splits, scaler, Stage 1 fit on `X_fit`, AE on fit normals, GAN on fit normals.  
- Combined anomaly score from train-normal MSE and \(\|z\|\) z-scores.  
- Hybrid routing: Stage 1 ≠ Normal → known attack; Stage 1 = Normal → Stage 2 decides Normal vs `zero_day`.  
- Validation negatives/positives built from **val true Normal + val GAN**, restricted to **Stage 1 = Normal** (with the same fallbacks as the parent script).

---

## 2. What changes in `imp_2`

### 2.1 Threshold tuning: F-beta instead of F1

On the validation binary task (Normal = 0, GAN = 1), the parent script maximizes **F1**. Here the objective is **`fbeta_score` with `FBETA_TUNING = 0.5`** (F\(_{0.5}\)), which **weights precision more than recall** for the positive class (`zero_day`). That targets fewer **Normal → `zero_day` false positives** at the cost of some recall when the score-only boundary is noisy.

### 2.2 Minimum recall on synthetic positives

Among \((T, \text{CAP})\) pairs, the tuner **prefers** those with **GAN recall ≥ `MIN_RECALL_ZERO_DAY_VAL`** (default **0.70**) on validation. If **no** pair meets that recall, it falls back to the best **unconstrained** F\(_{0.5}\) and prints that the minimum recall was not met.

### 2.3 Joint grid over \(T\) and CAP (Stage-1 gate)

**Stage-2 positive** (call it `zero_day`) is only allowed when **both** hold:

1. **AE:** `score > T` (same combined score as in `midsem_second_stage_imp.py`).  
2. **Gate:** `p(\text{Normal} \mid \text{Stage 1}) \leq \text{CAP}`.

Here `p(Normal)` is the **softmax probability for the Normal class** on the **same** window. Intuition: if Stage 1 is **very sure** Normal, many high AE scores are **spurious** for our purposes; requiring **moderate uncertainty** (Normal prob not too high) reduces **false `zero_day`** on benign traffic. GAN windows that still look “Normal” to Stage 1 but are odd in latent/recon space often have **lower** `p(Normal)`, so they can still be caught.

**Tuning:** `T` is searched on `THRESHOLD_GRID_POINTS` (default 400) linear steps between min and max of the concatenated val scores; **CAP** is searched on **`CAP_GRID_POINTS`** (default 14) linear steps in **[0.82, 0.995]**. The chosen pair maximizes **F\(_{0.5}\)** (with the recall constraint when feasible).

### 2.4 Outputs (do not overwrite `imp` figures)

| File |
|------|
| `midsecond2_cm_stage1.png` |
| `midsecond2_cm_stage2.png` |
| `midsecond2_cm_final_hybrid.png` |

---

## 3. New / adjusted hyperparameters (header)

| Name | Default | Role |
|------|---------|------|
| `FBETA_TUNING` | `0.5` | \(\beta\) in F\(_\beta\); &lt; 1 favors precision on `zero_day`. |
| `MIN_RECALL_ZERO_DAY_VAL` | `0.70` | Minimum validation recall on GAN positives when picking \((T,\text{CAP})\). |
| `CAP_GRID_POINTS` | `14` | Grid size for CAP in [0.82, 0.995]. |

`THRESHOLD_GRID_POINTS` is unchanged (400). Other AE/GAN/Stage-1 constants match the parent script unless you edit them locally.

---

## 4. How to read metrics after a run

The script logs **validation** precision/recall/F\(_{0.5}\)/F1 for the binary tuning set and prints **test** Stage 1, Stage 2 (Normal vs `zero_day` on S1=Normal only), and **6-class hybrid** blocks. Compare **`zero_day` precision** and **`Normal` recall** to `midsem_second_stage_imp.py`: **imp_2** is intended to **raise precision** and overall **macro precision** when the gate removes confident-Normal false alarms; **`zero_day` recall** may drop slightly if CAP is tight—then try lowering `MIN_RECALL_ZERO_DAY_VAL`, increasing `CAP_GRID_POINTS` / widening the CAP range, or setting `FBETA_TUNING` closer to 1.

---

## 5. Relation to `midsem_second_stage_imp.py`

| Aspect | `midsem_second_stage_imp.py` | `midsem_second_stage_imp_2.py` |
|--------|------------------------------|--------------------------------|
| Val objective | Max F1 on score &gt; T | Max F\(_{0.5}\) on (score &gt; T) ∧ (p(Normal) ≤ CAP), with min GAN recall |
| Test rule (S1=Normal) | `zero_day` iff score &gt; T | `zero_day` iff score &gt; T **and** p(Normal) ≤ CAP |
| Figures | `midsecond_cm_*.png` | `midsecond2_cm_*.png` |

Run **`midsem_second_stage_imp_2.py`** the same way as the parent (local `data_path` or Colab drive mount at the top of the file).
