
# CANGuard (arXiv:2603.25763v1) — Summary for the BTP

**Title:** CANGuard: A Spatio-Temporal CNN-GRU-Attention Hybrid Architecture for Intrusion Detection in In-Vehicle CAN Networks  

**Venue / ID:** arXiv:2603.25763v1 [cs.CR], 26 Mar 2026  

**Problem:** Connected vehicles (Internet of Vehicles) expose the **CAN bus** to **Denial-of-Service (DoS)** and **spoofing** attacks. CAN lacks built-in authentication and encryption, so **intrusion detection** is critical for safety.

**Proposed method — CANGuard:** A **hybrid deep model** that learns **spatial** patterns with **1D convolutions**, **temporal** dependencies with **stacked bidirectional GRUs**, and **feature/time focus** with an **attention** pooling step, followed by **fully connected** layers for **multi-class** prediction.

**Dataset:** **CICIoV2024** — large IoV-oriented CAN intrusion dataset (~1.4M samples, 12 input features after exclusions). **Six classes:** **BENIGN**, **DoS**, and four **spoofing** targets (**GAS**, **RPM**, **SPEED**, **STEERING WHEEL**).

**Preprocessing (high level):** Remove duplicates; drop non-predictive columns (e.g. ID / meta where applicable); build **sliding windows** of length \(T\) with label at the end of the window; mitigate imbalance with **BorderlineSMOTE** on the **training** portion (flatten window, oversample, reshape); **Z-score** normalize features; use **class weights**.

**Training:** **Adam**, learning rate **0.001**, **categorical cross-entropy**, batch size **64**, up to **50** epochs with **early stopping** (patience **10**); **dropout 0.3**, **L2** regularization (\(\lambda = 0.001\)); **batch normalization**; **gradient clipping**; GPU training.

**Interpretability:** **SHAP** analysis on payload-related inputs (**DATA0–DATA7**), showing which bytes most influence benign vs malicious predictions (paper highlights **DATA4** / **DATA5** among the strongest).

**Evaluation:** Strong **accuracy / precision / recall / F1** on CICIoV2024; **ablation** showing contribution of **CNN only**, **GRU only**, **CNN+GRU without attention**, and **full CNN+GRU+attention**; comparison table against prior ML/DL works (note: cross-dataset comparisons are approximate).

**Limitations (stated in the paper):** **Offline** evaluation on a **single** benchmark; **no** real-time on-vehicle deployment study; **no** **adversarial robustness** analysis. **Future work** mentions **cross-dataset** tests, **online** CAN monitoring, and **adversarial** settings.

**Takeaway for extensions:** Pairing this **supervised** architecture with a **second stage** trained only on **normal** traffic (e.g. autoencoder / one-class model) on the subset classified as benign is a natural way to address **open-set** or **unknown** attack behavior — beyond the paper’s closed-set multiclass setup.

**Reference:** Rakib Hossain Sajib et al., *CANGuard: A Spatio-Temporal CNN-GRU-Attention Hybrid Architecture for Intrusion Detection in In-Vehicle CAN Networks*, arXiv:2603.25763v1.
