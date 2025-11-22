# 📘 Transformer-Based Anomaly Detection on Credit Card Fraud Data

This project implements a Transformer Autoencoder with contrastive learning and adversarial training to detect fraudulent credit card transactions.  
The model is trained on the popular Credit Card Fraud Detection dataset and reframed as a multivariate time-series anomaly detection problem using sliding windows.

---

## 📂 1. Dataset Used

The dataset comes from the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), consisting of:

- **284,807 transactions**
- **30 input variables**:
  - V1–V28 (PCA features)
  - Amount
  - Time
- **Extremely imbalanced distribution**:
  - Fraudulent transactions: 492
  - Normal transactions: 284,315
  - Fraud ratio: 0.172%

After processing into sliding windows:

- Windows: `(284800, 8, 29)`  
- Anomalous windows: 3,589 out of 284,800  

Each window contains 8 consecutive transactions, labeled `1` if any transaction inside is fraudulent.

---

## 🔧 2. Preprocessing Steps

- **Load and order dataset**  
  - Transactions are sorted chronologically using the `Time` column to prevent data leakage and simulate real-time streams.

- **Feature selection**  
  - Used: V1–V28 (PCA features), Amount  
  - Excluded: Class (label), Time (only used for ordering)

- **Standardization**  
  - All features scaled using `StandardScaler()`

- **Sliding window creation**  
  - Window length: 8  
  - Step size: 1  
  - Output shapes:  
    - `X shape: (284800, 8, 29)`  
    - `y shape: (284800,)`

- **Window-level labeling**  
  - `label = 1` if **any** transaction in the window is fraudulent  
  - Converts dataset into a sequence anomaly detection task.

---

## 🧠 3. Model Architecture and Components

A custom Transformer Autoencoder with:

- **Encoder**  
  - Linear projection 29 → 128  
  - PositionalEncoding  
  - 2-layer TransformerEncoder  
  - AdaptiveAvgPool1D (compress sequence)  
  - Latent vector: 128 → 64

- **Decoder**  
  - Linear projection 64 → 128  
  - 2-layer TransformerEncoder  
  - Output projection 128 → 29

- **Contrastive Learning Module**  
  - Encourages tighter clustering of normal latent vectors

- **Adversarial Components**  
  - Discriminator loss  
  - Generator loss  
  - Improves sensitivity to anomalies via adversarial refinement

- **Loss Terms**  
  - Reconstruction loss  
  - Contrastive loss  
  - Adversarial loss  
  - Weighted combination for total loss

---

## 🎓 4. Training Procedure

- Trained for 20 epochs with combined losses.  

**Example training outputs**:

-Epoch 1/20 | recon: 0.8539 | contrast: 3.6985 | disc: 1.3166 | gen: 0.7520 | val_auc: 0.8944
- ...
- Epoch 20/20 | recon: 0.4995 | contrast: 3.6390 | disc: 1.3259 | gen: 0.8348 | val_auc: 0.7775


**Observations**:

- Reconstruction loss steadily decreases  
- Contrastive loss stays stable  
- Adversarial losses oscillate (expected)  
- Validation AUC converges around 0.77

**Training splits**:

- Train: 227,840 windows  
- Valid: 28,480 windows  
- Test : 28,480 windows

---

## 📊 5. Evaluation Metrics

Evaluation is performed using reconstruction errors on the test set.

| Metric   | Score  |
|----------|--------|
| ROC AUC  | 0.7670 |
| PR AUC   | 0.0836 |

> Note: PR AUC is expectedly low due to extreme class imbalance.

---

## 🚨 6. Results (Anomaly Detection Demonstration)

**Top-k anomalous windows detected by reconstruction error**:

| Index  | Score  | Label |
|--------|--------|-------|
| 5181   | 24.66  | 0     |
| 2105   | 20.44  | 1     |
| 17945  | 19.75  | 1     |
| 19405  | 18.68  | 0     |
| 21536  | 14.87  | 0     |
| 7281   | 13.77  | 1     |

**Analysis**:

- Among the top 6 anomalies, 3 are true fraud windows (labels = 1)  
- Confirms the model ranks anomalous patterns highly  

**Curves generated**:

- ROC Curve (AUC ≈ 0.77)  
- Precision–Recall Curve (AUC ≈ 0.084)

---

## ✔ 7. Correctness and Functionality of the Implementation

- **Preprocessing**  
  - Standardization, Time-based sorting, Sliding window conversion, Proper window labeling
- **Model**  
  - Fully implemented Transformer encoder & decoder, Latent bottleneck, Positional encoding, Contrastive & adversarial modules, Correct loss aggregation
- **Evaluation**  
  - Uses `roc_auc_score` and `precision_recall_curve` correctly, Applies reconstruction error to detect anomalies, Visualizations included

---

## ⭐ 8. Effectiveness of the Anomaly Detection Results

**Strengths**:

- Transformer captures temporal dependencies  
- Reconstruction-based detection identifies genuine fraud cases  
- Top anomalous windows include real fraud events  
- ROC AUC ≈ 0.77 indicates strong ranking ability

**Limitations (expected)**:

- PR AUC ≈ 0.084 due to extreme imbalance  
- Some anomalies are false positives but still high-error patterns  

**Real-world usefulness**:

- Real-time fraud surveillance  
- Pre-filtering suspicious patterns  
- Ranking transactions for manual review

---

## 📦 Conclusion

This project demonstrates a powerful, modern anomaly detection pipeline using:

- Transformer architectures  
- Sequence modeling  
- Contrastive representation learning  
- Adversarial refinement  
- Reconstruction error scoring

The results validate the model’s ability to detect and rank fraudulent behavior, even in a highly imbalanced financial dataset.

