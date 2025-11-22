# 📘 Transformer-Based Anomaly Detection on Credit Card Fraud Data

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This project implements a **Transformer Autoencoder** with **contrastive learning** and **adversarial training** to detect fraudulent credit card transactions.  
The model is trained on the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and reframed as a multivariate time-series anomaly detection problem using sliding windows.

---

## 📂 1. Dataset Used

- **Transactions:** 284,807  
- **Input variables:**  
  - V1–V28 (PCA features)  
  - Amount  
  - Time  
- **Imbalance:**  
  - Fraudulent transactions: 492  
  - Normal transactions: 284,315  
  - Fraud ratio: 0.172%

After preprocessing into sliding windows:

- Windows: `(284800, 8, 29)`  
- Anomalous windows: 3,589 out of 284,800  
- Each window contains 8 consecutive transactions labeled `1` if any transaction inside is fraudulent.

---

## 🔧 2. Preprocessing Steps

1. **Load and order dataset**  
   - Sort transactions chronologically by `Time` to prevent data leakage.

2. **Feature selection**  
   - Used: V1–V28, Amount  
   - Excluded: Class (label), Time (ordering only)

3. **Standardization**  
   - All features scaled using `StandardScaler()`

4. **Sliding window creation**  
   - Window length: 8  
   - Step size: 1  
   - Output shapes:  
     - `X shape: (284800, 8, 29)`  
     - `y shape: (284800,)`

5. **Window-level labeling**  
   - `label = 1` if **any** transaction in the window is fraudulent  

This converts the dataset into a **sequence anomaly detection task**.

---

## 🧠 3. Model Architecture and Components

- **Encoder:**  
  - Linear projection 29 → 128  
  - PositionalEncoding  
  - 2-layer TransformerEncoder  
  - AdaptiveAvgPool1D  
  - Latent vector 128 → 64

- **Decoder:**  
  - Linear projection 64 → 128  
  - 2-layer TransformerEncoder  
  - Output projection 128 → 29

- **Contrastive Learning Module:**  
  - Encourages clustering of normal latent vectors

- **Adversarial Components:**  
  - Discriminator and Generator losses  
  - Refines model sensitivity to anomalies

- **Loss Terms:**  
  - Reconstruction loss  
  - Contrastive loss  
  - Adversarial loss  
  - Weighted combination for total loss

---

## 🎓 4. Training Procedure

- **Epochs:** 20  
- **Losses:** Combined reconstruction, contrastive, and adversarial losses  

**Example outputs:**

- Epoch 1/20 | recon: 0.8539 | contrast: 3.6985 | disc: 1.3166 | gen: 0.7520 | val_auc: 0.8944
- ...
- Epoch 20/20 | recon: 0.4995 | contrast: 3.6390 | disc: 1.3259 | gen: 0.8348 | val_auc: 0.7775


**Observations:**

- Reconstruction loss steadily decreases  
- Contrastive loss stable  
- Adversarial losses oscillate (expected)  
- Validation AUC converges around 0.77

**Data splits:**

- Train: 227,840 windows  
- Valid: 28,480 windows  
- Test : 28,480 windows

---

## 📊 5. Evaluation Metrics

| Metric   | Score  |
|----------|--------|
| ROC AUC  | 0.7670 |
| PR AUC   | 0.0836 |

> Note: PR AUC is low due to extreme class imbalance.

---

## 🚨 6. Results (Anomaly Detection Demonstration)

**Top-k anomalous windows by reconstruction error:**

| Index  | Score  | Label |
|--------|--------|-------|
| 5181   | 24.66  | 0     |
| 2105   | 20.44  | 1     |
| 17945  | 19.75  | 1     |
| 19405  | 18.68  | 0     |
| 21536  | 14.87  | 0     |
| 7281   | 13.77  | 1     |

**Analysis:**

- 3 of the top 6 anomalies are true fraud windows  
- Model ranks anomalous patterns effectively  

**Curves generated:** ROC Curve (AUC ≈ 0.77), Precision–Recall Curve (AUC ≈ 0.084)

---

## ✔ 7. Correctness and Functionality

- **Preprocessing:** Standardization, time-based sorting, sliding window creation, correct labeling  
- **Model:** Transformer encoder & decoder, latent bottleneck, contrastive & adversarial modules  
- **Evaluation:** Reconstruction-based detection, ROC and PR metrics, visualizations included  

---

## ⭐ 8. Effectiveness of Results

**Strengths:**

- Captures temporal dependencies  
- Detects genuine fraud cases  
- ROC AUC ≈ 0.77 indicates strong ranking

**Limitations:**

- PR AUC ≈ 0.084 due to class imbalance  
- Some false positives, but still high-error patterns useful for investigation

**Use Cases:**

- Real-time fraud surveillance  
- Pre-filtering suspicious transactions  
- Ranking transactions for manual review

---

## 📦 9. Repository Structure

- ├─ DM_Project.ipynb # Main Jupyter Notebook
- ├─ Dockerfile # Docker environment setup
- └─ requirements.txt # Python dependencies


---

## ⚡ 10. Setup and Usage

### Using Python

```bash
# Clone the repository
git clone https://github.com/Aubakerzzz/DM_Projects
cd DM_projects

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Notebook
jupyter notebook DM_Project.ipynb


---

## 📦 9. Repository Structure

# Build Docker image
docker build -t transformer-fraud .

# Run container
docker run -p 8888:8888 transformer-fraud

# Open Jupyter Notebook in browser
http://localhost:8888


## 📌 11. Conclusion

This project demonstrates a modern anomaly detection pipeline using:

Transformer architectures

Sequence modeling

Contrastive representation learning

Adversarial refinement

Reconstruction error scoring

The results validate the model’s ability to detect and rank fraudulent behavior even in a highly imbalanced dataset.

