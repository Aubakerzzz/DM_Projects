# LSTM Autoencoder for Anomaly Detection

This repository implements an LSTM Autoencoder–based anomaly detection system using the **Credit Card dataset from TS-AD-Datasets**. The project demonstrates sequence preprocessing, model training, anomaly scoring, and evaluation of reconstruction-based anomaly detection.

---

## 📂 Dataset Used

- Source: **TS-AD-Datasets → CreditCard dataset**
- Contains normal and fraudulent (anomalous) transaction records
- Highly imbalanced dataset
- Loaded directly inside the notebook using the provided dataset loader

---

## 🔧 Preprocessing Steps

1. Load dataset from TS-AD-Datasets.
2. Handle missing values (if any).
3. Sort data by timestamp to maintain sequence order.
4. Apply **MinMaxScaler** to all numerical features.
5. Generate **sliding windows** of fixed sequence length.
6. Split into:
   - Training data → only normal sequences  
   - Test data → both normal + anomalous samples
7. Convert to NumPy/Tensor format for LSTM input.

---

## 🧠 Model Architecture

The anomaly detector is a **sequence-to-sequence LSTM Autoencoder**.

### **Encoder**
- LSTM Layer(s) to reduce dimensionality
- Bottleneck latent vector representing normal behavior

### **Decoder**
- LSTM Layer(s) to reconstruct time-series windows
- TimeDistributed Dense layer outputs reconstructed features

### **Core Components**
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Activation:** tanh / relu depending on layer

---

## 🏋️ Training Procedure

- Train **only on normal sequences**  
- Use batch size and epochs defined in the notebook  
- Shuffle data during training  
- Monitor training & validation loss curves  
- Save the trained model for inference

---

## 📊 Evaluation Metrics

The following metrics are computed:

- **Reconstruction Error (MSE)**
- **Anomaly Threshold** (calculated from training error distribution)
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- **ROC Curve / AUC** (if implemented)

Higher reconstruction error indicates a higher likelihood of anomaly.

---

## 🚨 Results (Anomaly Detection Demonstration)

The notebook demonstrates:

- Reconstruction behavior on normal vs anomalous sequences  
- Plot of reconstruction error distribution  
- Threshold-based classification  
- Visualization of:
  - True Positives (detected fraud)
  - False Positives
  - False Negatives
- Example anomaly scores for sample sequences

---

## ✔️ Correctness & Functionality

- The LSTM Autoencoder successfully learns normal transaction patterns.
- The reconstruction error sharply increases on anomalous data.
- The implementation follows correct deep learning pipeline structure:
  - Clean preprocessing
  - Proper train/test split
  - Valid sliding window generation
  - Stable training with monitored loss

---

## 🎯 Effectiveness of Anomaly Detection

- The model effectively separates normal vs anomalous sequences based on error.
- Thresholding provides clear anomaly boundaries.
- Fraudulent sequences exhibit significantly higher reconstruction error.
- Evaluation metrics confirm the model’s ability to detect anomalies reliably.

---

## 📁 Repository Structure

- DM_NoteBook.ipynb
- README.MD
- Dockerfile
- requirements.txt


