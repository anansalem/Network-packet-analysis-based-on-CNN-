# Network-packet-analysis-based-on-CNN-
Intrusion Detection Systems (IDS) aim to identify malicious activities in network traffic. Traditional rule-based systems fail to detect new or unknown attacks. Therefore, this project addresses the problem using machine learning, specifically deep learning, to automatically classify network traffic as either normal or attack.



# Intrusion Detection System using LSTM (Supervised Learning)

## Project Overview
This project implements an Intrusion Detection System (IDS) using a supervised Long Short-Term Memory (LSTM) neural network. The system classifies network traffic sequences as either **Normal** or **Attack** based on temporal patterns learned from labeled data.

The UNSW-NB15 dataset is used as a benchmark dataset for training and evaluation.

---

## Objectives
- Detect malicious network traffic with high accuracy
- Model temporal dependencies in network flows
- Compare anomaly-based and supervised learning approaches
- Achieve high accuracy and F1-score suitable for real-world IDS scenarios

---

## Dataset
- **Dataset:** UNSW-NB15
- **Total flows:** 82,332
- **Features:** 42 numerical features
- **Labels:**
  - `0` → Normal
  - `1` → Attack

### Preprocessing Steps
1. Encoding categorical features (`proto`, `service`, `state`)
2. Feature normalization using `StandardScaler`
3. Conversion of flows into sequences of length 10
4. Train / Validation / Test split

---

---

## Model Architecture
- **Input:** Sequence of 10 network flows
- **LSTM Layer:** Captures temporal dependencies
- **Fully Connected Layer:** Binary classification
- **Activation:** Sigmoid
- **Loss Function:** Binary Cross Entropy (BCELoss)
- **Optimizer:** Adam

---

## Training Details
- Epochs: 20
- Batch size: 64
- Learning rate: 0.001
- Training strategy: Supervised learning using labeled normal and attack traffic

---

## Evaluation Metrics
The system is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics provide a comprehensive evaluation of IDS performance.

---


### Validation Set
- Accuracy: **99.27%**
- F1-score: **99.33%**

### Test Set
- Accuracy: **99.43%**
- F1-score: **99.48%**

Confusion Matrix (Test Set):


