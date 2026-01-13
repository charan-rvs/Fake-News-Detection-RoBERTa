# Fake News Detection Using RoBERTa

## Overview
This project implements a **Fake News Detection system** using the **RoBERTa transformer model** to classify news articles as **Fake** or **Real**.  
The model is fine-tuned on a large, real-world dataset of news articles and achieves **~99% accuracy**, making it suitable for research, analysis, and practical NLP applications.

---

## Dataset
- **Source:** Kaggle – Fake and Real News Dataset
- **Files:**
  - `Fake.csv` – Fake news articles
  - `True.csv` – Real news articles
- **Total Samples:** ~44,000 articles
- **Features:**
  - Title
  - Text
  - Subject
  - Date
- **Classes:**  
  - `0` → Fake News  
  - `1` → Real News

---

## Model Architecture
- **Base Model:** RoBERTa (Pretrained Transformer)
- **Framework:** Hugging Face Transformers
- **Training Approach:**
  - Tokenization using RoBERTa tokenizer
  - Fine-tuning on downstream classification task
  - AdamW optimizer with weight decay
  - Evaluation per epoch

---

## Results
| Metric | Score |
|------|------|
| Accuracy | **99.9%** |
| Precision | **1.00** |
| Recall | **1.00** |
| F1-Score | **1.00** |

- Stable convergence  
- High confidence predictions  
- No overfitting observed

---

## Visualizations Included
- Training vs Validation Loss
- Accuracy Progression
- Precision / Recall / F1 Comparison
- Prediction Confidence Distribution
- Confusion Matrix

---

## Features
- Transformer-based NLP model
- High-accuracy fake news classification
- Clean preprocessing & tokenization
- Detailed evaluation metrics
- Resume & production-ready structure

---

## Tech Stack
- Python
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- NumPy
- Matplotlib

---
