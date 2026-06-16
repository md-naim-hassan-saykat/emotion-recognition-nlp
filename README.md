# Emotion Recognition using NLP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/md-naim-hassan-saykat/emotion-recognition-nlp/blob/main/notebooks/emotion_recognition.ipynb)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/get-started/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project implements **emotion recognition from text** using both **transformer-based deep learning (BERT)** and **classical machine learning models** (SVM, Random Forest, Logistic Regression).  
An **ensemble approach** is also tested by combining all models with majority voting.

---

## Project Overview
- **Task:** Multi-class emotion recognition from text  
- **Dataset:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (Hugging Face)  
- **Models Implemented:**  
  - BERT (transformer-based fine-tuned model)  
  - SVM (Support Vector Machine with TF-IDF features)  
  - Random Forest  
  - Logistic Regression  
  - Ensemble (BERT + SVM + RF + LR via majority voting)  
- **Metrics:** Classification Report (Precision, Recall, F1-Score)  
- **Goal:** Compare deep learning vs. classical ML for emotion recognition  

---

## Repository Structure
emotion-recognition-nlp/
│
├── notebooks/
│   └── emotion_recognition.ipynb     # Main Jupyter Notebook
│
├── docs/
│   ├── report.tex                    # LaTeX project report 
│   ├── report.pdf                    # Compiled project report
│   └── references.bib                # References for the report
│
├── requirements.txt
├── README.md
└── .gitignore
---

## Getting Started

### Clone the repo
git clone https://github.com/md-naim-hassan-saykat/emotion-recognition-nlp.git
cd emotion-recognition-nlp

## Install dependencies
pip install -r requirements.txt

## Run the notebook
Open notebooks/emotion_recognition.ipynb and run all cells to:
- Preprocess dataset
- Train BERT and classical models
- Evaluate performance
- Compare results

---

## Results
BERT (Transformer-based model)
- Best performing model with highest F1-score
- Handles semantic context well

---

## Classical ML Models (TF-IDF features)
- SVM, Random Forest, and Logistic Regression show competitive but lower performance than BERT

---

## Ensemble
- Combines all models using majority voting
- Achieves balanced precision/recall across classes

---

## References
	1.	Vaswani et al., Attention is All You Need, NeurIPS 2017.
	2.	Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019.
	3.	HuggingFace Datasets: dair-ai/emotion.
	4.	Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 2011.


# Author

 **Md Naim Hassan Saykat**  
*MSc in Artificial Intelligence, Université Paris-Saclay*  

[LinkedIn](https://www.linkedin.com/in/md-naim-hassan-saykat/)  
[GitHub](https://github.com/md-naim-hassan-saykat)  
[Academic Email](mailto:md-naim-hassan.saykat@universite-paris-saclay.fr)  
[Personal Email](mailto:mdnaimhassansaykat@gmail.com) 
