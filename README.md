# Emotion Recognition from Tweets

This project focuses on recognizing human emotions from short texts (tweets) using both traditional machine learning and deep learning models. It was developed as part of a Hands-on Natural Language Processing (NLP) course project at Université Paris-Saclay.

## Team Members

- Md Naim Hassan Saykat: SVM, Logistic Regression, Random Forest, BERT, Ensamble, Included in Traditional Models and BERT Parts of the Report
- Aloïs Vincent: Data Exploration and Visualization, References, SVM, Logistic Regression, Random Forest, Notebooks Merging, Presentation 
- Marija Brkic: Data and Dataset Analysis, State-of-the-art Models Research, Vectorization and Visualization, Convolutional Neural Network, Report

## Project Files
 
[Jupyter Notebook](./emotion_recognition_code.ipynb)
[Project Report](./emotion_recognition_report.pdf) 
[Presentation Slides](./emotion_recognition_presentation.pdf)

## Objective

The goal is to predict emotions expressed in tweets using a variety of models and compare their effectiveness. The project covers traditional classifiers, convolutional neural networks (CNN), and fine-tuned transformer models such as BERT.

## Dataset

- **Dataset Source**: [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)  
- **Number of tweets**: 416,809  
- **Emotion Classes**:
  - Joy
  - Sadness
  - Anger
  - Love
  - Fear
  - Surprise

## Models Implemented

| Model                         | Accuracy |
|------------------------------|----------|
| Logistic Regression          | 86%      |
| Random Forest                | 87%      |
| Support Vector Machine       | 89%      |
| Convolutional Neural Network | 89%      |
| BERT (fine-tuned)            | 92%      |
| Ensemble (BERT + SVM + RF + LR) | 90%   |

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow, Keras
- PyTorch
- HuggingFace Transformers
- Matplotlib, Seaborn
- NLTK
- Jupyter Notebook

## Evaluation Metrics

We used the following metrics to evaluate model performance:

- **Accuracy**: Overall correctness
- **Precision**: Correct positive predictions / Total predicted positives
- **Recall**: Correct positive predictions / Total actual positives
- **F1-Score**: Harmonic mean of precision and recall

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 86% | 0.87 | 0.85 | 0.86 |
| Random Forest       | 87% | 0.88 | 0.86 | 0.87 |
| SVM                 | 89% | 0.90 | 0.88 | 0.89 |
| CNN                 | 89% | 0.89 | 0.88 | 0.88 |
| BERT                | 92% | 0.93 | 0.91 | 0.92 |
| Ensemble            | 90% | 0.91 | 0.89 | 0.90 |

> Note: BERT gave the highest F1-score across all emotion classes. Surprise had the lowest scores due to class imbalance.

## Disclaimer

This project is shared for academic demonstration purposes only.  
Reuse, reproduction, or distribution is not permitted without explicit permission.
