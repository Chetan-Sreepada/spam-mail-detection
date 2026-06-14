# Spam Mail Detection

A machine learning model that detects spam messages using Logistic Regression and TF-IDF feature extraction. Deployed as an interactive Streamlit web application.

## Overview

Email spam costs businesses time and money. This project builds a binary classifier to automatically distinguish between spam (unwanted/promotional) and ham (legitimate) messages. The model is trained on SMS text data and provides real-time predictions with confidence scores.

Users can enter any message and receive an immediate spam/ham classification with probability breakdown.

## Features

- Real-time spam/ham prediction
- TF-IDF text vectorization
- Logistic Regression classifier
- Confidence score display (spam vs ham probability)
- Interactive Streamlit web interface
- Pre-trained model serialized with pickle

## Tech Stack

### Frontend / Interface
- Streamlit

### Machine Learning
- scikit-learn (LogisticRegression, TfidfVectorizer)
- pandas
- numpy

### Development
- Python 3.x
- Jupyter Notebook

## System Architecture

User Input Message
↓
Streamlit UI (app.py)
↓
Load Vectorizer (vectorizer.pkl) → Transform Text to TF-IDF Features
↓
Load Model (spam_model.pkl) → LogisticRegression.predict()
↓
Display Prediction (Spam/Ham) + Confidence Scores

## Dataset and Data Processing

**Dataset source:** SMS Spam Collection dataset (mail_data.csv)

**Dataset size:** 5,572 messages
- Ham (legitimate): ~4,800 messages
- Spam: ~770 messages

**Features used:**
- Message text (raw input)

**Processing steps:**
1. Load CSV and replace null values with empty strings
2. Label encoding: spam = 0, ham = 1
3. Train/test split: 80/20 with random_state=3
4. TF-IDF vectorization with:
   - min_df=1 (include all terms)
   - stop_words='english' (remove common words)
   - lowercase=True (case normalization)

**Why TF-IDF:** Converts raw text to numerical features while downweighting common words and highlighting discriminative terms.

## Methodology

**Algorithm:** Logistic Regression from scikit-learn.

**Why Logistic Regression:**
- Simple, interpretable, and fast
- Works well for high-dimensional sparse data (TF-IDF outputs)
- Provides probability estimates (not just binary predictions)
- Low inference latency for real-time applications

**Training setup:**
- Test size: 20%
- Random state: 3 (reproducibility)
- Default hyperparameters (C=1.0, solver='lbfgs')
- Max iterations: 100

## Results

**Model:** LogisticRegression()

| Dataset | Accuracy |
|---------|----------|
| Training | 96.77% |
| Test | 96.68% |

The near-identical training and test accuracy indicates minimal overfitting. The model generalizes well to unseen messages.

## Repository Structure
