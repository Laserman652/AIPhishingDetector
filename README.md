<!-- # phishing-detector
This project is a simple command-line tool that analyzes email text and predicts whether the email is Phishing, Suspicious, or Legitimate. The goal of the project is to demonstrate how machine learning and modern NLP embeddings can be used to detect phishing emails in a practical, lightweight, and explainable way. -->
# AI Powered Phishing Email Detector

## Overview
A command-line tool that analyzes email text and predicts whether an email is Phishing, Suspicious, or Legitimate.
This project demonstrates how machine learning and modern NLP embeddings can be used to detect phishing emails in a practical, lightweight, and explainable way.

## What This Project Does
- Takes raw email text as input
- Converts the text into **semantic embeddings using DistilBERT**
- Uses a **Logistic Regression classifier** to predict phishing probability
- Outputs a **human-friendly verdict** with confidence

## Verdictsgit add .

- **LEGIT** --> (Safe email)
- **SUSPICIOUS** --> (Needs review)
- **PHISHING** --> (High risk)

## Tech Stack Used
- Python
- PyTorch
- HuggingFace Transformers (DistilBERT)
- Scikit-learn
- Pandas
- Joblib

## Why This Approach
Instead of using basic keyword matching or TF-IDF alone, this project uses **DistilBERT embeddings** to capture the **intent and context** of email text (urgency, threats, authority abuse).  
The **Logistic Regression classifier** keeps the system:
- Interpretable
- Lightweight
- Easy to debug
- Interview-friendly

## Project Structure
```
phishing-detector/
├── data/
│   └── emails/
│       └── combined.csv  # Dataset with email text & labels
├── model/
│   └── phishing_model.pkl  # Trained DistilBERT + Logistic Regression
├── train.py  # Training script
├── test.py   # CLI testing script
├── requirements.txt  # Python dependencies
└── README.md

```
## Python environment setup

```
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
## Dataset Format
The CSV dataset must have two columns:
| Column | Description                      |
| ------ | -------------------------------- |
| text   | Email content                    |
| label  | 1 for phishing, 0 for legitimate |

## sample dataset
```
text,label
"Please verify your account immediately",1
"Team meeting at 5 PM today",0

```
## How Training Works
1. Load phishing email dataset
2. Tokenize emails using **DistilBERT tokenizer**
3. Generate embeddings from DistilBERT **CLS token**
4. Train **Logistic Regression** on embeddings
5. Evaluate accuracy
6. Save trained model using **Joblib**

## How Testing Works
1. Load saved model
2. Accept email text
3. Generate embedding using the same BERT model
4. Predict phishing probability
5. Convert probability into verdict

## Example Output
```
**Email:** Please verify your account immediately  
**Verdict:** PHISHING (High Risk)  
**Phishing Probability:** 99.93%

**Email:** Team meeting at 5 PM today  
**Verdict:** LEGIT  
**Phishing Probability:** 0.37%
```

## How to Run
1. Train the model:  
   ```
   python train.py
   ```
2. Test emails:  
   ```
   python test.py
   ```
## Model Evaluation Section
```
| Metric    | Score |
| --------- | ----- |
| Accuracy  | 96.4% |
| Precision | 95.1% |
| Recall    | 97.2% |
| F1-score  | 96.1% |
```
## Verdict Thresholds
| Probability | Verdict    |
| ----------- | ---------- |
| 0 – 50%     | LEGIT      |
| 50 – 90%    | SUSPICIOUS |
| > 90%       | PHISHING   |


## Limitations
- Model is only as good as the dataset
- Some legitimate transactional emails may appear suspicious
- Does not analyze links, headers, or sender metadata

## Future Improvements
- Compare with TF-IDF baseline
- Add URL and domain analysis
- Add email header inspection
- Add simple web interface
- Add LLM-based explanation (optional)

