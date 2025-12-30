import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# CSV must have: text,label
# -----------------------------
data = pd.read_csv("data/emails.csv")  

X = data["text"].tolist()
y = data["label"].tolist()

# -----------------------------
# Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Load Transformer (DistilBERT)
# -----------------------------
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# -----------------------------
# Function: Get CLS embeddings
# -----------------------------
def get_embeddings(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:,0,:].squeeze().numpy()  # CLS token
            embeddings.append(cls_emb)
    return embeddings

# -----------------------------
# Generate Embeddings
# -----------------------------
print("[+] Generating embeddings for training data...")
X_train_emb = get_embeddings(X_train)
X_test_emb = get_embeddings(X_test)

# -----------------------------
# Train Classifier
# -----------------------------
print("[+] Training classifier...")
import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# CSV must have: text,label
# -----------------------------
data = pd.read_csv("data/emails.csv")  

X = data["text"].tolist()
y = data["label"].tolist()

# -----------------------------
# Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Load Transformer (DistilBERT)
# -----------------------------
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# -----------------------------
# Function: Get CLS embeddings
# -----------------------------
def get_embeddings(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:,0,:].squeeze().numpy()  # CLS token
            embeddings.append(cls_emb)
    return embeddings

# -----------------------------
# Generate Embeddings
# -----------------------------
print("[+] Generating embeddings for training data...")
X_train_emb = get_embeddings(X_train)
X_test_emb = get_embeddings(X_test)

# -----------------------------
# Train Classifier
# -----------------------------
print("[+] Training classifier...")
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_emb, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = clf.predict(X_test_emb)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump((tokenizer, model, clf), "model/phishing_detector.pkl")
print("\n[✓] Model saved to model/phishing_detector.pkl")
clf.fit(X_train_emb, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = clf.predict(X_test_emb)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump((tokenizer, model, clf), "model/phishing_detector.pkl")
print("\n[✓] Model saved to model/phishing_detector.pkl")
