import sys
import torch
import joblib

# -----------------------------
# Load model
# -----------------------------
tokenizer, model, clf = joblib.load("model/phishing_detector.pkl")

# -----------------------------
# Input email text
# -----------------------------
email_text = sys.argv[1] if len(sys.argv) > 1 else input("Enter email text: ")

# -----------------------------
# Generate embedding
# -----------------------------
inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[:,0,:].squeeze().numpy()

# -----------------------------
# Predict
# -----------------------------
prediction = clf.predict([cls_emb])[0]
probability = clf.predict_proba([cls_emb])[0]

label = "PHISHING 🚨 (High Risk)" if prediction==1 else "SAFE ✅"
confidence = max(probability) * 100

print("\nResult:")
print("------")
print(f"Verdict   : {label}")
print(f"Confidence: {confidence:.2f}%")
