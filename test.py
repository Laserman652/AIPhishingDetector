import joblib
import torch

# -----------------------------
# Load trained model
# -----------------------------
tokenizer, model, clf = joblib.load("model/phishing_detector.pkl")

# -----------------------------
# List of sample emails to test
# -----------------------------
test_emails = [
    "Please verify your account immediately",
    "Team meeting at 5 PM today. See you all.",
    "Invoice attached for your recent payment. Please review.",
    "Your Amazon order has shipped. Track here.",
    "Update billing information to avoid service interruption.",
    "Family dinner on Sunday. RSVP please.",
    "Confirm your password reset now or lose access.",
    "Reminder: Meeting rescheduled to 3 PM tomorrow.",
    "Some random newsletter text here"
]

# -----------------------------
# Function: Predict single email
# -----------------------------
def predict_email(email_text):
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:,0,:].squeeze().numpy()

    prediction = clf.predict([cls_emb])[0]
    probability = clf.predict_proba([cls_emb])[0]
    confidence = max(probability) * 100

    if confidence < 40:
     label = "SAFE ✅"
    elif confidence < 70:
     label = "SUSPICIOUS ⚠️ (Review Recommended)"
    else:
     label = "PHISHING 🚨 (High Risk)"


    print("\nEmail:", email_text)
    print("Verdict:", label)
    print(f"Confidence: {confidence:.2f}%")
    print("-"*50)

# -----------------------------
# Run predictions
# -----------------------------
for email in test_emails:
    predict_email(email)
