from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load pre-trained model
with open('spam_hema.pkl', 'rb') as file:
    model = pickle.load(file)

import re

def extract_features_from_input(email_text):
    email_text = email_text.lower()  # Convert to lowercase for consistency
    length = len(email_text)
    has_free = int("free" in email_text)
    has_offer = int("offer" in email_text)
    has_exclamation = email_text.count("!")
    has_cash = int("cash" in email_text)
    has_win = int("win" in email_text)
    has_discount = int("discount" in email_text)
    has_http = int("http" in email_text or "www" in email_text)  # Check for links
    uppercase_ratio = sum(1 for c in email_text if c.isupper()) / max(1, len(email_text))  # Uppercase ratio
    avg_word_length = sum(len(word) for word in email_text.split()) / max(1, len(email_text.split()))  # Avg word length

    return [length, has_free, has_offer, has_exclamation, has_cash, has_win, has_discount, has_http, uppercase_ratio, avg_word_length]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        email_text = request.form.get("email_text")
        if email_text:
            features = extract_features_from_input(email_text)
            prediction = model.predict([features])[0]
            result = "Spam" if prediction == 1 else "Not Spam"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
