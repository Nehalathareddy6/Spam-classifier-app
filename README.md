# Spam-classifier-app
A simple AI/ML spam classifier with streamlit GUI
pip install streamlit pandas scikit-learnimport streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Training a simple spam model
# -----------------------------
data = {
    "message": [
        "Congratulations! You won a free lottery ticket",
        "Hi, are we meeting tomorrow?",
        "Get free recharge now!!!",
        "Hello, how are you?",
        "Win cash prize by clicking this link",
        "Hey, don’t forget our meeting today",
        "Claim your free reward now!!!",
        "Good morning, have a nice day"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

X = df["message"]
y = df["label"]

cv = CountVectorizer()
X_vec = cv.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Spam Mail Classifier", page_icon="📩", layout="centered")

st.title("📩 Spam Mail Classifier")
st.write("Enter a message and find out if it's **Spam or Not Spam**!")

# Input box
user_input = st.text_area("Type your message here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        user_vec = cv.transform([user_input])
        prediction = model.predict(user_vec)[0]
        
        if prediction == "spam":
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")streamlit run app.py
