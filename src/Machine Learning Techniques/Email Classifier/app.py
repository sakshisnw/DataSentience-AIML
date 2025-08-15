import streamlit as st
import joblib
import os

# ---------------------------
# Load the saved model & vectorizer
# ---------------------------
MODEL_PATH = os.path.join("models", "decision_tree_spam.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = joblib.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = joblib.load(f)

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="Email Spam Checker", page_icon="📧")

st.title("📧 Email Spam Checker")
st.write("Check if an email is spam or legitimate using a trained ML model.")

# ---------------------------
# Email Input
# ---------------------------
email_text = st.text_area("Enter Email Content", placeholder="Paste your email here...")

if st.button("Analyze Email"):
    if email_text.strip():
        with st.spinner("Analyzing..."):
            email_vectorized = vectorizer.transform([email_text])
            prediction = model.predict(email_vectorized)[0]
            probability = model.predict_proba(email_vectorized)[0]

        if prediction == 1:
            st.error(f"🚨 This email is **SPAM**\n\nConfidence: {probability[1]:.1%}")
        else:
            st.success(f"✅ This email is **LEGITIMATE**\n\nConfidence: {probability[0]:.1%}")
    else:
        st.warning("Please enter some email content.")

st.markdown("---")
st.caption("Powered by Decision Tree Classifier + TF-IDF • Built with Streamlit")
