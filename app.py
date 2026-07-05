import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Emotion Prediction",
    page_icon="🧠",
    layout="wide"
)
st.title("🧠 Emotion Prediction using NLP")
# ---------------- LOAD MODEL ---------------- #

model = joblib.load("lr_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# ---------------- LABELS ---------------- #

emotion_labels = {
    "sadness": "😢 Sadness",
    "anger": "😡 Anger",
    "love": "❤️ Love",
    "surprise": "😲 Surprise",
    "fear": "😨 Fear",
    "joy": "😄 Joy"
}

emotion_colors = {
    "sadness": "#3498db",
    "anger": "#e74c3c",
    "love": "#ff4b6e",
    "surprise": "#f39c12",
    "fear": "#9b59b6",
    "joy": "#2ecc71"
}

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

.title{
    text-align:center;
    font-size:52px;
    font-weight:800;
    color:white;
}

.subtitle{
    text-align:center;
    color:#d1d5db;
    font-size:20px;
    margin-bottom:30px;
}

textarea{
    border-radius:15px !important;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown(
    "<div class='title'>🧠 Emotion Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>TF-IDF + Logistic Regression</div>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📘 About")

st.sidebar.markdown("""
### Model Used

- TF-IDF Vectorizer
- Logistic Regression
- NLP Text Classification

### Predictable Emotions

- 😢 Sadness
- 😡 Anger
- ❤️ Love
- 😲 Surprise
- 😨 Fear
- 😄 Joy
""")

# ---------------- INPUT ---------------- #

text = st.text_area(
    "✍ Enter your statement",
    height=180,
    placeholder="Example: I am feeling very happy today..."
)

col1, col2 = st.columns(2)

predict = col1.button("🚀 Predict Emotion")
clear = col2.button("🧹 Clear")

# ---------------- CLEAR ---------------- #

if clear:
    st.rerun()

# ---------------- PREDICTION ---------------- #

if predict:

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        with st.spinner("Analyzing emotion..."):

            vector = tfidf.transform([text])

            prediction = str(model.predict(vector)[0]).lower()

            probabilities = model.predict_proba(vector)[0]

            confidence = np.max(probabilities) * 100

            color = emotion_colors.get(prediction, "#4A00E0")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div style="
                    background:{color};
                    padding:30px;
                    border-radius:18px;
                    text-align:center;
                    color:white;
                    box-shadow:0px 8px 25px rgba(0,0,0,.3);
                ">
                    <h3>Predicted Emotion</h3>
                    <h1>{emotion_labels.get(prediction,prediction.title())}</h1>
                    <h4>Confidence : {confidence:.2f}%</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("📊 Emotion Probability")

            labels = [
                "sadness",
                "anger",
                "love",
                "surprise",
                "fear",
                "joy"
            ]

            prob_dict = {
                emotion_labels[label]: probabilities[i]
                for i, label in enumerate(labels)
            }

            st.bar_chart(prob_dict)

# ---------------- EXAMPLES ---------------- #

st.markdown("---")

st.markdown("---")
st.subheader("💡 Example")

st.info("😊 I got promoted today!")

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.caption("Developed with ❤️ using Streamlit | NLP | TF-IDF | Logistic Regression")