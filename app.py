import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")
st.title("🍷 Wine Quality Prediction System")
st.caption("Classroom prototype: quality >= 6 = GOOD QUALITY; quality < 6 = NEEDS IMPROVEMENT")

@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

if not os.path.exists("wine_model.pkl") or not os.path.exists("wine_quality.csv"):
    st.error("Missing wine_model.pkl or wine_quality.csv. Run python train_model.py first.")
    st.stop()

model = load_model()
data = pd.read_csv("wine_quality.csv")
c1, c2, c3 = st.columns(3)
c1.metric("Dataset rows", len(data))
c2.metric("Average quality", round(data["quality"].mean(), 2))
c3.metric("Good-quality rows", int((data["quality"] >= 6).sum()))
st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)
st.subheader("Predict uploaded samples")
template = data.drop(columns=["quality"]).head(5)
st.download_button("Download sample input CSV", template.to_csv(index=False), "sample_wine_input.csv", "text/csv")
uploaded = st.file_uploader("Upload wine sample CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        required = list(model.feature_names_in_)
        missing = [c for c in required if c not in new_data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            predictions = model.predict(new_data[required])
            results = new_data.copy()
            results["Prediction"] = ["GOOD QUALITY" if p == 1 else "NEEDS IMPROVEMENT" for p in predictions]
            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted good-quality samples", int((predictions == 1).sum()))
            st.download_button("Download prediction results", results.to_csv(index=False), "wine_quality_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not process CSV: {e}")
