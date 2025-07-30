import streamlit as st
import pandas as pd
import shap
import joblib
from streamlit_shap import st_shap
import numpy as np
# 1. Config
st.set_page_config(layout="centered")
st.title("Stage 1 SHAP Dashboard — Fraud Detection")
FEATURE_COLS = [f"V{i}" for i in range(1,29)] + ["Amount"]

# 2. Load
@st.cache_data
def load_data():
    return pd.read_csv("../data/creditcard.csv")
@st.cache_resource
def load_model():
    return joblib.load("../models/lgbm_model.pkl")
data = load_data()
model = load_model()
features = data[FEATURE_COLS]

# 3. SHAP Explanation (for all rows)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

# 4. Sidebar for transaction index
idx = st.sidebar.number_input("Transaction index", 0, len(data)-1, 0)

# 5. GLOBAL BAR CHART (pre-rendered static image)
st.header("Global SHAP Feature Importance")
st.image("../figures/shap_bar_lightgbm.png", use_container_width=True)

# 6. TRANSACTION-LEVEL FORCE PLOT
st.header(f"SHAP Force Plot: Transaction {idx}")

# — THE KEY PART: Construct a SHAP Explanation for one sample —
sv_sample = shap_values[1][idx] if isinstance(shap_values, list) else shap_values[idx]
data_sample = features.iloc[idx].values
base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple, pd.Series, np.ndarray)) else explainer.expected_value

explanation = shap.Explanation(
    values = sv_sample,
    base_values = base_value,
    data = data_sample,
    feature_names = FEATURE_COLS
)

# Show in Streamlit
force_viz = shap.plots.force(explanation)
st_shap(force_viz, height=350)
st.info("This dashboard lets you explore global feature importance, individual predictions, and how feature values affect SHAP explanations. The interactive force plot shows why the model made a decision for each transaction.")
