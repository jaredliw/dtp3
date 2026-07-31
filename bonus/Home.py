import streamlit as st

from library import FEATURE_INFO, load_data, load_models

st.set_page_config(page_title="Food Security | Home", page_icon="\U0001F33E", layout="wide")

st.title("\U0001F33E Predicting the Prevalence of Undernourishment")
st.caption("01.020 Design Thinking Project III — Food Security & Sustainability")

st.markdown(
    """
Sustainability in food systems is a critical global challenge: long-term environmental,
economic, and social well-being depends on reliable access to sufficient, safe, and
nutritious food. Although global hunger has gradually declined since its post-pandemic
peak, hundreds of millions of people remain undernourished today.

This app explores the **FAO Suite of Food Security Indicators** and uses a
from-scratch (no scikit-learn) **Linear Regression** and **Multilayer Perceptron**
to model SDG Indicator 2.1.1, the **Prevalence of Undernourishment (PoU)**,
as a function of economic, infrastructure, and population-health indicators.
"""
)

with st.spinner("Loading data and models..."):
    df = load_data()
    models = load_models()

n_regions = df["Country"].nunique()
year_min, year_max = int(df["YR"].min()), int(df["YR"].max())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Observations", len(df))
col2.metric("Regions", n_regions)
col3.metric("Years Covered", f"{year_min}–{year_max}")

st.subheader("Predictor variables")
info_rows = [
    {"Code": k, "Name": v[0], "Unit": v[1], "Description": v[2]}
    for k, v in FEATURE_INFO.items()
]
st.dataframe(info_rows, hide_index=True, width="stretch")

st.subheader("Explore the app")
st.markdown(
    """
Use the sidebar to navigate:

1. **Explore Data**: correlations, distributions, and trends across regions and years.
2. **Predict PoU**: pick a region/year and adjust indicators to see the model's estimate.
"""
)
