import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from library import (
    AREA_CODE,
    FEATURE_INFO,
    FEATURES,
    INDICATORS,
    TARGET,
    YEAR,
    load_data,
    load_models,
    predict_pou,
    region_lookup,
)

st.set_page_config(page_title="Food Security | Predict PoU", page_icon="\U0001F52E", layout="wide")
st.title("\U0001F52E Live PoU Prediction")
st.markdown(
    "Adjust a region/year and the indicator values below to get a **live prediction** of the Prevalence of "
    "Undernourishment (PoU)."
)

df = load_data()
models = load_models()
regions = region_lookup(df)

left, right = st.columns([1, 2])

with left:
    st.subheader("Setup")
    model_name = st.radio(
        "Model", ["Multilayer Perceptron", "Linear Regression"],
        help="MLP has better held-out accuracy (higher R², lower RMSE). "
             "Linear Regression is fully interpretable via its coefficients.",
        horizontal=True,
    )
    region_col, year_col = st.columns(2)
    region_name = region_col.selectbox("Region", regions["Country"])
    area_code = int(regions.loc[regions["Country"] == region_name, AREA_CODE].iloc[0])
    year = year_col.number_input(
        "Year", min_value=int(df[YEAR].min()), max_value=int(df[YEAR].max()),
        value=int(df[YEAR].max()),
    )

    region_hist = df[df[AREA_CODE] == area_code].sort_values(YEAR)
    default_row = region_hist.iloc[(region_hist[YEAR] - year).abs().argsort()[:1]]

    use_actual = st.checkbox("Use actual values for this region/year")

    indicator_values = {}
    indicator_rows = [["GDP"], ["FSV"], ["DWS", "SAN"], ["ANM"], ["OWC", "OBE"]]
    for row in indicator_rows:
        row_cols = st.columns(len(row))
        for col, ind in zip(row_cols, row):
            name, unit, desc = FEATURE_INFO[ind]
            lo, hi = float(df[ind].min()), float(df[ind].max())
            default = float(default_row[ind].iloc[0])
            key = f"slider_{ind}"
            if use_actual or key not in st.session_state:
                st.session_state[key] = default
            indicator_values[ind] = col.slider(
                f"{name} ({unit})" if unit else name,
                min_value=lo, max_value=hi, step=(hi - lo) / 200 or 1.0,
                key=key, help=desc, disabled=use_actual,
            )

pred = predict_pou(model_name, area_code, year, indicator_values, models)

with right:
    st.subheader("Prediction")
    gauge_col, metric_col = st.columns([2, 1])
    with gauge_col:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"suffix": "%", "valueformat": ".2f"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, max(40, float(df[TARGET].max()))]},
                    "bar": {"color": "#B33F3F"},
                    "steps": [
                        {"range": [0, 5], "color": "#EAF3EA"},
                        {"range": [5, 15], "color": "#F7ECC7"},
                        {"range": [15, max(40, float(df[TARGET].max()))], "color": "#F5D6D0"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, width="stretch")
    with metric_col:
        actual_val = default_row[TARGET].iloc[0] if not default_row.empty else None
        st.metric(
            f"Actual PoU ({int(default_row[YEAR].iloc[0])})",
            f"{actual_val:.2f}%" if actual_val is not None else "n/a",
        )
        metrics = models["metrics"][model_name]
        st.metric(f"Model Test R²", f"{metrics['r2']:.3f}", f"RMSE {metrics['rmse']:.2f}",
                  delta_color="off", delta_arrow="off")

    st.subheader(f"Historical PoU for {region_name}")
    fig_hist_line = px.line(
        region_hist, x=YEAR, y=TARGET, markers=True,
        labels={YEAR: "Year", TARGET: "PoU (%)"},
    )
    pred_series = [
        predict_pou(
            model_name, area_code, float(row[YEAR]),
            {ind: float(row[ind]) for ind in INDICATORS}, models,
        )
        for _, row in region_hist.iterrows()
    ]
    fig_hist_line.add_scatter(
        x=region_hist[YEAR], y=pred_series, mode="lines+markers",
        marker=dict(size=10, color="#B33F3F", symbol="star"),
        line=dict(color="#B33F3F", dash="dash"),
        name="Predicted",
    )
    fig_hist_line.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig_hist_line, width="stretch")

if model_name == "Linear Regression":
    st.subheader("Linear Regression Coefficients")
    st.caption(
        "Each coefficient is the expected change in PoU (in standardized units) "
        "per one-standard-deviation increase in that predictor, holding others constant."
    )
    coef_df = pd.DataFrame(
        {"Feature": FEATURES, "Weight": models["lr_model"].weights_[1:]}
    ).sort_values("Weight")
    fig_coef = px.bar(
        coef_df, x="Weight", y="Feature", orientation="h",
        color="Weight", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
    )
    fig_coef.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_coef, width="stretch")
