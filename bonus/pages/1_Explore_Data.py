import numpy as np
import plotly.express as px
import streamlit as st

from library import FEATURE_INFO, REGION_COUNTRIES, TARGET, AREA_CODE, YEAR, load_data

st.set_page_config(page_title="Food Security | Explore Data", page_icon="\U0001F4CA", layout="wide")
st.title("\U0001F4CA Explore the Data")

df = load_data()
regions = sorted(df["Country"].unique())

# --- PoU Map ----------------------------------------------------------------------------------------------------------
st.subheader("PoU Map")
st.markdown("PoU across regions of the world over time.")

geo_df = df[df["Country"].isin(REGION_COUNTRIES)].copy()
geo_df["iso3"] = geo_df["Country"].map(REGION_COUNTRIES)
geo_df = geo_df.explode("iso3")

fig_map = px.choropleth(
    geo_df,
    locations="iso3",
    locationmode="ISO-3",
    color=TARGET,
    hover_name="Country",
    animation_frame=YEAR,
    color_continuous_scale="YlOrRd",
    range_color=[0, float(df[TARGET].max())],
    projection="natural earth",
    labels={TARGET: "PoU (%)"},
)
fig_map.update_layout(
    height=600,
    margin=dict(l=0, r=0, t=30, b=0),
    coloraxis_colorbar_title="PoU (%)",
)
fig_map.update_geos(showcountries=True, countrycolor="#DDDDDD", showland=True, landcolor="#F7F7F5")
st.plotly_chart(fig_map, width="stretch")

# --- PoU Trend --------------------------------------------------------------------------------------------------------
st.subheader("PoU Trend")
st.markdown("Prevalence of Undernourishment over time for the selected regions.")

selected_regions = st.multiselect("Regions", regions, default=[
    "Africa",
    "Asia",
    "Europe",
    # "North America",
    "South America",
    # "Oceania",
    "World"
])
trend_df = df[df["Country"].isin(selected_regions)]
st.caption(f"{len(trend_df)} observations selected.")

fig_trend = px.line(
    trend_df.sort_values(YEAR),
    x=YEAR,
    y=TARGET,
    color="Country",
    markers=True,
    labels={YEAR: "Year", TARGET: "PoU (%)", "Country": "Region"},
)
fig_trend.update_layout(legend_title_text="Region")
st.plotly_chart(fig_trend, width="stretch")

# --- Correlation Heatmap ----------------------------------------------------------------------------------------------
st.subheader("Correlation Heatmap")
st.markdown("Pearson correlation between each indicator and the Prevalence of Undernourishment.")
numeric_cols = ["AC", YEAR, *FEATURE_INFO.keys(), TARGET]
corr = df[numeric_cols].corr(numeric_only=True)

fig_corr = px.imshow(
    corr,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    text_auto=".2f",
    aspect="auto",
)
fig_corr.update_layout(coloraxis_colorbar_title="r")
st.plotly_chart(fig_corr, width="stretch")

# --- Scatter Plots of Predictors ---------------------------------------------------------------------------------------
st.subheader("Predictors vs. PoU")
st.markdown("Each indicator plotted against Prevalence of Undernourishment, ordered by absolute correlation strength "
            "(strongest first).")
sorted_corr = corr[TARGET].drop([TARGET, YEAR, AREA_CODE]).sort_values(key=abs, ascending=False)
all_features = sorted_corr.index.tolist()

for row_start in range(0, len(all_features), 3):
    cols = st.columns(3)
    for col, feat in zip(cols, all_features[row_start:row_start + 3]):
        name, unit, _ = FEATURE_INFO.get(feat, (feat, "", ""))
        fig = px.scatter(
            df,
            x=feat,
            y=TARGET,
            labels={feat: f"{name} ({unit})" if unit else name, TARGET: "PoU (%)"},
            title=f"{name} (r = {sorted_corr[feat]:.2f})",
        )
        fig.update_traces(marker=dict(size=6, opacity=0.6))
        col.plotly_chart(fig, width="stretch")
