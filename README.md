# 01.020 Design Thinking Project III — Food Security & Sustainability

Modeling SDG Indicator 2.1.1, the **Prevalence of Undernourishment (PoU)**, from economic,
infrastructure, and population-health indicators in the FAO *Suite of Food Security
Indicators*, using a from-scratch **Linear Regression** and **Multilayer Perceptron**.

## Project structure

```
dtp3/
├── model.ipynb                          # Main deliverable: data cleaning, EDA, model
│                                         # training/evaluation, and artifact export.
├── presentation.mp4                      # Recorded project presentation.
├── requirements.txt                      # Dependencies for both the notebook and app.
├── data/                                 # Raw FAOSTAT dataset (input to model.ipynb).
│   ├── Food_Security_Data_E_All_Data_(Normalized).csv
│   ├── Food_Security_Data_E_AreaCodes.csv
│   ├── Food_Security_Data_E_Flags.csv
│   └── Descriptions_and_Metadata.xlsx
└── bonus/                                # Interactive Streamlit dashboard.
    ├── Home.py                           # App entry point / landing page.
    ├── library.py                        # Shared data loading + model classes.
    ├── pages/
    │   ├── 1_Explore_Data.py             # PoU map, trend, correlations, scatter plots.
    │   └── 2_Predict_PoU.py              # Live prediction with adjustable inputs.
    └── data/                             # Cleaned data + pickled model weights.
        ├── data.csv                      # Model-ready dataset.
        ├── data_pou.csv                  # Full PoU series.
        ├── area_codes.csv                # Area code -> country/region name lookup.
        └── models.pkl                    # Trained scaler + model weights.
```

## Running the notebook

`model.ipynb` contains the full workflow: loading the raw data, cleaning and
validating it, exploratory analysis, feature preparation, training the linear
regression and MLP models from scratch, evaluating them, and (in its final section)
exporting the cleaned data and trained weights to `bonus/data/` for the Streamlit app.

```bash
pip install -r requirements.txt
jupyter notebook model.ipynb
```

Run the notebook top to bottom; the last section must be executed at least once to
(re)generate the artifacts used by the bonus app.

## Running the Streamlit app

The app loads the cleaned data and pretrained model weights already committed under
`bonus/data/`, so it does not need to retrain models at startup.

```bash
pip install -r requirements.txt
cd bonus
streamlit run Home.py
```

This opens the app in your browser (default `http://localhost:8501`) with three pages:

- **Home:** project overview and a summary of the predictor variables.
- **Explore Data:** a world PoU choropleth map, PoU trend by region, a correlation
  heatmap, and scatter plots of each predictor against PoU.
- **Predict PoU:** pick a region, year, and model (Linear Regression or MLP), adjust
  indicator sliders (or use the region's actual values), and get a live PoU prediction
  compared against the historical trend.
