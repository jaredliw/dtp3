"""Shared data loading, preprocessing, and model definitions for the PoU Streamlit app.

Ports the custom StandardScaler, LinearRegression, and MLP classes built in ``model.ipynb``. The models themselves
are trained in the notebook and pickled to ``data/models.pkl``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
CLEANED_CSV = DATA_DIR / "data.csv"
FULL_CSV = DATA_DIR / "data_pou.csv"
AREA_CODES_CSV = DATA_DIR / "area_codes.csv"
MODELS_PKL = DATA_DIR / "models.pkl"

AREA_CODE = "AC"
YEAR = "YR"
TARGET = "PoU"
FEATURES = [AREA_CODE, YEAR, "GDP", "FSV", "DWS", "SAN", "OWC", "OBE", "ANM"]
INDICATORS = ["GDP", "FSV", "DWS", "SAN", "OWC", "OBE", "ANM"]

FEATURE_INFO = {
    "GDP": ("GDP per capita, PPP", "$", "Gross domestic product per capita (constant 2021 international $)."),
    "FSV": ("Food supply variability", "kcal/cap/day", "Per-capita food supply variability across time."),
    "DWS": ("Basic drinking water access", "%", "Population using at least basic drinking water services."),
    "SAN": ("Basic sanitation access", "%", "Population using at least basic sanitation services."),
    "OWC": ("Child overweight", "%", "Children under 5 who are overweight (modelled estimates)."),
    "OBE": ("Adult obesity", "%", "Adults (18+) with obesity (BMI > 30 kg/m^2)."),
    "ANM": ("Anemia in women", "%", "Women 15-49 with anemia (low hemoglobin)."),
}

REGION_COUNTRIES = {
    "Northern America": ["BMU", "CAN", "GRL", "USA"],
    "Caribbean": [
        "ATG", "BHS", "BRB", "CUB", "DMA", "DOM", "GRD", "HTI", "JAM", "PRI", "KNA", "LCA", "VCT", "TTO",
    ],
    "Central America": ["BLZ", "CRI", "SLV", "GTM", "HND", "MEX", "NIC", "PAN"],
    "South America": ["ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PRY", "PER", "SUR", "URY", "VEN"],
    "Central Asia": ["KAZ", "KGZ", "TJK", "TKM", "UZB"],
    "Southern Asia": ["AFG", "BGD", "BTN", "IND", "IRN", "MDV", "NPL", "PAK", "LKA"],
    "South-eastern Asia": ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "TLS", "VNM"],
    "Western Asia": [
        "ARM", "AZE", "BHR", "CYP", "GEO", "IRQ", "ISR", "JOR", "KWT", "LBN", "OMN", "PSE", "QAT", "SAU",
        "SYR", "TUR", "ARE", "YEM",
    ],
    "Eastern Africa": [
        "BDI", "COM", "DJI", "ERI", "ETH", "KEN", "MDG", "MWI", "MUS", "MOZ", "RWA", "SYC", "SOM", "SSD",
        "UGA", "TZA", "ZMB", "ZWE",
    ],
    "Middle Africa": ["AGO", "CMR", "CAF", "TCD", "COG", "COD", "GNQ", "GAB", "STP"],
    "Northern Africa": ["DZA", "EGY", "LBY", "MAR", "SDN", "TUN"],
    "Southern Africa": ["BWA", "SWZ", "LSO", "NAM", "ZAF"],
    "Western Africa": [
        "BEN", "BFA", "CPV", "CIV", "GMB", "GHA", "GIN", "GNB", "LBR", "MLI", "MRT", "NER", "NGA", "SEN",
        "SLE", "TGO",
    ],
    "Eastern Europe": ["BLR", "BGR", "CZE", "HUN", "POL", "MDA", "ROU", "RUS", "SVK", "UKR"],
    "Northern Europe": ["DNK", "EST", "FIN", "ISL", "IRL", "LVA", "LTU", "NOR", "SWE", "GBR"],
    "Southern Europe": [
        "ALB", "AND", "BIH", "HRV", "GRC", "ITA", "MLT", "MNE", "MKD", "PRT", "SRB", "SVN", "ESP",
    ],
    "Western Europe": ["AUT", "BEL", "FRA", "DEU", "LUX", "NLD", "CHE"],
    "Australia and New Zealand": ["AUS", "NZL"],
    "Polynesia": ["ASM", "COK", "PYF", "NIU", "WSM", "TKL", "TON", "TUV"],
}


# ----------------------------------------------------------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_CSV)
    area_codes = pd.read_csv(AREA_CODES_CSV)[["Area Code", "Area"]].rename(
        columns={"Area Code": AREA_CODE, "Area": "Country"}
    )
    df = df.merge(area_codes, on=AREA_CODE, how="left")
    df = df[[AREA_CODE, YEAR, *INDICATORS, TARGET, "Country"]].copy()
    df[[AREA_CODE, YEAR, *INDICATORS, TARGET]] = df[[AREA_CODE, YEAR, *INDICATORS, TARGET]].astype(np.float64)
    df[AREA_CODE] = df[AREA_CODE].astype(int)
    df[YEAR] = df[YEAR].astype(int)
    return df.sort_values(["Country", YEAR]).reset_index(drop=True)


@st.cache_data
def load_pou_data() -> pd.DataFrame:
    df = pd.read_csv(FULL_CSV)
    area_codes = pd.read_csv(AREA_CODES_CSV)[["Area Code", "Area"]].rename(
        columns={"Area Code": AREA_CODE, "Area": "Country"}
    )
    df = df.merge(area_codes, on=AREA_CODE, how="left")
    df[AREA_CODE] = df[AREA_CODE].astype(int)
    df[YEAR] = df[YEAR].astype(int)
    return df.sort_values(["Country", YEAR]).reset_index(drop=True)


def region_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """One row per region: Area Code <-> Country/region name."""
    return df[[AREA_CODE, "Country"]].drop_duplicates().sort_values("Country").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------------------------------------------------

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.features_ = None

    def fit(self, X):
        self.mean_ = X.mean()
        self.std_ = X.std()
        self.features_ = X.columns
        return self

    def _check(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Encoder has not been fitted.")
        if self.features_ is None or not self.features_.equals(X.columns):
            raise ValueError("Columns does not match fitted data.")

    def transform(self, X):
        self._check(X)
        X = X.copy()
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        self._check(X)
        X = X.copy()
        return X * self.std_ + self.mean_


# ----------------------------------------------------------------------------------------------------------------------
# Linear Regression
# ----------------------------------------------------------------------------------------------------------------------

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=5000):
        self.lr_ = lr
        self.n_iters_ = n_iters
        self.weights_ = None
        self.history_ = []

    def _predict(self, X):
        return (X @ self.weights_).squeeze()

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        return self._predict(X)

    def _cost(self, X, y):
        n_samples, n_features = X.shape
        y_hat = self._predict(X)
        return (1 / (2 * n_samples)) * np.sum((y_hat - y) ** 2)

    def _cost_grad(self, X, y):
        n_samples, n_features = X.shape
        y_hat = self._predict(X)
        return (1 / n_samples) * X.T @ (y_hat - y)

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))

        n_samples, n_features = X.shape
        self.weights_ = np.zeros((n_features,))
        self.history_ = []

        for _ in range(self.n_iters_):
            self.weights_ -= self.lr_ * self._cost_grad(X, y)
            self.history_.append(self._cost(X, y))

        return self


# ----------------------------------------------------------------------------------------------------------------------
# Multilayer Perceptron
# ----------------------------------------------------------------------------------------------------------------------

class LinearRegressionLayer:
    def __init__(self, n_in, n_out, random_state=None):
        self.learnable_ = True
        self.random_state = random_state
        self.input_ = None
        self.weights_gradient_ = None
        rng = np.random.default_rng(random_state)
        self.weights_ = rng.standard_normal((n_out, n_in + 1)) * np.sqrt(2 / n_in)

    def forward(self, X):
        X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        self.input_ = X
        return X @ self.weights_.T

    def backward(self, delta):
        self.weights_gradient_ = delta.T @ self.input_
        return delta @ self.weights_[:, 1:]


class Relu:
    def __init__(self):
        self.learnable_ = False
        self.input_ = None

    def forward(self, X):
        self.input_ = X
        return np.maximum(0, X)

    def backward(self, delta):
        delta_relu = self.input_ > 0
        return delta * delta_relu


class RegressiveMultilayerPerceptron:
    def __init__(self, layers, lr=0.001, n_iters=5000):
        self.lr_ = lr
        self.n_iters_ = n_iters
        self.layers_ = layers
        self.history_ = []

    def predict(self, X):
        cur = X
        for layer in self.layers_:
            cur = layer.forward(cur)
        return cur.squeeze(-1)

    def _cost(self, X, y):
        n_samples, _ = X.shape
        y_hat = self.predict(X)
        return (1 / (2 * n_samples)) * np.sum((y_hat - y) ** 2)

    def _cost_grad_wrt_pred(self, X, y):
        n_samples, _ = X.shape
        y_hat = self.predict(X)
        return np.expand_dims((y_hat - y) / n_samples, axis=-1)

    def _backpropagate(self, X, y):
        delta = self._cost_grad_wrt_pred(X, y)
        for layer in self.layers_[::-1]:
            delta = layer.backward(delta)
            if layer.learnable_:
                layer.weights_ -= self.lr_ * layer.weights_gradient_

    def fit(self, X, y):
        self.history_ = []
        for _ in range(self.n_iters_):
            self._backpropagate(X, y)
            self.history_.append(self._cost(X, y))


# ----------------------------------------------------------------------------------------------------------------------
# Loading Pretrained Models
# ----------------------------------------------------------------------------------------------------------------------

@st.cache_resource
def load_models():
    with open(MODELS_PKL, "rb") as f:
        artifact = pickle.load(f)

    scaler = StandardScaler()
    scaler.mean_ = artifact["scaler_mean"]
    scaler.std_ = artifact["scaler_std"]
    scaler.features_ = artifact["scaler_features"]

    lr_model = LinearRegression()
    lr_model.weights_ = artifact["lr_weights"]

    mlp_model = RegressiveMultilayerPerceptron([
        LinearRegressionLayer(len(FEATURES), 32, random_state=42),
        Relu(),
        LinearRegressionLayer(32, 16, random_state=42),
        Relu(),
        LinearRegressionLayer(16, 1, random_state=42),
    ])
    for layer, weights in zip((l for l in mlp_model.layers_ if l.learnable_), artifact["mlp_weights"]):
        layer.weights_ = weights

    return {
        "scaler": scaler,
        "lr_model": lr_model,
        "mlp_model": mlp_model,
        "metrics": artifact["metrics"],
    }


def predict_pou(model_name: str, area_code: float, year: float, indicators: dict, models: dict) -> float:
    scaler = models["scaler"]
    row = {AREA_CODE: area_code, YEAR: year, **indicators}
    X = pd.DataFrame([row])[FEATURES]
    X_scaled = scaler.transform(X).to_numpy()

    model = models["lr_model"] if model_name == "Linear Regression" else models["mlp_model"]
    pred = np.asarray(model.predict(X_scaled)).reshape(-1)[0]
    return float(np.clip(pred, 0, 100))
