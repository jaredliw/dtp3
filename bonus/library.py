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
    "Northern America": ["BMU", "CAN", "GRL", "SPM", "USA"],
    "Caribbean": [
        "ABW", "AIA", "ATG", "BES", "BHS", "BLM", "BRB", "CUB", "CUW", "CYM", "DMA", "DOM", "GLP", "GRD",
        "HTI", "JAM", "MAF", "MSR", "MTQ", "PRI", "KNA", "LCA", "SXM", "TCA", "TTO", "VCT", "VGB", "VIR",
    ],
    "Central America": ["BLZ", "CRI", "SLV", "GTM", "HND", "MEX", "NIC", "PAN"],
    "South America": [
        "ARG", "BOL", "BRA", "BVT", "CHL", "COL", "ECU", "FLK", "GUF", "GUY", "PRY", "PER", "SGS", "SUR",
        "URY", "VEN",
    ],
    "Central Asia": ["KAZ", "KGZ", "TJK", "TKM", "UZB"],
    "Eastern Asia": ["CHN", "HKG", "MAC", "TWN", "PRK", "JPN", "MNG", "KOR"],
    "Southern Asia": ["AFG", "BGD", "BTN", "IND", "IRN", "MDV", "NPL", "PAK", "LKA"],
    "South-eastern Asia": ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "TLS", "VNM"],
    "Western Asia": [
        "ARM", "AZE", "BHR", "CYP", "GEO", "IRQ", "ISR", "JOR", "KWT", "LBN", "OMN", "PSE", "QAT", "SAU",
        "SYR", "TUR", "ARE", "YEM",
    ],
    "Eastern Africa": [
        "ATF", "BDI", "COM", "DJI", "ERI", "ETH", "IOT", "KEN", "MDG", "MWI", "MUS", "MOZ", "MYT", "REU",
        "RWA", "SYC", "SOM", "SSD", "UGA", "TZA", "ZMB", "ZWE",
    ],
    "Middle Africa": ["AGO", "CMR", "CAF", "TCD", "COG", "COD", "GNQ", "GAB", "STP"],
    "Northern Africa": ["DZA", "EGY", "ESH", "LBY", "MAR", "SDN", "TUN"],
    "Southern Africa": ["BWA", "SWZ", "LSO", "NAM", "ZAF"],
    "Western Africa": [
        "BEN", "BFA", "CPV", "CIV", "GMB", "GHA", "GIN", "GNB", "LBR", "MLI", "MRT", "NER", "NGA", "SEN",
        "SHN", "SLE", "TGO",
    ],
    "Eastern Europe": ["BLR", "BGR", "CZE", "HUN", "POL", "MDA", "ROU", "RUS", "SVK", "UKR"],
    "Northern Europe": [
        "ALA", "DNK", "EST", "FIN", "FRO", "GGY", "ISL", "IMN", "IRL", "JEY", "LVA", "LTU", "NOR", "SJM",
        "SWE", "GBR",
    ],
    "Southern Europe": [
        "ALB", "AND", "BIH", "HRV", "GIB", "GRC", "ITA", "MLT", "MNE", "MKD", "PRT", "SMR", "SRB", "SVN",
        "ESP", "VAT",
    ],
    "Western Europe": ["AUT", "BEL", "FRA", "DEU", "LIE", "LUX", "MCO", "NLD", "CHE"],
    "Australia and New Zealand": ["AUS", "CCK", "CXR", "HMD", "NFK", "NZL"],
    "Melanesia": ["FJI", "NCL", "PNG", "SLB", "VUT"],
    "Micronesia": ["FSM", "GUM", "KIR", "MHL", "MNP", "NRU", "PLW", "UMI"],
    "Polynesia": ["ASM", "COK", "NIU", "PCN", "PYF", "TKL", "TON", "TUV", "WLF", "WSM"],
}


# ----------------------------------------------------------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the cleaned, model-ready dataset used to train and evaluate the models.

    Reads ``data/data.csv`` (one row per area-year with no missing indicators), joins in region names from the area-code
    lookup table.

    Returns:
        DataFrame with columns ``[AREA_CODE, YEAR, *INDICATORS, TARGET, "Country"]``, sorted by country and year.
    """
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
    """Load the full PoU time series, **including** region-years missing an indicator.

    Reads ``data/data_pou.csv`` and joins in country/region names. Used for the map and trend views, which only need PoU
    and therefore do not require the stricter completeness that :func:`load_data` enforces.

    Returns:
        DataFrame with PoU observations for every area-year, sorted by country and year.
    """
    df = pd.read_csv(FULL_CSV)
    area_codes = pd.read_csv(AREA_CODES_CSV)[["Area Code", "Area"]].rename(
        columns={"Area Code": AREA_CODE, "Area": "Country"}
    )
    df = df.merge(area_codes, on=AREA_CODE, how="left")
    df[AREA_CODE] = df[AREA_CODE].astype(int)
    df[YEAR] = df[YEAR].astype(int)
    return df.sort_values(["Country", YEAR]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------------------------------------------------

class StandardScaler:
    """A from-scratch implementation of Z-score normalizer for a DataFrame of features.

    Standardizes each column to zero mean and unit variance: ``z = (x - mean) / std``.
    Fitted statistics are computed on the training data and reused to transform any subsequent data with the same
    columns.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.features_ = None

    def fit(self, X):
        """Compute the per-column mean and standard deviation of ``X``."""
        self.mean_ = X.mean()
        self.std_ = X.std()
        self.features_ = X.columns
        return self

    def _check(self, X):
        """Raise ``ValueError`` if the scaler is unfitted or ``X``'s columns don't match."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Encoder has not been fitted.")
        if self.features_ is None or not self.features_.equals(X.columns):
            raise ValueError("Columns does not match fitted data.")

    def transform(self, X):
        """Apply the fitted z-score normalization to ``X``."""
        self._check(X)
        X = X.copy()
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit to ``X``, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Undo the z-score normalization, recovering the original scale of ``X``."""
        self._check(X)
        X = X.copy()
        return X * self.std_ + self.mean_


# ----------------------------------------------------------------------------------------------------------------------
# Linear Regression
# ----------------------------------------------------------------------------------------------------------------------

class LinearRegression:
    """A from-scratch implementation of ordinary least-squares linear regression fitted by batch gradient descent.

    Args:
        lr: Gradient descent learning rate.
        n_iters: Number of full-batch gradient descent iterations to run in :meth:`fit`.
    """

    def __init__(self, lr=0.001, n_iters=5000):
        self.lr_ = lr
        self.n_iters_ = n_iters
        self.weights_ = None
        self.history_ = []

    def _predict(self, X):
        """Predict from an already bias-augmented feature matrix ``X``."""
        return (X @ self.weights_).squeeze()

    def predict(self, X):
        """Predict the target for each row of ``X`` (without a bias column)."""
        X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        return self._predict(X)

    def _cost(self, X, y):
        """Mean squared error cost (halved) for bias-augmented ``X`` against targets ``y``."""
        n_samples, n_features = X.shape
        y_hat = self._predict(X)
        return (1 / (2 * n_samples)) * np.sum((y_hat - y) ** 2)

    def _cost_grad(self, X, y):
        """Gradient of the cost with respect to the weights, for bias-augmented ``X``."""
        n_samples, n_features = X.shape
        y_hat = self._predict(X)
        return (1 / n_samples) * X.T @ (y_hat - y)

    def fit(self, X, y):
        """Fit the model to ``X``/``y`` via full-batch gradient descent.

        Runs ``n_iters_`` update steps, recording the cost after each step in ``history_``.
        """
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
    """A fully-connected (dense) layer with a learnable weight matrix and bias.

    Weights are He-initialized (``N(0, 2 / n_in)``), which suits downstream ReLU activations by keeping activation and
    gradient magnitudes stable during training.

    Args:
        n_in: Number of input features.
        n_out: Number of output units.
        random_state: Seed for the weight initialization RNG.
    """

    def __init__(self, n_in, n_out, random_state=None):
        self.learnable_ = True
        self.random_state = random_state
        self.input_ = None
        self.weights_gradient_ = None
        rng = np.random.default_rng(random_state)
        self.weights_ = rng.standard_normal((n_out, n_in + 1)) * np.sqrt(2 / n_in)

    def forward(self, X):
        """Bias-augment ``X``, cache it for :meth:`backward`, and return ``X @ W.T``."""
        X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        self.input_ = X
        return X @ self.weights_.T

    def backward(self, delta):
        """Given the upstream gradient ``delta``, set ``weights_gradient_`` and return
        the gradient to propagate to the previous layer."""
        self.weights_gradient_ = delta.T @ self.input_
        return delta @ self.weights_[:, 1:]


class Relu:
    """Elementwise ReLU activation: ``max(0, x)``."""

    def __init__(self):
        self.learnable_ = False
        self.input_ = None

    def forward(self, X):
        """Apply ReLU to ``X``, caching it to compute the gradient in :meth:`backward`."""
        self.input_ = X
        return np.maximum(0, X)

    def backward(self, delta):
        """Zero out ``delta`` wherever the cached input was non-positive."""
        delta_relu = self.input_ > 0
        return delta * delta_relu


class RegressiveMultilayerPerceptron:
    """A from-scratch implementation of a feedforward multilayer perceptron for regression, trained by backpropagation.

    Args:
        layers: Ordered list of layer objects (each exposing ``forward``, ``backward``, and a ``learnable_`` flag)
            making up the network.
        lr: Gradient descent learning rate.
        n_iters: Number of full-batch gradient descent iterations to run in :meth:`fit`.
    """

    def __init__(self, layers, lr=0.001, n_iters=5000):
        self.lr_ = lr
        self.n_iters_ = n_iters
        self.layers_ = layers
        self.history_ = []

    def predict(self, X):
        """Run the forward pass through every layer and return the predicted targets."""
        cur = X
        for layer in self.layers_:
            cur = layer.forward(cur)
        return cur.squeeze(-1)

    def _cost(self, X, y):
        """Mean squared error cost (halved) for ``X`` against targets ``y``."""
        n_samples, _ = X.shape
        y_hat = self.predict(X)
        return (1 / (2 * n_samples)) * np.sum((y_hat - y) ** 2)

    def _cost_grad_wrt_pred(self, X, y):
        """Gradient of the cost with respect to the network's predictions."""
        n_samples, _ = X.shape
        y_hat = self.predict(X)
        return np.expand_dims((y_hat - y) / n_samples, axis=-1)

    def _backpropagate(self, X, y):
        """Run one forward-backward pass, updating every learnable layer's weights."""
        delta = self._cost_grad_wrt_pred(X, y)
        for layer in self.layers_[::-1]:
            delta = layer.backward(delta)
            if layer.learnable_:
                layer.weights_ -= self.lr_ * layer.weights_gradient_

    def fit(self, X, y):
        """Train the network on ``X``/``y`` for ``n_iters_`` steps of backpropagation.

        Records the cost after each step in ``history_``.
        """
        self.history_ = []
        for _ in range(self.n_iters_):
            self._backpropagate(X, y)
            self.history_.append(self._cost(X, y))


# ----------------------------------------------------------------------------------------------------------------------
# Loading Pretrained Models
# ----------------------------------------------------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Reconstruct the fitted scaler, linear regression, and MLP models from disk.

    Rebuilds each object from the weights and scaler statistics pickled to ``data/models.pkl`` by ``model.ipynb``,
    rather than retraining at runtime.

    Returns:
        Dict with keys ``"scaler"``, ``"lr_model"``, ``"mlp_model"``, and ``"metrics"`` (a dict of held-out
        ``r2``/``rmse`` per model name, for display in the app).
    """
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
    """Predict PoU for a single region/year using a fitted model.

    Args:
        model_name: Either ``"Linear Regression"`` or ``"Multilayer Perceptron"`` (any other value falls back to the
            MLP model).
        area_code: FAO area code identifying the region.
        year: Year of the observation.
        indicators: Mapping of indicator code (see :data:`INDICATORS`) to its value.
        models: The dict returned by :func:`load_models`.

    Returns:
        The predicted Prevalence of Undernourishment, clipped to the valid ``[0, 100]`` percentage range.
    """
    scaler = models["scaler"]
    row = {AREA_CODE: area_code, YEAR: year, **indicators}
    X = pd.DataFrame([row])[FEATURES]
    X_scaled = scaler.transform(X).to_numpy()

    model = models["lr_model"] if model_name == "Linear Regression" else models["mlp_model"]
    pred = np.asarray(model.predict(X_scaled)).reshape(-1)[0]
    return float(np.clip(pred, 0, 100))
