from pathlib import Path

import pytest

TRAINING_SAMPLE_DIR = Path(__file__).parents[1] / "fixtures" / "training"


@pytest.fixture(scope="session")
def training_sample_dir():
    """Folder with the sample train/validation/test splits and dataset_card.json."""
    return TRAINING_SAMPLE_DIR

# --- Validation + sensitivity fixtures (evaluate.py / sensitivity.py) -------------------------------
# Self-contained: contract-shaped synthetic splits with the documented denial signal, and models fitted
# right here, so these tests don't depend on the training code. Everything is written under tmp_path.

import json  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from nsh_pipeline.modeling.dataset import (  # noqa: E402
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, LABEL_COLUMN, NUMERIC_FEATURES, TRAINING_COLUMNS,
)

CONTRACT_SPLITS = {  # split: (rows, first service date, seed)
    "train": (1500, "2025-01-01", 1),
    "validation": (600, "2025-05-01", 2),
    "test": (600, "2025-06-01", 3),
}


def make_contract_frame(n: int, start: str, seed: int, prefix: str) -> pd.DataFrame:
    """Contract-shaped split with the documented signal: amount, Dental/Vision, short tenure,
    high deductible and recent denials raise the denial probability."""
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, 28, n), unit="D")
    category = rng.choice(["Medical", "Drugs & Medicine", "Dental", "Vision"], n, p=[0.5, 0.25, 0.15, 0.1])
    amount = np.round(np.exp(rng.normal(np.log(75), 1.0, n)), 2)
    deductible = rng.choice([1000, 2000, 3000, 5000], n)
    tenure = rng.integers(10, 3000, n)
    prior = np.clip(rng.poisson(0.3, n), 0, 3)
    debit = rng.random(n) < 0.8
    logit = (-2.4 + 1.0 * np.log(amount / 75) + pd.Series(category).map(
        {"Medical": 0.0, "Drugs & Medicine": -0.3, "Dental": 1.3, "Vision": 1.1}).to_numpy()
        + 0.9 * (deductible >= 3000) + 1.1 * (tenure < 180) + 0.9 * prior + 0.3 * ~debit)
    frame = pd.DataFrame({
        "swipe_id": [f"{prefix}-{i:05d}" for i in range(n)],
        "member_id": [f"M-{i:04d}" for i in rng.integers(0, 300, n)],
        "client_id": [f"C-{i:02d}" for i in rng.integers(0, 20, n)],
        "service_date": dates,
        "amount_usd": amount, "plan_deductible": deductible.astype(float), "tenure_days": tenure,
        "prior_denials_90d": prior, "service_month": dates.month.astype(int),
        "service_category": category,
        "coverage_tier": rng.choice(["EE", "ES", "EC", "EF"], n),
        "carrier": rng.choice(["BCBS", "Aetna", "Cigna"], n),
        "age_band": rng.choice(["18-34", "35-54", "55-64", "65+"], n),
        "group_size_bucket": rng.choice(["1-9", "10-24", "25-49", "50-99"], n),
        "state": rng.choice(["CA", "NY", "TX", "OR"], n),
        "is_debit_card": debit,
        "is_vip": rng.random(n) < 0.05,
        LABEL_COLUMN: (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int),
    })
    return frame[TRAINING_COLUMNS]


@pytest.fixture(scope="session")
def contract_data_dir(tmp_path_factory):
    """data/training-shaped folder: train/validation/test parquet + dataset_card.json."""
    out = tmp_path_factory.mktemp("contract_training")
    card = {"dataset": "validation-test-fixture", "data_version": "fixture-v1", "seed": 42, "splits": {}}
    for name, (n, start, seed) in CONTRACT_SPLITS.items():
        frame = make_contract_frame(n, start, seed, prefix=name[:2].upper())
        frame.to_parquet(out / f"{name}.parquet", index=False)
        card["splits"][name] = {"rows": n, "positive_rate": float(frame[LABEL_COLUMN].mean())}
    (out / "dataset_card.json").write_text(json.dumps(card), encoding="utf-8")
    return out


def fit_fixture_pipeline(estimator, train: pd.DataFrame, verbose_names: bool) -> Pipeline:
    """Contract-shaped model: scaled numerics + one-hot categoricals + estimator."""
    preprocess = ColumnTransformer(
        [("numeric", StandardScaler(), NUMERIC_FEATURES),
         ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        verbose_feature_names_out=verbose_names,
    )
    model = Pipeline([("preprocess", preprocess), ("model", estimator)])
    return model.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])


def save_fixture_model(model, model_name: str, out_dir):
    """models/best layout: model.joblib + metadata.json with the contract keys."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    metadata = {"model_name": model_name, "params": {}, "cv_metrics": {}, "mlflow_run_id": None,
                "data_version": "fixture-v1", "git_sha": "fixture", "created_at": "2025-06-15T00:00:00+00:00"}
    (out_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return out_dir


@pytest.fixture(scope="session")
def linear_model_dir(contract_data_dir, tmp_path_factory):
    """Logistic regression fitted on the contract fixture (verbose 'numeric__' / 'categorical__' names)."""
    train = pd.read_parquet(contract_data_dir / "train.parquet")
    model = fit_fixture_pipeline(LogisticRegression(max_iter=2000), train, verbose_names=True)
    return save_fixture_model(model, "logistic_regression", tmp_path_factory.mktemp("linear_model") / "best")


@pytest.fixture(scope="session")
def tree_model_dir(contract_data_dir, tmp_path_factory):
    """Random forest fitted on the contract fixture (plain feature names, like the real preprocessing)."""
    train = pd.read_parquet(contract_data_dir / "train.parquet")
    forest = RandomForestClassifier(n_estimators=60, max_depth=6, min_samples_leaf=5, random_state=0, n_jobs=1)
    model = fit_fixture_pipeline(forest, train, verbose_names=False)
    return save_fixture_model(model, "random_forest", tmp_path_factory.mktemp("tree_model") / "best")


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    """Throwaway SQLite MLflow store under tmp_path (and ./mlruns artifacts there too, not in the repo)."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


@pytest.fixture
def write_params(tmp_path):
    """Write a params.yaml with a model_validation section (plus overrides) and return its path."""
    def _write(validation: dict | None = None, **top_level):
        params = {"seed": 42, "model": {"mlflow": {"experiment": "nsh-denial-model"}},
                  "model_validation": {"threshold_method": "best_f1", "min_pr_auc": 0.18, "min_pr_auc_lift": 1.25,
                                       "min_roc_auc": 0.60, "min_recall": 0.30,
                                       "sensitivity": {"shap_sample_size": 150, "shap_background_size": 50}}}
        params["model_validation"].update(validation or {})
        params.update(top_level)
        path = tmp_path / "params.yaml"
        path.write_text(yaml.safe_dump(params), encoding="utf-8")
        return path
    return _write
