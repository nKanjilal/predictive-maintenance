"""Automated checks for the data pipeline and the trained model.

Run from the repo root:  pytest -v
"""
import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (ALL_FEATURES, ENGINEERED, FAILURE_MODES, RAW_FEATURES,  # noqa: E402
                      add_features, load_dataset, load_raw)

MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


# ---------- Data quality ----------------------------------------------------------

def test_raw_data_has_expected_shape_and_columns():
    df = load_raw()
    assert len(df) == 10_000
    expected = {"type", "air_temp_k", "process_temp_k", "rpm", "torque_nm",
                "tool_wear_min", "failure", *FAILURE_MODES}
    assert expected.issubset(df.columns)


def test_no_missing_values_in_model_inputs(dataset):
    df, _, _ = dataset
    assert df[ALL_FEATURES].notna().all().all()


def test_target_is_binary_and_imbalanced(dataset):
    _, y, _ = dataset
    assert set(y.unique()) == {0, 1}
    # AI4I has ~3.4% failures; a big change means the data file changed
    assert 0.02 < y.mean() < 0.05


def test_sensor_values_are_in_physical_ranges(dataset):
    df, _, _ = dataset
    assert df["air_temp_k"].between(290, 310).all()
    assert df["process_temp_k"].between(300, 320).all()
    assert (df["rpm"] > 0).all()
    assert (df["torque_nm"] >= 0).all()
    assert (df["tool_wear_min"] >= 0).all()
    assert df["type"].isin(["L", "M", "H"]).all()


# ---------- Feature engineering ---------------------------------------------------

def test_engineered_features_match_formulas():
    row = pd.DataFrame([{"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0,
                         "rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100}])
    out = add_features(row).iloc[0]
    assert out["temp_diff_k"] == pytest.approx(10.0)
    assert out["power_w"] == pytest.approx(40.0 * 1500 * 2 * math.pi / 60)
    assert out["strain_nm_min"] == pytest.approx(4000.0)
    assert (out["type_L"], out["type_M"], out["type_H"]) == (0, 1, 0)


def test_add_features_does_not_modify_input():
    row = pd.DataFrame([{"type": "L", "air_temp_k": 299.0, "process_temp_k": 309.0,
                         "rpm": 1400, "torque_nm": 45.0, "tool_wear_min": 50}])
    before = row.copy()
    add_features(row)
    pd.testing.assert_frame_equal(row, before)


def test_no_label_leakage_in_features():
    leaky = set(FAILURE_MODES) | {"failure", "UDI", "Product ID"}
    assert not leaky & set(ALL_FEATURES)
    assert ALL_FEATURES == RAW_FEATURES + ENGINEERED


# ---------- Trained model -----------------------------------------------------------

def test_model_bundle_loads_and_predicts(dataset):
    bundle = joblib.load(MODEL_PATH)
    assert {"model", "features", "threshold"}.issubset(bundle)
    assert bundle["features"] == ALL_FEATURES
    assert 0 < bundle["threshold"] < 1

    df, _, _ = dataset
    probs = bundle["model"].predict_proba(df[bundle["features"]].head(200))[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()


def test_model_flags_high_risk_machine_over_normal_one():
    bundle = joblib.load(MODEL_PATH)
    rows = add_features(pd.DataFrame([
        {"type": "L", "air_temp_k": 300.0, "process_temp_k": 310.0,
         "rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 20},     # normal
        {"type": "L", "air_temp_k": 300.0, "process_temp_k": 310.0,
         "rpm": 1250, "torque_nm": 70.0, "tool_wear_min": 240},    # overstrained
    ]))
    normal, risky = bundle["model"].predict_proba(rows[bundle["features"]])[:, 1]
    assert risky > normal


# ---------- Reported results ---------------------------------------------------------

def test_reported_metrics_meet_targets():
    metrics = json.loads(METRICS_PATH.read_text())
    assert metrics["best_model"] == "XGBoost"
    best = metrics["comparison"]["XGBoost"]
    assert best["recall"] >= 0.80
    assert best["precision"] >= 0.90
    assert metrics["operations"]["recall"] >= 0.90
