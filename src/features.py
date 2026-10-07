"""Shared data loading and feature engineering for the AI4I 2020 dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"

RENAME = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
    "Type": "type",
}

# Raw sensor inputs (what a machine would actually report)
RAW_FEATURES = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min",
                "type_L", "type_M", "type_H"]

# Physics-based features an engineer would derive from the raw readings
ENGINEERED = ["temp_diff_k", "power_w", "strain_nm_min"]
ALL_FEATURES = RAW_FEATURES + ENGINEERED

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig").rename(columns=RENAME)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot product type and engineered features. Works on a single-row frame too."""
    df = df.copy()
    for t in ["L", "M", "H"]:
        df[f"type_{t}"] = (df["type"] == t).astype(int)
    # Heat dissipation: smaller gap between process and air temp = less cooling
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
    # Mechanical power = torque x angular velocity
    df["power_w"] = df["torque_nm"] * df["rpm"] * 2 * np.pi / 60
    # Overstrain proxy = tool wear x torque
    df["strain_nm_min"] = df["tool_wear_min"] * df["torque_nm"]
    return df


def load_dataset():
    """Returns features DataFrame, target Series, and the failure-mode flags (for analysis only)."""
    df = add_features(load_raw())
    # UDI / Product ID are identifiers; TWF..RNF are the failure labels themselves
    # (using them as inputs would leak the answer), so they are excluded from X.
    return df, df["failure"], df[FAILURE_MODES]
