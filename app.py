"""Step 6 (bonus): enter sensor readings, get a failure risk.  Run:  streamlit run app.py"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import add_features  # noqa: E402

MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"

st.set_page_config(page_title="Machine Failure Risk", page_icon="⚙️", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
model, feats, alert_thr = bundle["model"], bundle["features"], bundle["threshold"]

st.title("Machine Failure Risk")
st.caption("XGBoost model trained on the AI4I 2020 predictive maintenance dataset (10,000 machine cycles).")

c1, c2 = st.columns(2)
with c1:
    ptype = st.selectbox("Product quality type", ["L", "M", "H"],
                         help="L = low (50% of data), M = medium, H = high")
    air = st.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
    proc = st.slider("Process temperature (K)", 305.0, 314.0, 310.0, 0.1)
with c2:
    rpm = st.slider("Rotational speed (rpm)", 1150, 2900, 1500, 10)
    torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
    wear = st.slider("Tool wear (min)", 0, 255, 100, 1)

row = add_features(pd.DataFrame([{
    "type": ptype, "air_temp_k": air, "process_temp_k": proc,
    "rpm": rpm, "torque_nm": torque, "tool_wear_min": wear,
}]))
risk = float(model.predict_proba(row[feats])[:, 1][0])

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Failure risk", f"{risk:.1%}")
m2.metric("Alert threshold", f"{alert_thr:.0%}")
m3.metric("Power", f"{row.power_w.iloc[0]/1000:.1f} kW")

if risk >= alert_thr:
    st.error("⚠️ MAINTENANCE ALERT — schedule an inspection / tool change before the next run.")
else:
    st.success("✅ Normal — no maintenance action needed.")

# Plain-language flags based on the physics behind each failure mode
flags = []
if row.temp_diff_k.iloc[0] < 8.6 and rpm < 1380:
    flags.append("Low process–air temperature gap at low speed → heat dissipation risk")
p = row.power_w.iloc[0]
if p < 3500 or p > 9000:
    flags.append(f"Power {p:,.0f} W is outside the normal 3,500–9,000 W band → power failure risk")
strain_limit = {"L": 11000, "M": 12000, "H": 13000}[ptype]
if row.strain_nm_min.iloc[0] > strain_limit * 0.9:
    flags.append(f"Torque × tool wear = {row.strain_nm_min.iloc[0]:,.0f} (limit ≈ {strain_limit:,} for type {ptype}) → overstrain risk")
if wear >= 200:
    flags.append("Tool wear ≥ 200 min → tool is near end of life")
if flags:
    st.subheader("What's driving the risk")
    for f in flags:
        st.write("• " + f)

with st.expander("How the alert threshold was chosen"):
    st.write(
        f"The threshold ({alert_thr:.2f}) minimizes expected cost on cross-validated training data, "
        "assuming a missed failure costs $10,000 and each alert (planned inspection) costs $500. "
        "On the held-out test set it caught 62 of 68 failures (91%) with about 6.5 alerts per 100 cycles. "
        "See README for the full trade-off table."
    )
