"""dashboard.py — Stage 06: interactive feature-exploration dashboard (Streamlit).

Companion to app.py. app.py is the clinician single-patient predictor; THIS is an
exploratory dashboard over the SYNTHETIC dataset so a supervisor/examiner can see
the features, their class separation, correlations, and the frozen model's live
predictions. It reads the Stage 01 dataset and the Stage 04 frozen model only —
it never trains and never touches real patient data.

Run:  streamlit run 06_gui/dashboard.py
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cvd.config import load_config  # noqa: E402

cfg = load_config()
POS = "#C44E52"   # CVD
NEG = "#4C72B0"   # no-CVD
CMAP = {0: NEG, 1: POS, "no-CVD": NEG, "CVD": POS}

st.set_page_config(page_title="CVD Synthetic Data Explorer", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load_data() -> pd.DataFrame:
    parquet = ROOT / cfg["data_generation"]["output"]
    if not parquet.exists():
        # Self-heal on a fresh host: the dataset is deterministic (seed 42), so
        # regenerating it is safe and fast (~1s) and yields the identical rows.
        from cvd.generate import generate
        generate()
    df = pd.read_parquet(parquet)
    df["label"] = df["cvd"].map({0: "no-CVD", 1: "CVD"})
    return df


@st.cache_resource
def load_model():
    p = ROOT / cfg["gui"]["model_path"]
    return joblib.load(p) if p.exists() else None


df = load_data()
model = load_model()
TARGET = "cvd"
features = [c for c in df.columns if c not in (TARGET, "label")]
num_cols = [c for c in features
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool]
cat_cols = [c for c in features if c not in num_cols]

# ---------------------------------------------------------------- sidebar filters
st.sidebar.title("🫀 Cohort filters")
st.sidebar.caption("Filter the synthetic cohort — every chart updates live.")

age_min, age_max = int(df["age"].min()), int(df["age"].max())
a_lo, a_hi = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
sex_pick = st.sidebar.multiselect("Sex", sorted(df["sex"].unique()),
                                  default=sorted(df["sex"].unique()))
smoke_vals = sorted(df["smoker"].unique(), key=str)
smoke_pick = st.sidebar.multiselect("Smoker", smoke_vals, default=smoke_vals)

mask = (df["age"].between(a_lo, a_hi) & df["sex"].isin(sex_pick)
        & df["smoker"].isin(smoke_pick))
fdf = df[mask]

st.sidebar.divider()
st.sidebar.metric("Patients in cohort", f"{len(fdf):,}")
prev = fdf[TARGET].mean() if len(fdf) else 0.0
st.sidebar.metric("CVD prevalence", f"{prev*100:.1f}%")
st.sidebar.caption("Synthetic data · seed "
                   f"{cfg['random_seed']} · not real patients.")

# ---------------------------------------------------------------- header
st.title("🫀 Cardiovascular Disease — Synthetic Feature Explorer")
st.caption("Interactive view of the medically-grounded synthetic dataset "
           "(10,000 patients, 17 features). **Synthetic data — not for clinical use.**")

if len(fdf) < 10:
    st.warning("Cohort too small — widen the filters in the sidebar.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔬 Feature Explorer", "🔗 Correlations",
    "⚖️ Class Separation", "🎯 Live Risk Predictor"])

# ============================================================ TAB 1: OVERVIEW
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(fdf):,}")
    c2.metric("CVD positive", f"{int(fdf[TARGET].sum()):,}", f"{prev*100:.1f}%")
    c3.metric("Features", len(features))
    c4.metric("Median age", f"{fdf['age'].median():.0f} yrs")

    left, right = st.columns([1, 1.3])
    with left:
        counts = fdf["label"].value_counts()
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.55,
            marker_colors=[CMAP[i] for i in counts.index],
            textinfo="label+percent"))
        fig.update_layout(title="Class balance", showlegend=False,
                          margin=dict(t=40, b=0, l=0, r=0), height=340)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Sample of the synthetic cohort**")
        st.dataframe(fdf[features + [TARGET]].head(200), height=300,
                     use_container_width=True)
        st.download_button(
            "⬇️ Download this cohort as CSV",
            fdf[features + [TARGET]].to_csv(index=False).encode("utf-8"),
            file_name="cvd_cohort.csv", mime="text/csv")

# ==================================================== TAB 2: FEATURE EXPLORER
with tab2:
    st.subheader("Explore any feature, split by outcome")
    feat = st.selectbox("Pick a feature", features, index=features.index("age"))

    if feat in num_cols:
        colA, colB = st.columns([1.4, 1])
        with colA:
            chart = st.radio("Chart", ["Histogram", "Violin", "Box"],
                             horizontal=True, key="numchart")
            if chart == "Histogram":
                fig = px.histogram(fdf, x=feat, color="label", barmode="overlay",
                                   nbins=40, opacity=0.7,
                                   color_discrete_map=CMAP,
                                   marginal="rug")
            elif chart == "Violin":
                fig = px.violin(fdf, y=feat, x="label", color="label", box=True,
                                points=False, color_discrete_map=CMAP)
            else:
                fig = px.box(fdf, y=feat, x="label", color="label",
                             color_discrete_map=CMAP)
            fig.update_layout(height=430, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            m0 = fdf.loc[fdf[TARGET] == 0, feat].mean()
            m1 = fdf.loc[fdf[TARGET] == 1, feat].mean()
            st.metric(f"Mean · no-CVD", f"{m0:.2f}")
            st.metric(f"Mean · CVD", f"{m1:.2f}", f"{m1 - m0:+.2f}")
            direction = "higher" if m1 > m0 else "lower"
            st.info(f"In this cohort, **{feat}** is **{direction}** in CVD-positive "
                    f"patients (Δmean = {m1 - m0:+.2f}).")
            sd = fdf[feat].std() + 1e-9
            st.caption(f"Standardized separation |Δ|/σ = {abs(m1 - m0)/sd:.2f}")
    else:
        prop = (fdf.groupby("label")[feat].value_counts(normalize=True)
                .rename("proportion").reset_index())
        prop[feat] = prop[feat].astype(str)
        fig = px.bar(prop, x=feat, y="proportion", color="label", barmode="group",
                     color_discrete_map=CMAP, text_auto=".0%")
        fig.update_layout(height=430, yaxis_tickformat=".0%", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Category proportions within each outcome group.")

# ======================================================== TAB 3: CORRELATIONS
with tab3:
    st.subheader("How the numeric features move together")
    corr = fdf[num_cols + [TARGET]].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=560, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Scatter — pick any two features**")
    s1, s2, s3 = st.columns(3)
    xf = s1.selectbox("X axis", num_cols, index=num_cols.index("age"))
    yf = s2.selectbox("Y axis", num_cols, index=num_cols.index("systolic_bp"))
    n = s3.slider("Points to plot", 200, min(4000, len(fdf)),
                  min(1500, len(fdf)), step=100)
    samp = fdf.sample(n, random_state=cfg["random_seed"])
    fig2 = px.scatter(samp, x=xf, y=yf, color="label", opacity=0.6,
                      color_discrete_map=CMAP)
    fig2.update_layout(height=460, legend_title_text="")
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================== TAB 4: CLASS SEPARATION
with tab4:
    st.subheader("Which features separate CVD from no-CVD the most?")
    st.caption("Standardized mean difference (Cohen's d style): "
               "(mean_CVD − mean_noCVD) / pooled SD. Bigger bars = stronger signal.")
    rows = []
    for c in num_cols:
        g0 = fdf.loc[fdf[TARGET] == 0, c]
        g1 = fdf.loc[fdf[TARGET] == 1, c]
        pooled = np.sqrt((g0.var() + g1.var()) / 2) + 1e-9
        rows.append({"feature": c, "smd": (g1.mean() - g0.mean()) / pooled})
    sep = pd.DataFrame(rows).sort_values("smd")
    fig = px.bar(sep, x="smd", y="feature", orientation="h",
                 color="smd", color_continuous_scale="RdBu_r",
                 range_color=[-1.5, 1.5])
    fig.update_layout(height=560, coloraxis_showscale=False,
                      xaxis_title="standardized mean difference (CVD − no-CVD)")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================== TAB 5: RISK PREDICTOR
with tab5:
    st.subheader("🎯 Live risk — frozen best model")
    if model is None:
        st.error("No frozen model found. Run `python -m cvd.run --stage all` first.")
    else:
        st.caption(f"Model: **{cfg['evaluation']['select_best_by']}**-best from Stage 04. "
                   "Move the controls and watch the prediction update.")
        inp = {}
        cols = st.columns(3)
        for i, c in enumerate(features):
            with cols[i % 3]:
                if c in num_cols:
                    lo, hi = float(df[c].min()), float(df[c].max())
                    med = float(df[c].median())
                    inp[c] = st.slider(c, round(lo, 1), round(hi, 1), round(med, 1))
                else:
                    opts = sorted(df[c].unique(), key=str)
                    inp[c] = st.selectbox(c, opts, key=f"pred_{c}")
        row = pd.DataFrame([inp])[features]
        # preserve training dtypes so the encoder recognises categories
        for c in cat_cols:
            row[c] = row[c].astype(df[c].dtype)
        proba = float(model.predict_proba(row)[:, 1][0])

        g1, g2 = st.columns([1.2, 1])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                number={"suffix": "%"},
                title={"text": "Estimated CVD risk"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": POS if proba >= 0.5 else NEG},
                       "steps": [{"range": [0, 50], "color": "#E8F0FE"},
                                 {"range": [50, 100], "color": "#FDE8E8"}],
                       "threshold": {"line": {"color": "black", "width": 3},
                                     "thickness": 0.75, "value": 50}}))
            gauge.update_layout(height=340, margin=dict(t=50, b=0))
            st.plotly_chart(gauge, use_container_width=True)
        with g2:
            if proba >= 0.5:
                st.error(f"### ⚠️ Elevated risk\n**{proba*100:.1f}%** predicted CVD risk")
            else:
                st.success(f"### ✅ Lower risk\n**{proba*100:.1f}%** predicted CVD risk")
            st.caption("Explanation drivers: risk rises with age, blood pressure, "
                       "glucose, BMI, ST-depression, smoking; falls with higher HDL, "
                       "more sleep and activity. See Stage 05 SHAP/LIME plots.")
        st.info("Synthetic-trained research prototype — **not for clinical use.**")
