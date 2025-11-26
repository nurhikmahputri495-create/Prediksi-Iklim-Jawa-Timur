import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==================================
# --- CONFIG PAGE ---
# ==================================
st.set_page_config(
    page_title="🌦️ Dashboard Prediksi Iklim Jawa Timur",
    layout="wide"
)

# ==================================
# --- LOAD DATA ---
# ==================================
FILE_PATH = "Data Jawa Timur_Putri Nurhikmah.xlsx"
SHEET_NAME = "Data Harian - Table"

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df = df.loc[:, ~df.columns.duplicated()]
df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)

# Rename kolom raw
if "kecepatan_angin" in df.columns:
    df.rename(columns={"kecepatan_angin": "FF_X"}, inplace=True)

df["Tahun"] = df["Tanggal"].dt.year
df["Bulan"] = df["Tanggal"].dt.month

# ==================================
# --- LABEL VARIABEL ---
# ==================================
possible_vars = ["Tn","Tx","Tavg","kelembaban","curah_hujan","matahari","FF_X","DDD_X"]
available_vars = [v for v in possible_vars if v in df.columns]

label = {
    "Tn": "Suhu Minimum (°C)",
    "Tx": "Suhu Maksimum (°C)",
    "Tavg": "Suhu Rata-rata (°C)",
    "kelembaban": "Kelembaban Udara (%)",
    "curah_hujan": "Curah Hujan (mm)",
    "matahari": "Durasi Penyinaran Matahari (jam)",
    "FF_X": "Kecepatan Angin Maksimum (m/s)",
    "DDD_X": "Arah Angin saat Kecepatan Maksimum (°)"
}

# =====================================================
# ========== SIDEBAR =================================
# =====================================================

st.sidebar.title("⚙️ Filter Dashboard")
tahun_list = sorted(df["Tahun"].unique())

year_range = st.sidebar.slider(
    "Pilih rentang Tahun",
    min_value=int(min(tahun_list)),
    max_value=int(max(tahun_list)),
    value=(int(min(tahun_list)), int(max(tahun_list)))
)

bulan_list = st.sidebar.multiselect(
    "Filter Bulan",
    list(range(1,13)),
    default=list(range(1,13))
)

selected_vars = st.sidebar.multiselect(
    "Variabel yang ditampilkan",
    [label[v] for v in available_vars],
    default=[label["Tavg"], label["curah_hujan"]]
)

chart_mode = st.sidebar.radio(
    "Mode Grafik",
    ["Line Chart", "Scatter", "Bar Chart"]
)

rolling = st.sidebar.selectbox(
    "Smooth Trend (Moving Average)",
    [None, 3, 6, 12],
    format_func=lambda x: "Tidak" if x is None else f"{x} Bulan"
)

highlight_top = st.sidebar.checkbox("Tandai Nilai Ekstrim (Top 5)")

# =====================================================
# ========== FILTER DATA =============================
# =====================================================
df = df[(df["Tahun"] >= year_range[0]) & (df["Tahun"] <= year_range[1])]
df = df[df["Bulan"].isin(bulan_list)]

# =====================================================
# ========== AGREGASI MONTHLY ========================
# =====================================================
agg_dict = {v:"mean" for v in available_vars}
if "curah_hujan" in available_vars:
    agg_dict["curah_hujan"] = "sum"

monthly = df.groupby(["Tahun","Bulan"]).agg(agg_dict).reset_index()
monthly["Tanggal"] = pd.to_datetime(
    monthly['Tahun'].astype(str) + "-" + monthly['Bulan'].astype(str) + "-01"
)

# =====================================================
# ========== HEADER DASHBOARD ========================
# =====================================================

st.title("🌦️ Dashboard Analitik & Prediksi Iklim — Jawa Timur")

# ===================== CARDS INFO ====================
colA, colB, colC = st.columns(3)

with colA:
    st.metric("🌡️ Rata-rata Suhu", f"{monthly['Tavg'].mean():.2f} °C")

with colB:
    st.metric("🌧️ Total Hujan", f"{monthly['curah_hujan'].sum():.0f} mm")

with colC:
    st.metric("💨 Angin Maksimum", f"{monthly['FF_X'].max():.2f} m/s")

st.write("---")

# =====================================================
# ========== MODEL TRAIN =============================
# =====================================================
X = monthly[["Tahun","Bulan"]]
models = {}
metrics = {}

for var in available_vars:
    y = monthly[var]
    X_tr,X_ts,y_tr,y_ts = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(200,random_state=42)
    model.fit(X_tr,y_tr)
    pred = model.predict(X_ts)

    models[var]=model
    metrics[var] = {
        "RMSE": np.sqrt(mean_squared_error(y_ts,pred)),
        "R2": r2_score(y_ts,pred)
    }

# =====================================================
# ========== PREDIKSI FUTURE ==========================
# =====================================================

future = pd.DataFrame(
    [(y,m) for y in range(2025,2076) for m in range(1,13)],
    columns=["Tahun","Bulan"]
)
for v in available_vars:
    future[f"Pred_{v}"] = models[v].predict(future[["Tahun","Bulan"]])

future["Tanggal"] = pd.to_datetime(
    future['Tahun'].astype(str) + "-" + future['Bulan'].astype(str) + "-01"
)

# =====================================================
# ========== VISUALISASI GRAFIK ======================
# =====================================================

st.subheader("📊 Grafik Tren Iklim")

var_show = [k for k,v in label.items() if v in selected_vars]

for var in var_show:
    data_hist = monthly.copy()
    data_pred = future.copy()
    data_hist["Sumber"]="Historis"
    data_pred["Sumber"]="Prediksi"
    data_pred.rename(columns={f"Pred_{var}:":var}, inplace=False)

combined = pd.concat([
    data_hist[["Tanggal",var,"Sumber"]],
    data_pred[["Tanggal",f"Pred_{var}","Sumber"]]\
        .rename(columns={f"Pred_{var}":var})
])

if rolling:
    combined[var] = combined[var].rolling(rolling,min_periods=1).mean()

# --- CHART MODE ---
if chart_mode=="Line Chart":
    fig = px.line(combined, x="Tanggal", y=var, color="Sumber", title=label[var])
elif chart_mode=="Bar Chart":
    fig = px.bar(combined, x="Tanggal", y=var, color="Sumber", title=label[var])
else:
    fig = px.scatter(combined, x="Tanggal", y=var, color="Sumber", title=label[var])

st.plotly_chart(fig,use_container_width=True)

# =====================================================
# ========== CORRELATION HEATMAP ======================
# =====================================================
st.subheader("🔬 Korelasi Antar Variabel")
fig2, ax = plt.subplots(figsize=(8,4))
sns.heatmap(monthly[available_vars].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig2)

# =====================================================
# ========== ERROR DISTRIBUTION ======================
# =====================================================
st.subheader("📐 Distribusi Error Model")
err_df = pd.DataFrame([
    {"Variabel":label[v], "R2":metrics[v]["R2"], "RMSE":metrics[v]["RMSE"]}
    for v in available_vars
])
st.dataframe(err_df)

# =====================================================
# ========== TABLE & DOWNLOAD ========================
# =====================================================
st.subheader("📁 Dataset Historis (setelah filter)")
st.dataframe(df)

st.subheader("📥 Download dataset prediksi")
csv = future.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "prediksi_iklim_jatim.csv")
