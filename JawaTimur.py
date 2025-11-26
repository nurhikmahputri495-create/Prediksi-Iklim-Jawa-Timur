import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# =======================================
# KONFIGURASI FILE
# =======================================
FILE_PATH = "Data Jawa Timur_Putri Nurhikmah.xlsx"
SHEET_NAME = "Data Harian - Table"

# =======================================
# PAGE CONFIG
# =======================================
st.set_page_config(
    page_title="Prediksi Iklim Jawa Timur",
    layout="wide"
)

st.title("🌦️ Prediksi Iklim Jawa Timur 2025–2075")

# =======================================
# BACA DATA
# =======================================
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df = df.loc[:, ~df.columns.duplicated()]

if "kecepatan_angin" in df.columns:
    df.rename(columns={"kecepatan_angin": "FF_X"}, inplace=True)

df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["Tanggal"])
df["Tahun"] = df["Tanggal"].dt.year
df["Bulan"] = df["Tanggal"].dt.month

# =======================================
# VARIABEL IKLIM
# =======================================
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

# =======================================
# SIDEBAR KIRI — FILTER TAHUN
# =======================================
st.sidebar.title("📌 Menu Navigasi")
st.sidebar.subheader("Filter Data Historis")

tahun_tersedia = sorted(df["Tahun"].unique())
tahun_pilih = st.sidebar.slider(
    "Pilih rentang tahun:",
    min_value=int(min(tahun_tersedia)),
    max_value=int(max(tahun_tersedia)),
    value=(int(min(tahun_tersedia)), int(max(tahun_tersedia)))
)

df_filtered = df[(df["Tahun"] >= tahun_pilih[0]) & (df["Tahun"] <= tahun_pilih[1])]

# =======================================
# AGREGASI BULANAN
# =======================================
agg_dict = {v: "mean" for v in available_vars}
if "curah_hujan" in available_vars:
    agg_dict["curah_hujan"] = "sum"

monthly_df = df_filtered.groupby(["Tahun","Bulan"]).agg(agg_dict).reset_index()

# =======================================
# MODEL TRAIN
# =======================================
X = monthly_df[["Tahun","Bulan"]]
models = {}
metrics = {}

for var in available_vars:
    y = monthly_df[var]
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_ts)

    models[var] = model
    metrics[var] = {
        "RMSE": np.sqrt(mean_squared_error(y_ts, pred)),
        "R2": r2_score(y_ts, pred),
    }

# =======================================
# PREDIKSI 2025–2075
# =======================================
future = pd.DataFrame(
    [(y,m) for y in range(2025,2076) for m in range(1,13)],
    columns=["Tahun","Bulan"]
)
for var in available_vars:
    future[f"Pred_{var}"] = models[var].predict(future[["Tahun","Bulan"]])

# =======================================
# DATA VISUALISASI
# =======================================
monthly_df["Sumber"] = "Historis"
future["Sumber"] = "Prediksi"

full_plot = []
for var in available_vars:
    h = monthly_df[["Tahun","Bulan",var,"Sumber"]].rename(columns={var:"Nilai"})
    h["Variabel"] = label[var]

    p = future[["Tahun","Bulan",f"Pred_{var}","Sumber"]].rename(columns={f"Pred_{var}":"Nilai"})
    p["Variabel"] = label[var]

    full_plot.append(pd.concat([h,p]))

full_plot = pd.concat(full_plot)
full_plot["Tanggal"] = pd.to_datetime(
    full_plot["Tahun"].astype(str) + "-" + full_plot["Bulan"].astype(str) + "-01"
)

# =======================================
# LAYOUT KIRI – KANAN
# =======================================
col_kiri, col_kanan = st.columns([1.3, 2])

# =======================
# 📌 KIRI
# =======================
with col_kiri:
    st.subheader("📊 Ringkasan Data Bulanan")
    st.dataframe(monthly_df)

    st.subheader("📈 Evaluasi Model")
    for var, m in metrics.items():
        st.write(f"**{label[var]}** → RMSE: {m['RMSE']:.3f} | R²: {m['R2']:.3f}")

# =======================
# 📌 KANAN
# =======================
with col_kanan:
    st.subheader("📉 Grafik Tren Iklim")
    var_select = st.selectbox("Pilih Variabel:", [label[v] for v in available_vars])
    fig = px.line(
        full_plot[full_plot["Variabel"] == var_select],
        x="Tanggal",
        y="Nilai",
        color="Sumber",
        title=var_select
    )
    st.plotly_chart(fig, use_container_width=True)

# =======================================
# DOWNLOAD DATA
# =======================================
csv = future.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Prediksi 2025–2075",
    data=csv,
    file_name="prediksi_jawa_timur_2025_2075.csv",
    mime="text/csv"
)
