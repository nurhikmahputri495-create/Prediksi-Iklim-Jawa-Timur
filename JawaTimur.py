import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

st.title("🌦️ Prediksi Iklim Jawa Timur 2025–2075")

# ================================
# 0. UPLOAD FILE
# ================================
uploaded_file = st.file_uploader(
    "📂 Upload File Excel Data Iklim",
    type=["xlsx"]
)

if uploaded_file:

    # ================================
    # 1. AMBIL NAMA WILAYAH DARI FILE
    # ================================
    file_name = uploaded_file.name.replace(".xlsx", "").replace("_", " ")
    wilayah = " ".join([w for w in file_name.split() if w.lower() not in ["data","table","harian"]])

    st.subheader(f"📍 Data Iklim Wilayah **{wilayah}**")

    # ================================
    # 2. BACA DATA
    # ================================
    df = pd.read_excel(uploaded_file, sheet_name="Data Harian - Table")

    # Penanganan duplikat kolom
    df = df.loc[:, ~df.columns.duplicated()]

    # Rename wind speed jika nama beda
    if "kecepatan_angin" in df.columns:
        df.rename(columns={"kecepatan_angin": "FF_X"}, inplace=True)

    # Parsing tanggal
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month

    # ================================
    # 3. VARIABEL IKLIM
    # ================================
    possible_vars = [
        "Tn","Tx","Tavg","kelembaban",
        "curah_hujan","matahari",
        "FF_X","DDD_X"
    ]
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

    # ================================
    # 4. AGREGASI MENJADI BULANAN
    # ================================
    agg_dict = {v: "mean" for v in available_vars}
    if "curah_hujan" in available_vars:
        agg_dict["curah_hujan"] = "sum"

    monthly_df = df.groupby(["Tahun","Bulan"]).agg(agg_dict).reset_index()

    st.write("📊 **Data Ringkas Bulanan**")
    st.dataframe(monthly_df)

    # ================================
    # 5. TRAIN MODEL RF
    # ================================
    X = monthly_df[['Tahun','Bulan']]
    models = {}
    metrics = {}

    for var in available_vars:
        y = monthly_df[var]

        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_ts)

        models[var] = model
        metrics[var] = {
            "RMSE": np.sqrt(mean_squared_error(y_ts, pred)),
            "R2": r2_score(y_ts, pred)
        }

    st.subheader("📈 Evaluasi Model")
    for var, m in metrics.items():
        st.write(
            f"**{label[var]}** → RMSE: **{m['RMSE']:.3f}** | R²: **{m['R2']:.3f}**"
        )

    # ================================
    # 6. PREDIKSI 2025–2075
    # ================================
    future = pd.DataFrame(
        [(y,m) for y in range(2025,2076) for m in range(1,13)],
        columns=["Tahun","Bulan"]
    )

    for var in available_vars:
        future[f"Pred_{var}"] = models[var].predict(future[['Tahun','Bulan']])

    # ================================
    # 7. GRAFIK TREN
    # ================================
    st.subheader(f"📊 Grafik Tren Iklim **{wilayah}**")

    monthly_df['Sumber'] = 'Historis'
    future['Sumber'] = 'Prediksi'

    full_plot = []
    for var in available_vars:
        hist = monthly_df[['Tahun','Bulan',var,'Sumber']].rename(columns={var:'Nilai'})
        hist['Variabel'] = label[var]

        pred = future[['Tahun','Bulan',f'Pred_{var}','Sumber']].rename(columns={f'Pred_{var}':'Nilai'})
        pred['Variabel'] = label[var]

        full_plot.append(pd.concat([hist,pred]))

    full_plot = pd.concat(full_plot)
    full_plot['Tanggal'] = pd.to_datetime(
        full_plot['Tahun'].astype(str) + "-" + full_plot['Bulan'].astype(str) + "-01"
    )

    chosen = st.selectbox("📌 Pilih Variabel Iklim", [label[v] for v in available_vars])

    fig = px.line(
        full_plot[full_plot['Variabel']==chosen],
        x='Tanggal',
        y='Nilai',
        color='Sumber',
        title=f"{chosen} — {wilayah}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ================================
    # 8. DOWNLOAD CSV HASIL
    # ================================
    csv = future.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Prediksi Iklim 2025–2075",
        data=csv,
        file_name=f"prediksi_{wilayah.lower().replace(' ','_')}_2025_2075.csv",
        mime="text/csv"
    )
