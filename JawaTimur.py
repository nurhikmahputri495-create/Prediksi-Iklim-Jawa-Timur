if uploaded_file:
    # ======================== 0. IDENTIFIKASI WILAYAH ========================
    file_name = uploaded_file.name.lower()

    if "jawa timur" in file_name or "jatim" in file_name:
        wilayah = "Jawa Timur"
    else:
        wilayah = uploaded_file.name.replace(".xlsx","").replace("_"," ")

    st.subheader(f"📍 Data Iklim Wilayah **{wilayah}**")

    # ================================ 
    # 1. BACA DATA
    # ================================
    df = pd.read_excel(uploaded_file, sheet_name='Data Harian - Table')

    # handle duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # mapping nama untuk konsistensi
    if "kecepatan_angin" in df.columns:
        df = df.rename(columns={"kecepatan_angin": "FF_X"})

    # perbaiki parsing tanggal
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=["Tanggal"])

    df['Tahun'] = df['Tanggal'].dt.year
    df['Bulan'] = df['Tanggal'].dt.month

    # ================================ 
    # 2. VARIABLE YANG DIPAKAI
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
    # 3. AGREGASI BULANAN
    # ================================
    agg_dict = {v:'mean' for v in available_vars}
    if "curah_hujan" in available_vars:
        agg_dict["curah_hujan"] = "sum"   # hujan akumulatif

    monthly_df = df.groupby(["Tahun","Bulan"]).agg(agg_dict).reset_index()
    st.subheader("📊 Data Bulanan")
    st.dataframe(monthly_df)

    # ================================ 
    # 4. TRAIN MODEL
    # ================================
    X = monthly_df[['Tahun', 'Bulan']]
    models = {}; metrics = {}

    for var in available_vars:
        y = monthly_df[var]
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_ts)

        models[var] = model
        metrics[var] = {
            "RMSE": np.sqrt(mean_squared_error(y_ts, pred)),
            "R2": r2_score(y_ts, pred)
        }

    st.subheader("📈 Evaluasi Model Machine Learning")
    for var,m in metrics.items():
        st.write(f"**{label[var]}** → RMSE: {m['RMSE']:.3f} | R²: {m['R2']:.3f}")

    # ================================ 
    # 5. PREDIKSI OTOMATIS 2025–2075
    # ================================
    future = pd.DataFrame(
        [(y,m) for y in range(2025,2076) for m in range(1,13)],
        columns=["Tahun","Bulan"]
    )
    for var in available_vars:
        future[f"Pred_{var}"] = models[var].predict(future[['Tahun','Bulan']])

    # ================================ 
    # 6. GABUNG HISTORIS + PREDIKSI
    # ================================
    monthly_df['Sumber'] = 'Historis'
    future['Sumber']  = 'Prediksi'

    merge_list = []
    for var in available_vars:
        hist = monthly_df[['Tahun','Bulan',var,'Sumber']].rename(columns={var:'Nilai'})
        hist['Variabel'] = label[var]

        pred = future[['Tahun','Bulan',f'Pred_{var}','Sumber']].rename(columns={f'Pred_{var}':'Nilai'})
        pred['Variabel'] = label[var]

        merge_list.append(pd.concat([hist,pred]))

    full_plot = pd.concat(merge_list)

    full_plot['Tanggal'] = pd.to_datetime(
        full_plot['Tahun'].astype(str) + "-" + full_plot['Bulan'].astype(str) + "-01"
    )

    # ================================
    # 6B. GRAFIK OTOMATIS TOP 4
    # ================================
    st.subheader("📉 Grafik Otomatis Variabel Utama Jawa Timur")

    default_vars = ["Tavg","Tx","Tn","curah_hujan"]
    default_vars = [v for v in default_vars if v in available_vars]

    for var in default_vars:
        subset = full_plot[full_plot["Variabel"] == label[var]]
        fig_auto = px.line(
            subset,
            x="Tanggal",
            y="Nilai",
            color="Sumber",
            title=f"{label[var]} — {wilayah}"
        )
        st.plotly_chart(fig_auto, use_container_width=True)

    # ================================ 
    # 7. GRAFIK PILIH VARIABEL
    # ================================
    st.subheader("📊 Grafik Variabel Pilihan")

    chosen = st.selectbox("Pilih Variabel Cuaca", [label[v] for v in available_vars])

    fig = px.line(
        full_plot[full_plot['Variabel']==chosen],
        x='Tanggal',
        y='Nilai',
        color='Sumber',
        title=f"{chosen} — {wilayah}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ================================ 
    # 8. DOWNLOAD DATA
    # ================================
    csv = future.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Prediksi Iklim 2025–2075",
        data=csv,
        file_name=f"prediksi_{wilayah.lower().replace(' ','_')}_2025_2075.csv",
        mime="text/csv"
    )
