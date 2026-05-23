import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Analisis Saham Top Tech",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ANALISIS PERGERAKAN SAHAM DAN KINERJA TOP TECH COMPANIES")

st.markdown("""
### **Business Understanding**
Permasalahan utama yang diangkat pada proyek ini adalah tingginya kompleksitas dan volatilitas pergerakan harga saham perusahaan teknologi di pasar modal. Menganalisis metrik historis saham ini secara manual sangatlah tidak efisien. Oleh karena itu, diperlukan pendekatan komputasi berbasis data menggunakan *Machine Learning* dan *Time-Series Forecasting* untuk memetakan perilaku pasar.
""")

# ==========================================
# 2. FUNGSI LOAD DATA
# ==========================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date'])
    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
    df['Volatility'] = df['High'] - df['Low']
    return df

st.sidebar.header("📂 1. Upload Dataset")
st.sidebar.markdown("Unggah `Dataset 10 Perusahaan Besar dibidang Teknologi.csv`")
uploaded_file = st.sidebar.file_uploader("", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    st.sidebar.markdown("---")
    st.sidebar.header("📅 2. Filter Rentang Waktu (EDA)")
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    start_date, end_date = st.sidebar.slider("Pilih Rentang Tanggal:", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    df_eda = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)].copy()

    with st.expander("👁️ Lihat 5 Baris Pertama Dataset Mentah"):
        st.dataframe(df.head())
    st.markdown("---")
    
    # ==========================================
    # 3. BAGIAN EDA
    # ==========================================
    st.header("📊 Exploratory Data Analysis (EDA)")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Distribusi & Outlier", "Tren & Moving Average", "Korelasi Heatmap", 
        "Distribusi Volatilitas", "Volume vs Volatilitas", "Cumulative Return", "Likuiditas"
    ])
    
    with tab1:
        st.subheader("Distribusi Volume Perdagangan")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df_eda['Volume'], bins=50, kde=True, color='teal', ax=ax)
        st.pyplot(fig)
        st.markdown("---")
        st.subheader("Analisis Outlier (Volume Berdasarkan Ticker)")
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_eda, x='Ticker', y='Volume', palette='Set2', ax=ax4)
        st.pyplot(fig4)
        
    with tab2:
        st.subheader(f"Tren Harga Saham ({start_date} hingga {end_date})")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_eda, x='Date', y='Close', hue='Ticker', linewidth=1.5, ax=ax2)
        st.pyplot(fig2)
        st.markdown("---")
        st.subheader("Analisis Moving Average (MA)")
        ticker_ma = st.selectbox("Pilih Ticker:", df_eda['Ticker'].unique(), index=0)
        df_ma = df_eda[df_eda['Ticker'] == ticker_ma].copy()
        df_ma['MA_50'] = df_ma['Close'].rolling(window=50).mean()
        df_ma['MA_200'] = df_ma['Close'].rolling(window=200).mean()
        fig_ma, ax_ma = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_ma, x='Date', y='Close', label='Harga Asli', alpha=0.5, ax=ax_ma)
        sns.lineplot(data=df_ma, x='Date', y='MA_50', label='MA 50 Hari', color='blue', ax=ax_ma)
        sns.lineplot(data=df_ma, x='Date', y='MA_200', label='MA 200 Hari', color='red', ax=ax_ma)
        st.pyplot(fig_ma)
        
    with tab3:
        st.subheader("Heatmap Korelasi Variabel Numerik")
        kolom_numerik = df_eda[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.heatmap(kolom_numerik.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
        st.pyplot(fig3)
        st.markdown("---")
        st.subheader("Matriks Korelasi Pergerakan Harga (Daily Return)")
        daily_return_pivot = df_eda.pivot_table(index='Date', columns='Ticker', values='Daily_Return')
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(daily_return_pivot.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
        st.pyplot(fig_corr)

    with tab4:
        st.subheader("Distribusi Daily Return (Volatilitas Risiko)")
        fig_vio, ax_vio = plt.subplots(figsize=(14, 7))
        sns.violinplot(data=df_eda.dropna(subset=['Daily_Return']), x='Ticker', y='Daily_Return', hue='Ticker', palette='Set3', inner='quartile', legend=False, ax=ax_vio)
        st.pyplot(fig_vio)

    with tab5:
        st.subheader("Hubungan Volume Perdagangan dan Volatilitas Harga Harian")
        fig_scat, ax_scat = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_eda.sample(n=min(5000, len(df_eda)), random_state=42), x='Volume', y='Volatility', hue='Ticker', alpha=0.6, palette='tab10', ax=ax_scat)
        ax_scat.set_xscale('log')
        st.pyplot(fig_scat)

    with tab6:
        st.subheader("Cumulative Return (Pertumbuhan Investasi)")
        avg_close = df_eda.groupby('Ticker')['Close'].mean().sort_values(ascending=False).head(5).index.tolist()
        df_top5 = df_eda[df_eda['Ticker'].isin(avg_close)].copy()
        df_top5['Cumulative_Return'] = df_top5.groupby('Ticker')['Daily_Return'].transform(lambda x: (1 + x).cumprod() - 1)
        fig_cum, ax_cum = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_top5, x='Date', y='Cumulative_Return', hue='Ticker', ax=ax_cum)
        ax_cum.yaxis.set_major_formatter(PercentFormatter(1))
        st.pyplot(fig_cum)

    with tab7:
        st.subheader("Proporsi Likuiditas (Total Volume) per Emiten")
        liquidity_dist = df_eda.groupby('Ticker')['Volume'].sum().sort_values(ascending=False)
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        ax_pie.pie(liquidity_dist, labels=liquidity_dist.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85, colors=sns.color_palette('Set2', len(liquidity_dist)))
        fig_pie.gca().add_artist(plt.Circle((0,0),0.70,fc='white'))
        st.pyplot(fig_pie)

    st.markdown("---")
    
    # ==========================================
    # 4. PREPROCESSING & MACHINE LEARNING
    # ==========================================
    df_ml = df.copy().dropna()
    le = LabelEncoder()
    df_ml['Ticker'] = le.fit_transform(df_ml['Ticker'])
    numerik = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    scaler = StandardScaler()
    df_ml[numerik] = scaler.fit_transform(df_ml[numerik])

    st.header("🤖 Model Machine Learning & Forecasting")
    tab_clf, tab_reg, tab_sim, tab_fut = st.tabs(["Klasifikasi (Arah Harga)", "Regresi (H+1)", "🎮 Simulasi H+1", "🔮 Prediksi Masa Depan"])
    
    with tab_clf:
        st.subheader("Evaluasi Klasifikasi (Random Forest)")
        df_clf = df_ml.copy()
        df_clf['Target'] = (df_clf['Close'] > df_clf['Open']).astype(int)
        X_clf = df_clf.drop(columns=['Close', 'Date', 'Target'])
        y_clf = df_clf['Target']
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
        rf_clf = RandomForestClassifier(random_state=42).fit(X_train_c, y_train_c)
        st.success(f"**Akurasi Model: {accuracy_score(y_test_c, rf_clf.predict(X_test_c)):.2%}**")
        
    with tab_reg:
        st.subheader("Evaluasi Regresi (Random Forest)")
        df_reg = df_ml.copy()
        df_reg['Close_H+1'] = df_reg.groupby('Ticker')['Close'].shift(-1)
        df_reg = df_reg.dropna(subset=['Close_H+1'])
        X_reg = df_reg.drop(columns=['Close_H+1', 'Date'])
        y_reg = df_reg['Close_H+1']
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        rf_reg = RandomForestRegressor(random_state=42).fit(X_train_r, y_train_r)
        
        fig_reg, ax_reg = plt.subplots(figsize=(12, 6))
        ax_reg.plot(y_test_r.values[:80], label='Aktual (Biru)', marker='o', color='blue', alpha=0.7)
        ax_reg.plot(rf_reg.predict(X_test_r)[:80], label='Prediksi (Oranye)', marker='x', linestyle='--', color='orange')
        ax_reg.legend()
        st.pyplot(fig_reg)

    with tab_sim:
        st.subheader("🎮 Simulasi: Tebak Harga Esok Hari!")
        col1, col2, col3 = st.columns(3)
        with col1:
            sim_ticker = st.selectbox("Pilih Emiten:", df['Ticker'].unique(), key='sim_ticker')
            sim_open = st.number_input("Open $:", value=150.0)
        with col2:
            sim_high = st.number_input("High $:", value=155.0)
            sim_low = st.number_input("Low $:", value=148.0)
        with col3:
            sim_close = st.number_input("Close Hari Ini $:", value=152.0)
            sim_vol = st.number_input("Volume:", value=50000000)
            
        if st.button("🚀 Prediksi Harga Besok!"):
            input_data = pd.DataFrame({'Ticker': [sim_ticker], 'Open': [sim_open], 'High': [sim_high], 'Low': [sim_low], 'Close': [sim_close], 'Adj Close': [sim_close], 'Volume': [sim_vol], 'Daily_Return': [0.0], 'Volatility': [sim_high - sim_low]})
            input_data['Ticker'] = le.transform(input_data['Ticker'])
            input_data[numerik] = scaler.transform(input_data[numerik])
            input_data = input_data[X_reg.columns]
            
            prediksi_scaled = rf_reg.predict(input_data)[0]
            dummy_array = np.zeros((1, len(numerik)))
            dummy_array[0, numerik.index('Close')] = prediksi_scaled
            prediksi_asli = scaler.inverse_transform(dummy_array)[0, numerik.index('Close')]
            
            st.success("Selesai!")
            st.metric(label=f"Prediksi Close {sim_ticker} Besok", value=f"${prediksi_asli:.2f}", delta=f"{(prediksi_asli - sim_close):.2f} dari hari ini")
            st.balloons()

    # --- TAB PREDIKSI JANGKA PANJANG DENGAN PENJELASAN DINAMIS ---
    with tab_fut:
        st.subheader("🔮 Prediksi Proyeksi Jangka Panjang")
        st.markdown("Algoritma **Exponential Smoothing / Time-Series** digunakan untuk memprediksi tren masa depan jangka panjang berdasarkan pola data historis.")
        
        col_fut1, col_fut2 = st.columns(2)
        with col_fut1:
            future_ticker = st.selectbox("Pilih Emiten yang Ingin Diprediksi:", df['Ticker'].unique(), key='future_ticker')
        with col_fut2:
            target_year = st.slider("Pilih Tahun Target Prediksi:", min_value=2024, max_value=2035, value=2026)
        
        if st.button(f"🚀 Ramalkan Tren hingga {target_year}!"):
            with st.spinner('Sedang melatih model mesin waktu...'):
                df_ts = df[df['Ticker'] == future_ticker][['Date', 'Close']].copy()

                # Pastikan format datetime
                df_ts['Date'] = pd.to_datetime(df_ts['Date'])

                # Set index
                df_ts = df_ts.set_index('Date')

                # Resample mingguan
                df_ts_weekly = df_ts.resample('W').mean()

                # Hapus NaN
                df_ts_weekly = df_ts_weekly.dropna()

                # Validasi jumlah data
                if len(df_ts_weekly) < 2:
                    st.error("Data historis tidak cukup untuk melakukan forecasting Time-Series.")
                    st.stop()

                # Training model
                model_hw = ExponentialSmoothing(
                df_ts_weekly['Close'],
                trend='add',
                seasonal=None,
                initialization_method="estimated"
                )

                fit_model = model_hw.fit()

                last_date = df_ts_weekly.index[-1]
                target_date = pd.to_datetime(f'{target_year}-12-31')
                weeks_to_predict = int((target_date - last_date).days / 7)
                
                if weeks_to_predict > 0:
                    forecast = fit_model.forecast(weeks_to_predict)
                    forecast_index = pd.date_range(start=last_date + datetime.timedelta(days=7), periods=weeks_to_predict, freq='W')
                    
                    fig_fut, ax_fut = plt.subplots(figsize=(12, 6))
                    ax_fut.plot(df_ts_weekly.index, df_ts_weekly['Close'], label='Data Historis (Aktual)', color='blue')
                    ax_fut.plot(forecast_index, forecast, label=f'Proyeksi Masa Depan (Hingga {target_year})', color='red', linestyle='--')
                    ax_fut.fill_between(forecast_index, forecast * 0.90, forecast * 1.10, color='red', alpha=0.1, label='Rentang Variansi')
                    
                    ax_fut.set_title(f"Proyeksi Tren Harga Saham {future_ticker} ke {target_year}", fontweight='bold', fontsize=14)
                    ax_fut.set_xlabel("Tahun")
                    ax_fut.set_ylabel("Harga (USD)")
                    ax_fut.legend()
                    st.pyplot(fig_fut)
                    st.success(f"Berhasil meramalkan tren hingga akhir tahun {target_year}!")
                    
                    # -------------------------------------------------------------
                    # --- FITUR BARU: GENERASI PENJELASAN HASIL OTOMATIS (DINAMIS) ---
                    # -------------------------------------------------------------
                    st.markdown("### 📝 Analisis & Kesimpulan Hasil Proyeksi")
                    
                    # Mengambil nilai penting untuk narasi
                    harga_terakhir = df_ts_weekly['Close'].iloc[-1]
                    harga_prediksi = forecast.iloc[-1]
                    persen_perubahan = ((harga_prediksi - harga_terakhir) / harga_terakhir) * 100
                    
                    # Menentukan status tren
                    if persen_perubahan > 5:
                        status_tren = "**Bullish (Cenderung Naik Target Jangka Panjang)** 🟢"
                        rekomendasi = "Investor dapat mempertimbangkan opsi **Hold** atau **Buy on Weakness** karena model mendeteksi adanya kekuatan tren positif yang berkelanjutan didorong oleh pertumbuhan fundamental sektor teknologi."
                    elif persen_perubahan < -5:
                        status_tren = "**Bearish (Cenderung Turun / Koreksi)** 🔴"
                        rekomendasi = "Disarankan untuk lebih **Berhati-hati (Wait and See)** atau melakukan profit-taking parsial karena model mendeteksi adanya indikasi kejenuhan pasar atau potensi koreksi tren jangka panjang."
                    else:
                        status_tren = "**Konsolidasi (Stagnan / Sideways)** 🟡"
                        rekomendasi = "Pergerakan harga diproyeksikan stabil dalam rentang harga saat ini. Cocok untuk strategi **Swing Trading** jangka pendek memanfaatkan riwayat volatilitas hariannya."
                    
                    # Menampilkan Box Informasi Penjelasan yang dinamis
                    st.info(f"""
                    **Hasil Analisis Model Eksponensial untuk Emiten {future_ticker}:**
                    * **Harga Historis Terakhir:** `${harga_terakhir:.2f}`
                    * **Proyeksi Harga Akhir ({target_year}):** `${harga_prediksi:.2f}`
                    * **Estimasi Persentase Perubahan:** `{persen_perubahan:.2f}%`
                    * **Kondisi Tren Masa Depan:** {status_tren}
                    
                    **Business Insight & Rekomendasi:**
                    Berdasarkan visualisasi deret waktu di atas, garis merah putus-putus menggambarkan arah proyeksi emiten **{future_ticker}** hingga tahun **{target_year}**. {rekomendasi} Kompleksitas pasar modal di sektor teknologi sangat reaktif terhadap volume likuiditas, sehingga rentang variansi (area bayangan merah transparan) harus tetap diperhatikan sebagai batas risiko toleransi volatilitas.
                    """)
                    
                else:
                    st.warning("Tahun target yang dipilih terlalu dekat atau sudah terlewati oleh data historis.")

else:
    st.info("👈 Silakan upload file dataset CSV Anda di panel sebelah kiri untuk memuat dashboard analisis.")