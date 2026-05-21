import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Analisis Saham Top Tech",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ANALISIS PERGERAKAN SAHAM DAN KINERJA TOP TECH COMPANIES DI CAPITAL MARKET")

st.markdown("""
### **Business Understanding**
Permasalahan utama yang diangkat pada proyek ini adalah tingginya kompleksitas dan volatilitas pergerakan harga saham perusahaan-perusahaan raksasa di sektor teknologi (Top Tech Companies) di pasar modal. Sektor teknologi memiliki dinamika yang sangat reaktif terhadap sentimen pasar, yang tercermin dari fluktuasi harga harian (Open, High, Low, Close) dan likuiditas volume transaksi. Menganalisis metrik historis saham ini secara manual untuk mengidentifikasi pola tren jangka panjang dan mengukur tingkat risiko investasi sangatlah tidak efisien dan rentan terhadap bias.

Oleh karena itu, diperlukan suatu pendekatan komputasi berbasis data untuk membantu investor dan pemangku kepentingan dalam memetakan perilaku pasar. Hal ini dapat diatasi dengan mengimplementasikan algoritma Machine Learning dan Time Series Forecasting.
""")

# ==========================================
# 2. FUNGSI LOAD DATA
# ==========================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date'])
    
    # Feature Engineering (Sesuai Colab)
    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
    df['Volatility'] = df['High'] - df['Low']
    
    return df

st.sidebar.header("📂 Upload Dataset")
st.sidebar.markdown("Unggah file `Dataset 10 Perusahaan Besar dibidang Teknologi.csv` milik kelompok Anda.")
uploaded_file = st.sidebar.file_uploader("", type="csv")

if uploaded_file is not None:
    # Memuat data ke DataFrame utama
    df = load_data(uploaded_file)
    
    st.markdown("""
    ---
    ### **Data Understanding & Kualitas Data**
    Dataset yang digunakan dalam analisis ini memuat data riwayat pergerakan harga saham harian dari perusahaan-perusahaan besar, khususnya raksasa teknologi. Tidak ditemukan adanya nilai yang kosong (*0 missing values*) pada seluruh kolom metrik, dan tidak ada baris data yang terekam ganda (*no duplicated rows*). Oleh karena itu, data ini sudah bersih secara struktural.
    """)
    
    with st.expander("👁️ Lihat 5 Baris Pertama Dataset Mentah"):
        st.dataframe(df.head())
        
    st.markdown("---")
    
    # ==========================================
    # 3. BAGIAN EDA (Exploratory Data Analysis)
    # ==========================================
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Distribusi & Outlier", 
        "Tren & Moving Average", 
        "Korelasi Heatmap", 
        "Distribusi Daily Return",
        "Volume vs Volatilitas",
        "Cumulative Return",
        "Analisis Likuiditas"
    ])
    
    with tab1:
        st.subheader("Distribusi Volume Perdagangan (Analisis Univariat)")
        st.markdown("Grafik histogram ini memperlihatkan seberapa sering rentang volume tertentu terjadi. Kurva KDE membantu melihat pola distribusi data.")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['Volume'], bins=50, kde=True, color='teal', ax=ax)
        ax.set_title('Distribusi Volume Perdagangan Saham Teknologi', fontweight='bold')
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Analisis Outlier (Volume Berdasarkan Ticker)")
        st.markdown("""
        * **NVDA Paling Dominan & Volatil:** NVDA memiliki fluktuasi volume dan lonjakan transaksi paling ekstrem. Ini menunjukkan tingginya likuiditas dan spekulasi pasar.
        * **Lonjakan Volume Cukup Sering:** Distribusi data tidak normal. Hari-hari di mana terjadi lonjakan volume transaksi saham yang tinggi cukup lumrah terjadi.
        """)
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='Ticker', y='Volume', palette='Set2', ax=ax4)
        ax4.set_title('Deteksi Outlier pada Volume Perdagangan Saham', fontweight='bold')
        st.pyplot(fig4)
        
    with tab2:
        st.subheader("Tren Harga Saham dari Waktu ke Waktu (Analisis Multivariat)")
        st.markdown("Grafik Time-Series ini memetakan lintasan harga saham. Data historis pergerakan harga (Close price) yang bersih seperti ini adalah pondasi utama untuk prediksi.")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df, x='Date', y='Close', hue='Ticker', linewidth=1.5, ax=ax2)
        ax2.set_title('Tren Harga Penutupan Perusahaan Terpilih', fontweight='bold')
        st.pyplot(fig2)

        st.markdown("---")
        st.subheader("Analisis Moving Average (MA) Jangka Panjang")
        st.markdown("Garis biru (MA 50) dan merah (MA 200) jauh lebih mulus. Jika garis biru memotong garis merah ke arah atas (Golden Cross), itu adalah sinyal tren pasar Bullish.")
        
        ticker_ma = st.selectbox("Pilih Ticker untuk Analisis MA:", df['Ticker'].unique(), index=0)
        df_ma = df[df['Ticker'] == ticker_ma].copy()
        df_ma['MA_50'] = df_ma['Close'].rolling(window=50).mean()
        df_ma['MA_200'] = df_ma['Close'].rolling(window=200).mean()
        df_ma_recent = df_ma[df_ma['Date'] >= '2020-01-01']
        
        fig_ma, ax_ma = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_ma_recent, x='Date', y='Close', label='Harga Asli', alpha=0.5, ax=ax_ma)
        sns.lineplot(data=df_ma_recent, x='Date', y='MA_50', label='MA 50 Hari', color='blue', ax=ax_ma)
        sns.lineplot(data=df_ma_recent, x='Date', y='MA_200', label='MA 200 Hari', color='red', ax=ax_ma)
        ax_ma.set_title(f'Moving Average 50 dan 200 Hari untuk {ticker_ma}', fontweight='bold')
        st.pyplot(fig_ma)
        
    with tab3:
        st.subheader("Heatmap Korelasi Variabel Numerik Dasar")
        kolom_numerik = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.heatmap(kolom_numerik.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
        ax3.set_title("Heatmap Korelasi Variabel Saham")
        st.pyplot(fig3)
        
        st.markdown("---")
        st.subheader("Matriks Korelasi Pergerakan Harga (Return Correlation Heatmap)")
        st.markdown("""
        Matriks ini membandingkan 10 emiten. Warna merah berarti korelasinya positif (bergerak searah).
        * **Temuan Menarik:** GOOG (Google) dan GOOGL (Alphabet) memiliki korelasi 0.99. Ini wajar karena mereka dasarnya entitas yang sama.
        * **Temuan Industri:** MSFT dan AAPL punya korelasi tinggi. Ini menunjukkan sentimen pasar terhadap sektor teknologi secara umum itu seragam.
        """)
        
        daily_return_pivot = df.pivot_table(index='Date', columns='Ticker', values='Daily_Return')
        corr_matrix = daily_return_pivot.corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
        ax_corr.set_title('Matriks Korelasi Pergerakan Harga (Daily Return)\n', fontsize=14, fontweight='bold')
        ax_corr.set_xlabel('')
        ax_corr.set_ylabel('')
        st.pyplot(fig_corr)

    with tab4:
        st.subheader("Distribusi Daily Return (Volatilitas Saham)")
        st.markdown("Insight: Grafik Violin Plot akan menunjukkan bentuk 'perut'. Jika perutnya gemuk dan pendek, berarti saham itu stabil. Jika memanjang ke atas dan ke bawah (seperti NVDA dan META), berarti saham berisiko tinggi.")
        df_return = df.dropna(subset=['Daily_Return'])
        
        fig_vio, ax_vio = plt.subplots(figsize=(14, 7))
        sns.violinplot(data=df_return, x='Ticker', y='Daily_Return', hue='Ticker', palette='Set3', inner='quartile', legend=False, ax=ax_vio)
        ax_vio.set_title('Distribusi Daily Return (Volatilitas Risiko)', fontweight='bold')
        st.pyplot(fig_vio)

    with tab5:
        st.subheader("Scatter Plot: Apakah Volume Transaksi Mempengaruhi Volatilitas Harga?")
        st.markdown("Ada tren yang cukup terlihat: semakin ke kanan (volume transaksi membesar), titik-titiknya cenderung menyebar lebih ke atas (volatilitas membesar). Artinya, pada hari-hari kepanikan/antusiasme pasar (volume tinggi), jarak harga tertinggi dan terendah makin lebar.")
        df_sample = df.sample(n=5000, random_state=42)
        
        fig_scat, ax_scat = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_sample, x='Volume', y='Volatility', hue='Ticker', alpha=0.6, palette='tab10', ax=ax_scat)
        ax_scat.set_xscale('log')
        ax_scat.set_title('Hubungan Volume Perdagangan dan Volatilitas Harga Harian', fontweight='bold')
        st.pyplot(fig_scat)

    with tab6:
        st.subheader("Cumulative Return (Siapa Pemenang Investasi Jangka Panjang?)")
        st.markdown("Temuan Epik: Perhatikan garis (seperti LLY atau AMZN). Di akhir grafik (tahun 2024), garisnya mencapai persentase ekstrem, membuktikan bahwa uang yang ditanamkan pada emiten tersebut tumbuh berlipat-lipat ganda.")
        
        avg_close = df.groupby('Ticker')['Close'].mean().sort_values(ascending=False)
        top_5_tickers = avg_close.head(5).index.tolist()
        df_top5 = df[df['Ticker'].isin(top_5_tickers)].copy()
        df_top5['Cumulative_Return'] = df_top5.groupby('Ticker')['Daily_Return'].transform(lambda x: (1 + x).cumprod() - 1)
        
        fig_cum, ax_cum = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_top5, x='Date', y='Cumulative_Return', hue='Ticker', ax=ax_cum)
        ax_cum.set_title('Cumulative Return Top 5 Tech Companies', fontweight='bold')
        ax_cum.yaxis.set_major_formatter(PercentFormatter(1))
        st.pyplot(fig_cum)

    with tab7:
        st.subheader("Proporsi Data dan Analisis Likuiditas")
        st.markdown("Grafik Donut Chart di bawah memvisualisasikan proporsi likuiditas pasar berdasarkan agregat total volume transaksi saham dari 10 perusahaan teknologi raksasa. Menyoroti ketimpangan distribusi transaksi (konsentrasi pasar).")
        
        liquidity_dist = df.groupby('Ticker')['Volume'].sum().sort_values(ascending=False)
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        ax_pie.pie(liquidity_dist, labels=liquidity_dist.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85, colors=sns.color_palette('Set2', len(liquidity_dist)))
        
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig_pie.gca().add_artist(centre_circle)
        ax_pie.set_title("Proporsi Likuiditas (Total Volume) per Emiten", fontweight='bold')
        st.pyplot(fig_pie)

    st.markdown("---")
    
    # ==========================================
    # 4. PREPROCESSING (Sesuai Colab untuk ML)
    # ==========================================
    # Melakukan preprocessing untuk menyamakan akurasi ML
    df_ml = df.copy()
    
    # Drop Missing Values akibat pct_change()
    df_ml = df_ml.dropna()
    
    # Label Encoding untuk Ticker
    le = LabelEncoder()
    df_ml['Ticker'] = le.fit_transform(df_ml['Ticker'])
    
    # Normalisasi (StandardScaler) pada kolom numerik 
    numerik = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    scaler = StandardScaler()
    df_ml[numerik] = scaler.fit_transform(df_ml[numerik])

    # ==========================================
    # 5. MACHINE LEARNING
    # ==========================================
    st.header("🤖 Model Machine Learning (Random Forest)")
    
    tab_clf, tab_reg = st.tabs(["Klasifikasi (Arah Harga)", "Regresi (Prediksi Harga)"])
    
    with tab_clf:
        st.subheader("Evaluasi Performa Model Klasifikasi Kinerja Saham")
        st.markdown("Algoritma **Random Forest Classifier** digunakan untuk memprediksi arah pergerakan kinerja saham perusahaan teknologi, yakni apakah akan menunjukkan tren positif (naik) atau negatif (turun).")
        
        # Persiapan Data Klasifikasi
        df_clf = df_ml.copy()
        df_clf['Target'] = (df_clf['Close'] > df_clf['Open']).astype(int)
        
        # X membuang Close, Date, dan Target
        X_clf = df_clf.drop(columns=['Close', 'Date', 'Target'])
        y_clf = df_clf['Target']
        
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
        
        rf_clf = RandomForestClassifier(random_state=42)
        rf_clf.fit(X_train_c, y_train_c)
        y_pred_c = rf_clf.predict(X_test_c)
        acc = accuracy_score(y_test_c, y_pred_c)
        
        st.success(f"**Tingkat Akurasi Model: {acc:.2%}** (Sesuai dengan laporan proyek ~82.86%)")
        st.markdown("Angka ini menunjukkan bahwa dari keseluruhan data yang diuji, model mampu memprediksi arah kinerja saham dengan tingkat kebenaran tersebut.")
        
        st.markdown("### Analisis Faktor Penentu Kinerja Saham (Feature Importance)")
        st.markdown("Indikator dengan batang terpanjang merupakan fitur yang paling berkontribusi dalam pengambilan keputusan algoritma. Insight ini sangat berguna secara analisis teknikal.")
        
        df_importance = pd.DataFrame({
            'Fitur': X_clf.columns,
            'Kepentingan': rf_clf.feature_importances_
        }).sort_values(by='Kepentingan', ascending=False)

        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_importance, x='Kepentingan', y='Fitur', palette='viridis', ax=ax_imp)
        ax_imp.set_title("Feature Importance - Klasifikasi")
        st.pyplot(fig_imp)
        
    with tab_reg:
        st.subheader("Visualisasi Prediksi Harga Saham (Time Series Forecasting)")
        st.markdown("Grafik di bawah merupakan hasil pengujian model **Random Forest Regressor** dalam memprediksi angka pasti harga penutupan saham (*Close*) untuk keesokan harinya (H+1). Dapat dilihat bahwa model mampu mengikuti pola fluktuasi (naik-turun) dari harga aslinya dengan sangat baik dan jarak (error) yang sangat minim.")
        
        # Persiapan Data Regresi (Prediksi H+1)
        df_reg = df_ml.copy()
        df_reg['Close_H+1'] = df_reg.groupby('Ticker')['Close'].shift(-1)
        df_reg = df_reg.dropna(subset=['Close_H+1'])
        
        X_reg = df_reg.drop(columns=['Close_H+1', 'Date'])
        y_reg = df_reg['Close_H+1']
        
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        
        rf_reg = RandomForestRegressor(random_state=42)
        rf_reg.fit(X_train_r, y_train_r)
        y_pred_r = rf_reg.predict(X_test_r)
        
        # Visualisasi Aktual vs Prediksi
        fig_reg, ax_reg = plt.subplots(figsize=(12, 6))
        ax_reg.plot(y_test_r.values[:80], label='Harga Aktual (Biru)', marker='o', color='blue', alpha=0.7)
        ax_reg.plot(y_pred_r[:80], label='Prediksi Model (Oranye)', marker='x', linestyle='--', color='orange')
        ax_reg.legend()
        ax_reg.set_title("Perbandingan Harga Aktual vs Prediksi Model (H+1) - 80 Data Uji Pertama")
        st.pyplot(fig_reg)

else:
    st.info("👈 Silakan upload file dataset CSV Anda di panel sebelah kiri untuk memuat dashboard analisis.")