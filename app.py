import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

MODEL_PATH = "trained_arima.pkl"

# 📌 Fungsi untuk membaca data CSV
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("stok_gudang_3_Tahun.csv", encoding="utf-8-sig", sep=";")
        df.columns = df.columns.str.replace("ï»¿", "")  # Hapus karakter BOM
        
        if "Tanggal" not in df.columns or "Produk" not in df.columns or "Jumlah Terjual" not in df.columns:
            st.error("❌ File CSV harus memiliki kolom: 'Tanggal', 'Produk', dan 'Jumlah Terjual'.")
            return None
        
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%Y-%m", errors="coerce")
        df = df.dropna(subset=["Tanggal"])
        df = df.set_index("Tanggal")
        
        return df
    except Exception as e:
        st.error(f"❌ Gagal membaca file CSV: {e}")
        return None

# 📌 Fungsi untuk mengecek stasioneritas data
def check_stationarity(series):
    result = adfuller(series.dropna())
    st.write(f"ADF Statistic: {result[0]}")
    st.write(f"p-value: {result[1]}")
    
    if result[1] > 0.05:
        st.warning("⚠️ Data tidak stasioner, mungkin perlu diferensiasi (d > 0)")
    else:
        st.success("✅ Data sudah stasioner!")

# 📌 Fungsi untuk mencari parameter ARIMA terbaik secara otomatis
def find_best_arima(df_produk):
    try:
        model_auto = auto_arima(df_produk["Jumlah Terjual"], seasonal=True, m=12, stepwise=True, trace=True)
        return model_auto.order  # Return (p, d, q)
    except Exception as e:
        st.error(f"❌ Gagal mencari parameter terbaik: {e}")
        return None

# 📌 Fungsi untuk melatih model ARIMA
def train_arima(df, produk, p=None, d=None, q=None):
    try:
        df_produk = df[df["Produk"] == produk][["Jumlah Terjual"]].copy()
        
        if df_produk.empty:
            st.warning(f"⚠️ Tidak ada data untuk produk '{produk}'.")
            return None, None

        df_produk = df_produk.resample('M').sum().asfreq('M').fillna(0)
        
        # Uji stasioneritas
        check_stationarity(df_produk["Jumlah Terjual"])

        # Transformasi log untuk stabilisasi variansi
        df_produk["Jumlah Terjual"] = np.log1p(df_produk["Jumlah Terjual"])
        
        train_size = int(len(df_produk) * 0.8)
        train_data, test_data = df_produk.iloc[:train_size], df_produk.iloc[train_size:]

        # Jika tidak ada parameter, cari otomatis
        if p is None or d is None or q is None:
            p, d, q = find_best_arima(train_data)

        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        
        if len(test_data) > 0:
            predictions = model_fit.forecast(steps=len(test_data))
            mape = mean_absolute_percentage_error(np.expm1(test_data), np.expm1(predictions))
        else:
            mape = None

        # Simpan model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_fit, f)

        return model_fit, mape
    except Exception as e:
        st.error(f"❌ Gagal melatih model ARIMA: {e}")
        return None, None

# 📌 Fungsi untuk memuat model ARIMA yang telah dilatih
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

# 📌 Fungsi untuk melakukan prediksi
def forecast_stock(model_fit, df, bulan_ke_depan=6):
    try:
        last_date = df.index.max()
        forecast = model_fit.forecast(steps=bulan_ke_depan)
        forecast = np.expm1(forecast)  # Kembalikan ke skala asli
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=bulan_ke_depan, freq="M")
        return pd.DataFrame({"Tanggal": forecast_dates, "Prediksi Jumlah Terjual": forecast.values})
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi ARIMA: {e}")
        return None
    

# 📌 Fungsi untuk menampilkan grafik
def plot_forecast(df, forecast_df):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Jumlah Terjual"], label="Data Aktual", marker='o', linestyle='-')
    plt.plot(forecast_df["Tanggal"], forecast_df["Prediksi Jumlah Terjual"], label="Prediksi", marker='o', linestyle='--', color='red')
    plt.xlabel("Tanggal")
    plt.ylabel("Jumlah Terjual")
    plt.title("Prediksi Permintaan Stok Barang")
    plt.legend()
    st.pyplot(plt)

# === STREAMLIT UI ===
st.title("📊 Prediksi Permintaan Stok Barang")

df = load_data()

if df is not None:
    produk_list = df["Produk"].unique()
    produk = st.selectbox("Pilih Produk:", produk_list)

    st.sidebar.header("⚙️ Pengaturan Model ARIMA")
    
    auto_arima_choice = st.sidebar.checkbox("Gunakan AutoARIMA (Rekomendasi)", value=True)

    if auto_arima_choice:
        p, d, q = None, None, None
    else:
        p = st.sidebar.slider("p (Autoregressive)", 0, 5, 2)
        d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
        q = st.sidebar.slider("q (Moving Average)", 0, 5, 2)

    if st.sidebar.button("🔄 Latih Model ARIMA"):
        with st.spinner("⏳ Melatih model..."):
            model_fit, mape = train_arima(df, produk, p, d, q)
            if model_fit:
                st.success("✅ Model berhasil dilatih!")
                if mape is not None:
                    st.sidebar.write(f"📌 **MAPE:** {mape:.2f}%")
                else:
                    st.sidebar.warning("⚠️ MAPE tidak dapat dihitung.")

    bulan_ke_depan = st.sidebar.slider("Prediksi untuk berapa bulan ke depan?", 1, 12, 6)

    if st.sidebar.button("📈 Buat Prediksi"):
        trained_model = load_trained_model()
        if trained_model:
            forecast_df = forecast_stock(trained_model, df, bulan_ke_depan)
            if forecast_df is not None:
                st.dataframe(forecast_df)
                plot_forecast(df[df["Produk"] == produk], forecast_df)
        else:
            st.warning("⚠️ Model belum dilatih. Silakan latih model terlebih dahulu.")
