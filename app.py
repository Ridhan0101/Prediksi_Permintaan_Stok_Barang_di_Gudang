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

warnings.filterwarnings("ignore")

# =================== Fungsi untuk Membaca Data CSV ===================
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig", sep=";")
        df.columns = df.columns.str.replace("ï»¿", "")  # Hapus karakter BOM
        
        if "Tanggal" not in df.columns or "Produk" not in df.columns or "Jumlah Terjual" not in df.columns:
            st.error("❌ File CSV harus memiliki kolom: 'Tanggal', 'Produk', dan 'Jumlah Terjual'.")
            return None
        
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%Y-%m", errors="coerce")
        df = df.dropna(subset=["Tanggal"])
        df = df.set_index("Tanggal")
        
        st.session_state["df"] = df  # Simpan data agar tidak hilang
        return df
    except Exception as e:
        st.error(f"❌ Gagal membaca file CSV: {e}")
        return None

# =================== Fungsi untuk Melatih Model ARIMA ===================
def train_arima(df, produk, auto_arima_choice=True, p=1, d=1, q=1):
    model_filename = f"trained_arima_{produk.replace(' ', '_')}.pkl"
    
    if os.path.exists(model_filename):
        with open(model_filename, "rb") as f:
            model_fit = pickle.load(f)
        st.success("✅ Model sudah ada dan langsung digunakan.")
        return model_fit, df[df["Produk"] == produk]
    
    df_produk = df[df["Produk"] == produk][["Jumlah Terjual"]].copy()
    df_produk = df_produk.resample('M').sum().asfreq('M').fillna(0)

    if auto_arima_choice:
        try:
            p, d, q = auto_arima(df_produk["Jumlah Terjual"], seasonal=True, m=12, stepwise=True).order
        except:
            p, d, q = (1, 1, 1)
    
    model = ARIMA(df_produk, order=(p, d, q))
    model_fit = model.fit()
    
    with open(model_filename, "wb") as f:
        pickle.dump(model_fit, f)
    
    st.success(f"✅ Model berhasil dilatih dengan parameter (p={p}, d={d}, q={q})")
    st.session_state[f"model_{produk}"] = model_fit
    return model_fit, df_produk

# =================== Fungsi untuk Memuat Model dari File ===================
def load_model_for_product(produk):
    model_filename = f"trained_arima_{produk.replace(' ', '_')}.pkl"
    if os.path.exists(model_filename):
        with open(model_filename, "rb") as f:
            return pickle.load(f)
    return None

# =================== Fungsi untuk Prediksi Stok Bulanan ===================
def forecast_stock(model_fit, df, bulan_ke_depan=12):
    try:
        last_date = df.index.max()
        forecast = model_fit.forecast(steps=bulan_ke_depan)
        forecast_months = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=bulan_ke_depan, freq="M")
        return pd.DataFrame({"Bulan": forecast_months.strftime('%Y-%m'), "Prediksi Jumlah Terjual": forecast.values})
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi ARIMA: {e}")
        return None

# =================== STREAMLIT UI ===================
st.set_page_config(page_title="Prediksi Stok Barang", layout="wide")

st.sidebar.title("🔍 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Upload & Latih Model", "Prediksi Stok"])

if page == "Upload & Latih Model":
    st.title("📂 Upload Data dan Latih Model")
    uploaded_file = st.file_uploader("Upload Data Training (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.success("✅ Data berhasil diunggah dan diproses!")
            st.dataframe(df.head())
            
            produk_list = df["Produk"].unique()
            produk = st.selectbox("🔹 Pilih Produk untuk Latihan Model:", produk_list)
            auto_arima_choice = st.checkbox("Gunakan Auto ARIMA", value=True)
            
            if not auto_arima_choice:
                p = st.number_input("Masukkan nilai p:", min_value=0, value=1)
                d = st.number_input("Masukkan nilai d:", min_value=0, value=1)
                q = st.number_input("Masukkan nilai q:", min_value=0, value=1)
            
            if st.button("🔄 Latih Model ARIMA"):
                with st.spinner("⏳ Melatih model ARIMA..."):
                    trained_model, df_produk = train_arima(df, produk, auto_arima_choice, p if not auto_arima_choice else 1, d if not auto_arima_choice else 1, q if not auto_arima_choice else 1)

elif page == "Prediksi Stok":
    st.title("📊 Prediksi Permintaan Stok Barang")
    df = st.session_state.get("df", None)
    if df is not None:
        produk_list = df["Produk"].unique()
        produk = st.selectbox("🔹 Pilih Produk untuk Prediksi:", produk_list)
        trained_model = load_model_for_product(produk)
        bulan_ke_depan = st.slider("📅 Prediksi untuk berapa bulan ke depan?", 1, 24, 12)
        
        if trained_model is None:
            st.warning("⚠️ Model belum dilatih untuk produk ini. Silakan latih model terlebih dahulu.")
        else:
            if st.button("📈 Lakukan Prediksi"):
                forecast_df = forecast_stock(trained_model, df[df["Produk"] == produk], bulan_ke_depan)
                if forecast_df is not None:
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df.index, df["Jumlah Terjual"], label="Data Aktual", marker='o', linestyle='-', color='blue')
                    ax.plot(pd.to_datetime(forecast_df["Bulan"]), forecast_df["Prediksi Jumlah Terjual"], label="Prediksi", marker='o', linestyle='--', color='red')
                    ax.set_xlabel("Bulan")
                    ax.set_ylabel("Jumlah Terjual")
                    ax.set_title(f"📈 Prediksi Permintaan Stok Barang: {produk}")
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig)
    else:
        st.warning("⚠️ Pastikan Anda telah mengunggah data terlebih dahulu.")
