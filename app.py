import streamlit as st
import pandas as pd
import joblib  # Gunakan joblib jika model disimpan dengan joblib
import numpy as np

# Judul Aplikasi
st.title("Prediksi Income Berdasarkan Usia dan Pengalaman")

# Deskripsi
st.write("""
Aplikasi ini memprediksi penghasilan (Income) berdasarkan input usia (Age) dan pengalaman kerja (Experience) menggunakan model regresi.
""")

# Load model
try:
    model = joblib.load("regresi.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Input dari pengguna
age = st.number_input("Masukkan usia (Age):", min_value=0.0, step=1.0)
experience = st.number_input("Masukkan pengalaman kerja (Experience):", min_value=0.0, step=1.0)

# Tombol untuk prediksi
if st.button("Prediksi Income"):
    try:
        # Buat DataFrame dari input
        new_data_df = pd.DataFrame([[age, experience]], columns=['Age', 'Experience'])

        # Prediksi
        predicted_income = model.predict(new_data_df)

        # Tampilkan hasil prediksi
        st.success(f"Prediksi Income adalah: ${predicted_income.item():,.2f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
