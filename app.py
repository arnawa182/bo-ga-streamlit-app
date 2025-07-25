import streamlit as st
from main_pipeline import run_pipeline_from_streamlit
import os

st.set_page_config(page_title="Hybrid BO-GA Optimization", layout="centered")

st.title("🔍 Hybrid BO-GA Deep Learning Classifier")
st.markdown("""
Upload dataset Anda (format CSV, kolom harus memiliki `Class` untuk label) untuk menjalankan pipeline pelatihan model dengan optimasi Bayesian dan Genetic Algorithm.
""")

uploaded_file = st.file_uploader("📂 Upload dataset (contoh: creditcard.csv)", type=["csv"])

if uploaded_file:
    with open("uploaded_dataset.csv", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File berhasil diunggah.")
    
    if st.button("🚀 Jalankan Optimasi dan Pelatihan"):
        with st.spinner("Training model dan optimasi hyperparameter, mohon tunggu..."):
            txt_path, roc_img, pr_img = run_pipeline_from_streamlit("uploaded_dataset.csv")

        st.success("✅ Model selesai dilatih!")
        st.markdown("### 📄 Laporan Evaluasi")
        with open(txt_path, "r") as f:
            st.text(f.read())

        st.markdown("### 📊 Kurva ROC dan PR")
        st.image(roc_img, caption="ROC Curve")
        st.image(pr_img, caption="Precision-Recall Curve")

        with open(txt_path, "rb") as f:
            st.download_button("⬇️ Unduh Hasil Evaluasi", f, file_name=txt_path)
