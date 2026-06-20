import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
from datetime import datetime, timedelta
import pandas as pd
import random
import plotly.express as px  # Library untuk grafik berwarna-warni profesional

# ===================== 1. KONFIGURASI HALAMAN =====================
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS untuk Background dan Estetika Profesional
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f3fdf6 0%, #e8f5e9 100%);
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.7); 
        padding: 2rem;
        border-radius: 20px;
        margin: 10px;
    }
    h1 {
        color: #1b5e20 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .result-card {
        background: white !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .recommendation-box {
        background: #ffffff !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ===================== 2. DATA REFERENSI PENGOLAHAN =====================
waste_guide = {
    'organik': {'buang': 'Tempat Sampah Hijau', 'daur_ulang': 'Bisa (Secara Biologis)', 'rekomendasi': 'Dijadikan pupuk kompos, pakan ternak/maggot, atau dibuat Eco-Enzyme.', 'faktor_pemulihan': 0.95, 'warna': '#2e7d32'},
    'plastik': {'buang': 'Tempat Sampah Kuning', 'daur_ulang': 'Bisa (Tinggi)', 'rekomendasi': 'Cuci bersih, keringkan, lalu pilah berdasarkan jenis untuk disetor ke Bank Sampah.', 'faktor_pemulihan': 0.80, 'warna': '#fbc02d'},
    'kertas': {'buang': 'Tempat Sampah Kuning', 'daur_ulang': 'Bisa (Tinggi)', 'rekomendasi': 'Hindari terkena air/minyak. Kumpulkan dan ikat rapi untuk diproses menjadi bubur kertas kembali.', 'faktor_pemulihan': 0.85, 'warna': '#ff9800'},
    'logam': {'buang': 'Tempat Sampah Kuning', 'daur_ulang': 'Bisa (Sangat Tinggi)', 'rekomendasi': 'Kaleng bekas makanan/minuman dibersihkan. Memiliki nilai ekonomi tinggi.', 'faktor_pemulihan': 0.90, 'warna': '#78909c'},
    'kaca': {'buang': 'Tempat Sampah Kuning', 'daur_ulang': 'Bisa', 'rekomendasi': 'Pastikan tidak pecah. Bisa dilebur ulang tanpa batas.', 'faktor_pemulihan': 0.75, 'warna': '#0288d1'},
    'unorganik': {'buang': 'Tempat Sampah Merah/Abu', 'daur_ulang': 'Sulit', 'rekomendasi': 'Sampah saset makanan (kemasan berlapis), stirofoam, langsung dibuang ke TPA.', 'faktor_pemulihan': 0.10, 'warna': '#d32f2f'}
}

color_map = {k: v['warna'] for k, v in waste_guide.items()}

# ===================== 3. FUNGSI UTILITAS & DATA CADANGAN =====================
def hitung_efisiensi_daur_ulang(faktor_pemulihan, confidence_ai):
    return confidence_ai * faktor_pemulihan

def generate_dummy_data():
    """Fungsi pengaman untuk menghasilkan data simulasi 2 tahun jika Firebase kosong"""
    kategori_list = ['kaca', 'kertas', 'logam', 'organik', 'plastik', 'unorganik']
    dummy = []
    sekarang = datetime.now()
    for _ in range(50):
        hari_lalu = random.randint(0, 700)
        waktu_log = sekarang - timedelta(days=hari_lalu, hours=random.randint(0,23))
        kat = random.choice(kategori_list)
        conf = random.uniform(65.0, 99.0)
        f_pemulihan = waste_guide[kat]['faktor_pemulihan']
        efisiensi = conf * f_pemulihan
        durasi = random.uniform(0.25, 0.65)
        dummy.append({
            'waktu_klasifikasi': waktu_log.strftime("%Y-%m-%d %H:%M:%S"),
            'jenis_sampah': kat,
            'waktu_analisa': f"{durasi:.3f} detik",
            'waktu_analisa_num': durasi,
            'daur_ulang': 'Bisa' if kat != 'unorganik' else 'Sulit',
            'efisiensi_daur_ulang': f"{efisiensi:.2f}%",
            'efisiensi_num': efisiensi,
            'rekomendasi': waste_guide[kat]['rekomendasi']
        })
    return dummy

import firebase_admin
from firebase_admin import credentials, firestore

# ===================== 4. KONEKSI DATABASE FIREBASE CLOUD =====================
@st.cache_resource
def inisialisasi_firebase():
    """Fungsi untuk mengoneksikan Streamlit ke Firebase secara aman (cached)"""
    if not firebase_admin._apps:
        # GANTI DENGAN ALAMAT JALUR LENGKAP LOKASI FILE DI LAPTOP AHMAD
        cred = credentials.Certificate(r'D:\SmartWasteProject\kunci-firebase.json.json')
        firebase_admin.initialize_app(cred)
    return firestore.client()

try:
    db = inisialisasi_firebase()
except Exception as e:
    st.error(f"❌ Gagal terhubung ke database Firebase Cloud: {e}")
    db = None

def ambil_data_dari_firebase():
    if db is None:
        return []
    try:
        docs = db.collection('log_sampah').stream()
        data_list = []
        for doc in docs:
            d = doc.to_dict()
            jenis = str(d.get('jenis_sampah', 'unorganik')).lower()
            waktu_analisa_raw = d.get('waktu_analisa', 0.5) 
            efisiensi_raw = d.get('efisiensi_daur_ulang', 0.0) 
            
            data_list.append({
                'waktu_klasifikasi': d.get('waktu_klasifikasi', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'jenis_sampah': jenis,
                'waktu_analisa': f"{waktu_analisa_raw:.3f} detik",
                'waktu_analisa_num': float(waktu_analisa_raw), 
                'daur_ulang': 'Bisa' if jenis != 'unorganik' else 'Sulit',
                'efisiensi_daur_ulang': f"{efisiensi_raw:.2f}%",
                'efisiensi_num': float(efisiensi_raw), 
                'rekomendasi': d.get('rekomendasi', '-')
            })
        return data_list
    except Exception as e:
        st.error(f"⚠️ Gagal membaca data dari Firestore: {e}")
        return []

# Jalankan penarikan data dari cloud database
data_terbaru = ambil_data_dari_firebase()

if not data_terbaru:
    data_terbaru = generate_dummy_data()

st.session_state.history = data_terbaru

# ===================== 5. LOAD MODEL AI =====================
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model_sampah_ahmad.h5')

try:
    model = load_my_model()
except:
    model = None

class_names = ['kaca', 'kertas', 'logam', 'organik', 'plastik', 'unorganik']

if 'history' not in st.session_state or len(st.session_state.history) == 0:
    st.session_state.history = generate_dummy_data()

# ===================== 6. SIDEBAR MENU =====================
with st.sidebar:
    st.header("📋 Menu Utama")
    page = st.radio("Navigasi:", ["Klasifikasi Sampah", "Dashboard Analisis", "Rekomendasi Pengolahan", "Riwayat Klasifikasi", "Tentang Aplikasi"])
    st.markdown("---")
    st.info(f"**Ahmad Nurul Fajri**\n14012100086\nUNIBA - Konsentrasi AI")

# ===================== 7. LOGIKA HALAMAN =====================

# --- HALAMAN 1: KLASIFIKASI SAMPAH ---
if page == "Klasifikasi Sampah":
    st.title("♻️ Smart Waste Classifier")
    st.markdown("### Pilih Metode Pengambilan Gambar")
    
    input_mode = st.radio("Metode Input:", ["Unggah File (Galeri)", "Ambil Foto (Kamera Live)"], index=0)
    image_data = None
    
    if input_mode == "Ambil Foto (Kamera Live)":
        image_data = st.camera_input("Arahkan kamera ke objek sampah")
    else:
        image_data = st.file_uploader("Pilih gambar dari perangkat...", type=["jpg", "jpeg", "png"])

    if image_data is not None:
        img = Image.open(image_data)
        if input_mode == "Unggah File (Galeri)":
            st.image(img, caption="Gambar yang dipilih", use_container_width=True)
        
        if st.button("Mulai Analisis AI"):
            if model is None:
                st.error("File 'model_sampah_ahmad.h5' tidak ditemukan di direktori proyek!")
            else:
                with st.spinner("🤖 AI sedang menganalisis objek..."):
                    img_resized = img.resize((224, 224))
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    
                    start_time = time.time()
                    prediction = model.predict(img_array, verbose=0)
                    end_time = time.time()
                    
                    duration = end_time - start_time
                    predicted_idx = np.argmax(prediction[0])
                    predicted_class = class_names[predicted_idx].lower()
                    confidence = float(prediction[0][predicted_idx] * 100)
                    
                    info = waste_guide.get(predicted_class)
                    skor_efisiensi = hitung_efisiensi_daur_ulang(info['faktor_pemulihan'], confidence)
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h2 style='text-align: center; color: #1b5e20; margin-bottom: 5px;'>Hasil: {predicted_class}</h2>
                        <p style='text-align: center; color: #666; font-size: 1.1em;'>Tingkat Keyakinan AI: <b>{confidence:.2f}%</b></p>
                        <hr style='border: 0.5px solid #e0e0e0; margin: 10px 0;'>
                        <p style='text-align: center; color: #2e7d32; font-size: 0.9em;'>
                            ⏱️ Waktu Analisa Mesin AI: <b>{duration:.3f} detik</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <p style='margin-bottom: 5px;'>📍 <b>Tempat Buang:</b> {info['buang']}</p>
                        <p style='margin-bottom: 5px;'>♻️ <b>Daur Ulang:</b> {info['daur_ulang']}</p>
                        <p style='margin-bottom: 5px;'>📈 <b>Efisiensi Teknis Daur Ulang:</b> {skor_efisiensi:.2f}%</p>
                        <p style='margin-bottom: 0;'>📝 <b>Saran Pengolahan:</b> {info['rekomendasi']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # [KODE LAMA AHMAD] Menyimpan ke log riwayat lokal browser
                    st.session_state.history.insert(0, {
                        'waktu_klasifikasi': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'jenis_sampah': predicted_class,
                        'waktu_analisa': f"{duration:.3f} detik",
                        'waktu_analisa_num': duration,
                        'daur_ulang': info['daur_ulang'],
                        'efisiensi_daur_ulang': f"{skor_efisiensi:.2f}%",
                        'efisiensi_num': skor_efisiensi,
                        'rekomendasi': info['rekomendasi']
                    })

                    # 🚀 [KODE BARU] KIRIM LANGSUNG DATA WEB INI KE FIREBASE CLOUD FIRESTORE
                    if db is not None:
                        try:
                            db.collection('log_sampah').add({
                                'waktu_klasifikasi': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'jenis_sampah': predicted_class,
                                'waktu_analisa': float(duration),  # disimpan dalam bentuk angka sesuai spek android
                                'efisiensi_daur_ulang': float(skor_efisiensi), # disimpan dalam bentuk angka
                                'rekomendasi': info['rekomendasi']
                            })
                            st.success("☁️ Data Analisis Web Berhasil Disinkronkan ke Firebase Cloud!")
                        except Exception as firebase_error:
                            st.warning(f"⚠️ Gagal mengirim data ke cloud: {firebase_error}")

                    st.balloons()

# --- HALAMAN 2: DASHBOARD SMART WASTE (DESAIN ASLI DATA WARNA & FITUR MATA) ---
elif page == "Dashboard Analisis":
    st.title("📊 Dashboard Analisis Cerdas & Tren Ragam Waktu")
    st.write("Visualisasi log data menggunakan perhitungan Indeks Efisiensi Daur Ulang Teknis hulu ke hilir.")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        df['waktu_klasifikasi'] = pd.to_datetime(df['waktu_klasifikasi'])
        df['jenis_sampah'] = df['jenis_sampah'].str.lower()
        
        st.subheader("⏱️ Filter Analisis Periode Waktu")
        pilihan_waktu = st.selectbox(
            "Pilih Jangkauan Data Real-Time:",
            ["Semua Data (Total Rentang 2 Tahun)", "1 Minggu Terakhir", "1 Bulan Terakhir", "1 Tahun Terakhir"]
        )
        
        sekarang = datetime.now()
        if pilihan_waktu == "1 Minggu Terakhir":
            df_filtered = df[df['waktu_klasifikasi'] >= (sekarang - timedelta(days=7))]
        elif pilihan_waktu == "1 Bulan Terakhir":
            df_filtered = df[df['waktu_klasifikasi'] >= (sekarang - timedelta(days=30))]
        elif pilihan_waktu == "1 Tahun Terakhir":
            df_filtered = df[df['waktu_klasifikasi'] >= (sekarang - timedelta(days=365))]
        else:
            df_filtered = df.copy()
            
        if not df_filtered.empty:
            avg_duration = df_filtered['waktu_analisa_num'].mean()
            avg_duration_str = f"{avg_duration:.3f} detik"
            avg_efficiency = df_filtered['efisiensi_num'].mean()
            avg_efficiency_str = f"{avg_efficiency:.2f}%"
        else:
            avg_duration_str = "N/A"
            avg_efficiency_str = "N/A"
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.04); border-top: 4px solid #2e7d32; text-align: center;">
                <p style="margin: 0; font-size: 0.9em; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Total Sampah Terdeteksi</p>
                <h2 style="margin: 8px 0 0 0; color: #1b5e20; font-size: 2.2em; font-weight: 700;">{len(df_filtered)}</h2>
                <p style="margin: 3px 0 0 0; font-size: 0.75em; color: #999;">Kuantitas Log Sistem</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            kategori_dominan = df_filtered['jenis_sampah'].mode()[0] if not df_filtered.empty else "n/a"
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.04); border-top: 4px solid #fbc02d; text-align: center;">
                <p style="margin: 0; font-size: 0.9em; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Kategori Dominan</p>
                <h2 style="margin: 8px 0 0 0; color: #f57f17; font-size: 2.2em; font-weight: 700;">{kategori_dominan.lower()}</h2>
                <p style="margin: 3px 0 0 0; font-size: 0.75em; color: #999;">Volume Tertinggi</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.04); border-top: 4px solid #0288d1; text-align: center;">
                <p style="margin: 0; font-size: 0.9em; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Rata rata Waktu Analisa AI</p>
                <h2 style="margin: 8px 0 0 0; color: #01579b; font-size: 2.2em; font-weight: 700;">{avg_duration_str}</h2>
                <p style="margin: 3px 0 0 0; font-size: 0.75em; color: #999;">Komputasi Keras h5</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div style="background: white; padding: 18px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.04); border-top: 4px solid #e91e63; text-align: center;">
                <p style="margin: 0; font-size: 0.9em; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Rata rata Efisiensi Daur Ulang</p>
                <h2 style="margin: 8px 0 0 0; color: #880e4f; font-size: 2.2em; font-weight: 700;">{avg_efficiency_str}</h2>
                <p style="margin: 3px 0 0 0; font-size: 0.75em; color: #999;">Indeks Kualitas Teknis</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📈 Visualisasi & Analisis Tren Interaktif (Multi-Filter)")
        
        def tentukan_warna_drop(jenis):
            if jenis == 'organik':
                return 'Tempat Sampah Hijau'
            elif jenis in ['plastik', 'kertas', 'logam', 'kaca']:
                return 'Tempat Sampah Kuning'
            else:
                return 'Tempat Sampah Merah/Abu (Residu)'

        df_filtered['warna_tempat_sampah'] = df_filtered['jenis_sampah'].apply(tentukan_warna_drop)
        
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            kategori_tersedia = ["Semua Kategori"] + [c.lower() for c in class_names]
            kategori_terpilih = st.selectbox(
                "🔍 Filter 1: Pilih Kategori Sampah:",
                kategori_tersedia,
                index=0
            )
            
        with filter_col2:
            wadah_tersedia = ["Semua Tempat Sampah", "Tempat Sampah Hijau", "Tempat Sampah Kuning", "Tempat Sampah Merah/Abu (Residu)"]
            wadah_terpilih = st.selectbox(
                "🗑️ Filter 2: Pilih Warna Tempat Sampah:",
                wadah_tersedia,
                index=0
            )
            
        df_graph = df_filtered.copy()
        
        if kategori_terpilih != "Semua Kategori":
            df_graph = df_graph[df_graph['jenis_sampah'] == kategori_terpilih]
            
        if wadah_terpilih != "Semua Tempat Sampah":
            df_graph = df_graph[df_graph['warna_tempat_sampah'] == wadah_terpilih]
            
        pembagi_kolom_teks = f"({kategori_terpilih} | {wadah_terpilih})"
        show_isi_grafik = st.checkbox("👁️ Tampilkan Semua Tabel Hasil Data Grafik (Show All Data View)")
        
        color_map_bin = {
            'Tempat Sampah Hijau': '#2e7d32',
            'Tempat Sampah Kuning': '#fbc02d',
            'Tempat Sampah Merah/Abu (Residu)': '#d32f2f'
        }
        
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        # --- GRAFIK 1: VOLUME KATEGORI SAMPAH ---
        with row1_col1:
            st.markdown(f"**Volume Sampah Berdasarkan Kategori {pembagi_kolom_teks}**")
            if not df_graph.empty:
                df_counts = df_graph['jenis_sampah'].value_counts().reset_index()
                df_counts.columns = ['jenis_sampah', 'jumlah_log']
                
                fig_bar = px.bar(
                    df_counts, 
                    x='jenis_sampah', 
                    y='jumlah_log', 
                    color='jenis_sampah',
                    color_discrete_map=color_map,
                    labels={'jenis_sampah': 'Kategori', 'jumlah_log': 'Total Log'},
                    template="simple_white"
                )
                fig_bar.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)
                
                if show_isi_grafik:
                    st.markdown("📋 **Hasil Angka Volume Kategori (Semua Data):**")
                    st.dataframe(df_counts, use_container_width=True, hide_index=True, height=180)
            else:
                st.info("Tidak ada data yang cocok dengan kombinasi filter.")
                
        # --- GRAFIK 2: VOLUME TEMPAT SAMPAH WADAH ---
        with row1_col2:
            st.markdown(f"**Volume Berdasarkan Warna Tempat Sampah {pembagi_kolom_teks}**")
            if not df_graph.empty:
                df_bin_counts = df_graph['warna_tempat_sampah'].value_counts().reset_index()
                df_bin_counts.columns = ['warna_tempat_sampah', 'jumlah_log']
                
                fig_bin = px.bar(
                    df_bin_counts,
                    x='warna_tempat_sampah',
                    y='jumlah_log',
                    color='warna_tempat_sampah',
                    color_discrete_map=color_map_bin,
                    labels={'warna_tempat_sampah': 'Warna Tempat Sampah', 'jumlah_log': 'Total Log'},
                    template="simple_white"
                )
                fig_bin.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_bin, use_container_width=True)
                
                if show_isi_grafik:
                    st.markdown("📋 **Hasil Angka Distribusi Wadah (Semua Data):**")
                    st.dataframe(df_bin_counts, use_container_width=True, hide_index=True, height=180)
            else:
                st.info("Tidak ada data yang cocok dengan kombinasi filter.")

        # --- GRAFIK 3: TREN EFISIENSI REAL-TIME ---
        with row2_col1:
            st.markdown(f"**Tren Efisiensi Real-Time {pembagi_kolom_teks}**")
            if not df_graph.empty:
                df_line = df_graph.sort_values('waktu_klasifikasi')
                
                fig_line = px.line(
                    df_line,
                    x='waktu_klasifikasi',
                    y='efisiensi_num',
                    color='jenis_sampah',
                    color_discrete_map=color_map,
                    labels={'waktu_klasifikasi': 'Tanggal', 'efisiensi_num': 'Efisiensi (%)'},
                    template='simple_white'
                )
                fig_line.update_traces(mode='lines+markers', line=dict(width=1.5), marker=dict(size=4))
                fig_line.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_line, use_container_width=True)
                
                if show_isi_grafik:
                    st.markdown("📋 **Hasil Angka Efisiensi (Semua Data):**")
                    df_line_all = df_line[['waktu_klasifikasi', 'jenis_sampah', 'efisiensi_daur_ulang']]
                    st.dataframe(df_line_all, use_container_width=True, hide_index=True, height=180)
            else:
                st.info("Tidak ada data yang cocok dengan kombinasi filter.")
                
        # --- GRAFIK 4: KECEPATAN ANALISA AI (DETIK) ---
        with row2_col2:
            st.markdown(f"**Kecepatan Analisa AI (Detik) {pembagi_kolom_teks}**")
            if not df_graph.empty:
                df_time = df_graph.sort_values('waktu_klasifikasi')
                
                fig_time = px.area(
                    df_time,
                    x='waktu_klasifikasi',
                    y='waktu_analisa_num',
                    color='jenis_sampah',
                    color_discrete_map=color_map,
                    labels={'waktu_klasifikasi': 'Tanggal', 'waktu_analisa_num': 'Durasi (Detik)'},
                    template='simple_white'
                )
                fig_time.update_layout(showlegend=True, legend_title_text='Jenis:', margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_time, use_container_width=True)
                
                if show_isi_grafik:
                    st.markdown("📋 **Hasil Angka Durasi Komputasi (Semua Data):**")
                    df_time_all = df_time[['waktu_klasifikasi', 'jenis_sampah', 'waktu_analisa']]
                    st.dataframe(df_time_all, use_container_width=True, hide_index=True, height=180)
            else:
                st.info("Tidak ada data yang cocok dengan kombinasi filter.")
        
        st.markdown("---")

        df_display = df_filtered.drop(columns=['efisiensi_num', 'waktu_analisa_num'], errors='ignore').copy()
        df_display['waktu_klasifikasi'] = df_display['waktu_klasifikasi'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        kolom_rapi = ['waktu_klasifikasi', 'jenis_sampah', 'waktu_analisa', 'daur_ulang', 'efisiensi_daur_ulang', 'rekomendasi']
        df_display = df_display.reindex(columns=kolom_rapi)
        
        st.subheader("📋 Log Riwayat Data Hasil Analisis Efisiensi Teknis")
        st.dataframe(df_display, use_container_width=True)
        
        if st.button("Reset / Hasilkan Ulang Data Acak 2 Tahun"):
            st.session_state.history = generate_dummy_data()
            st.rerun()
    else:
        st.info("Log dashboard kosong.")

# --- HALAMAN 3: REKOMENDASI PENGOLAHAN ---
elif page == "Rekomendasi Pengolahan":
    st.title("🗑️ Panduan & Rekomendasi Pengolahan")
    data_panduan = [
        {"Jenis": "organik", "Warna": "Hijau", "Daur Ulang": "Bisa (Kompos)", "Aksi": "Jadikan pupuk kompos, pakan ternak, atau Eco-Enzyme."},
        {"Jenis": "plastik", "Warna": "Kuning", "Daur Ulang": "Ya (Tinggi)", "Aksi": "Cuci bersih, keringkan, dan setor ke Bank Sampah."},
        {"Jenis": "kertas", "Warna": "Kuning", "Daur Ulang": "Ya (Tinggi)", "Aksi": "Kumpulkan, ikat rapi, pastikan tidak basah/berminyak."},
        {"Jenis": "logam", "Warna": "Kuning", "Daur Ulang": "Ya (Sangat Tinggi)", "Aksi": "Bersihkan sisa makanan, kumpulkan untuk pengepul/daur ulang."},
        {"Jenis": "kaca", "Warna": "Kuning", "Daur Ulang": "Ya", "Aksi": "Pisahkan dari sampah lain. Jika pecah, bungkus dengan aman."},
        {"Jenis": "unorganik", "Warna": "Merah/Abu", "Daur Ulang": "Sulit", "Aksi": "Bungkus rapat dan buang ke TPA sebagai residu."}
    ]
    st.table(data_panduan)

# --- HALAMAN 4: RIWAYAT KLASIFIKASI ---
elif page == "Riwayat Klasifikasi":
    st.title("📜 Riwayat Penggunaan Sesi")
    if st.session_state.history:
        for item in st.session_state.history[:15]: 
            st.info(f"**[{item['waktu_klasifikasi']}]** Kategori: **{item['jenis_sampah'].lower()}** | Efisiensi Daur Ulang: **{item['efisiensi_daur_ulang']}**")

# --- HALAMAN 5: TENTANG APLIKASI ---
elif page == "Tentang Aplikasi":
    st.title("ℹ️ Tentang Proyek")
    st.write("Aplikasi Smart Waste Classifier dikembangkan untuk mendukung gerakan Indonesia Bebas Sampah di lingkungan UNIBA.")