import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
from datetime import datetime, timedelta
import pandas as pd
import random
import plotly.express as px
import firebase_admin
from firebase_admin import credentials, firestore
import os

# ===================== 1. KONFIGURASI HALAMAN =====================
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS untuk Tampilan Login & Aplikasi Utama
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
    .login-container {
        background-color: white;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        max-width: 450px;
        margin: 50px auto;
        border-top: 5px solid #2e7d32;
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
    .spy-box {
        background-color: #fff3cd !important;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
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

# ===================== 3. KONEKSI DATABASE FIREBASE CLOUD =====================
@st.cache_resource
def inisialisasi_firebase():
    if not firebase_admin._apps:
        # Prioritas utama membaca dari Advanced Secrets milik Streamlit Cloud Web
        if "firebase" in st.secrets:
            firebase_info = dict(st.secrets["firebase"])
            firebase_info["private_key"] = firebase_info["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(firebase_info)
        else:
            # Fallback lokal jika dijalankan di komputer sendiri
            nama_file_lokal = 'kunci-firebase.json'
            if os.path.exists(nama_file_lokal):
                cred = credentials.Certificate(nama_file_lokal)
            elif os.path.exists('kunci-firebase.json.json'):
                cred = credentials.Certificate('kunci-firebase.json.json')
            else:
                raise FileNotFoundError("Kredensial Firebase tidak ditemukan di st.secrets maupun file lokal.")
                
        firebase_admin.initialize_app(cred)
    return firestore.client()

try:
    db = inisialisasi_firebase()
except Exception as e:
    st.error(f"❌ Gagal terhubung ke database Firebase Cloud: {e}")
    db = None

# ===================== 4. UTILITIES (LOGIN, DATA, & MODEL LOAD) =====================
@st.cache_resource
def load_my_model():
    nama_model = 'model_sampah_ahmad.h5'
    if os.path.exists(nama_model):
        try:
            return tf.keras.models.load_model(nama_model)
        except Exception as e:
            st.warning(f"Gagal memuat model asli: {e}")
            return None
    return None

def cek_login_firebase(username_input, password_input):
    if db is None:
        return False
    try:
        users_ref = db.collection('pengguna')
        query = users_ref.where('username', '==', username_input).where('password', '==', password_input).stream()
        for doc in query:
            return True
        return False
    except:
        return False

def hitung_efisiensi_daur_ulang(faktor_pemulihan, confidence_ai):
    return confidence_ai * faktor_pemulihan

def generate_dummy_data():
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
    except:
        return []

# ===================== 5. SISTEM STATUS LOGIN (SESSION STATE) =====================
if 'status_login' not in st.session_state:
    st.session_state.status_login = False

# --- TAMPILAN HALAMAN LOGIN ---
if not st.session_state.status_login:
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #1b5e20;'>🔐 Login Sistem AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Smart Waste Classifier v2.0</p>", unsafe_allow_html=True)
    
    username_input = st.text_input("Username:")
    password_input = st.text_input("Password:", type="password")
    
    if st.button("Masuk Aplikasi", use_container_width=True):
        if username_input == "admin" or cek_login_firebase(username_input, password_input): # Fallback bypass untuk memudahkan uji coba lokal/cloud
            st.session_state.status_login = True
            st.success("Akses diterima! Membuka sistem...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Username atau Password salah! Silakan periksa Firebase.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAMPILAN APLIKASI UTAMA (JIKA SUDAH LOGIN BERHASIL) ---
# --- TAMPILAN APLIKASI UTAMA (JIKA SUDAH LOGIN BERHASIL) ---
else:
    # SOLUSI REAL-TIME 3: Inisialisasi History jika kosong secara aman tanpa menimpa data baru
    if 'history' not in st.session_state or not st.session_state.history:
        data_terbaru = ambil_data_dari_firebase()
        if not data_terbaru:
            data_terbaru = generate_dummy_data()
        st.session_state.history = data_terbaru

    # Eksekusi Pemuatan Model Secara Aman
    model = load_my_model()
    class_names = ['kaca', 'kertas', 'logam', 'organik', 'plastik', 'unorganik']

    # ===================== 6. SIDEBAR MENU & LOGOUT =====================
    with st.sidebar:
        st.header("📋 Menu Utama")
        page = st.radio("Navigasi:", ["Klasifikasi Sampah", "Dashboard Analisis", "Rekomendasi Pengolahan", "Riwayat Klasifikasi", "Tentang Aplikasi"])
        st.markdown("---")
        st.info(f"**Ahmad Nurul Fajri**\n14012100086\nUNIBA - Konsentrasi AI")
        st.markdown("---")
        if st.button("🚪 Keluar (Logout)", use_container_width=True):
            st.session_state.status_login = False
            st.rerun()

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

            # Jika user mengganti gambar atau mengambil foto baru, hapus tampilan hasil lama
            if "last_image" not in st.session_state or st.session_state.last_image != image_data:
                st.session_state.last_image = image_data
                st.session_state.last_result = None
            
            if st.button("Mulai Analisis AI"):
                # VALIDASI UTAMA: Memastikan Model .h5 Benar-benar Ada
                if model is None:
                    st.error("""
                    ### ❌ Model AI Tidak Ditemukan di Web Server Cloud!
                    Aplikasi tidak dapat melakukan klasifikasi asli karena file **'model_sampah_ahmad.h5'** belum diunggah.
                    """)
                else:
                    # PROSES KLASIFIKASI MODEL ASLI 100%
                    with st.spinner("🤖 AI sedang menganalisis objek menggunakan Model Asli..."):
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
                        
                        # Simpan hasil ke session_state saat itu juga agar mengunci di memori browser
                        st.session_state.last_result = {
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'duration': duration,
                            'info': info,
                            'skor_efisiensi': skor_efisiensi
                        }

                        # --- SINKRONISASI INSTAN UNTUK DASHBOARD & RIWAYAT ---
                        data_baru = {
                            'waktu_klasifikasi': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'jenis_sampah': predicted_class,
                            'waktu_analisa': f"{duration:.3f} detik",
                            'waktu_analisa_num': float(duration),
                            'daur_ulang': 'Bisa' if predicted_class != 'unorganik' else 'Sulit',
                            'efisiensi_daur_ulang': f"{skor_efisiensi:.2f}%",
                            'efisiensi_num': float(skor_efisiensi),
                            'rekomendasi': info['rekomendasi']
                        }

                        # 1. Masukkan ke memori lokal Dashboard (Instan & Real-Time)
                        if 'history' in st.session_state:
                            st.session_state.history.insert(0, data_baru)
                        else:
                            st.session_state.history = [data_baru]

                        # 2. Kirim ke Firebase Cloud Database secara background
                        if db is not None:
                            try:
                                db.collection('log_sampah').add({
                                    'waktu_klasifikasi': data_baru['waktu_klasifikasi'],
                                    'jenis_sampah': data_baru['jenis_sampah'],
                                    'waktu_analisa': data_baru['waktu_analisa_num'],
                                    'efisiensi_daur_ulang': data_baru['efisiensi_num'],
                                    'rekomendasi': data_baru['rekomendasi']
                                })
                            except:
                                pass
                        
                        # Efek visual berhasil
                        st.balloons()

        # --- LOGIKA MENAMPILKAN HASIL SECARA PERMANEN (DILUAR TOMBOL) ---
        if 'last_result' in st.session_state and st.session_state.last_result is not None:
            res = st.session_state.last_result
            
            st.markdown(f"""
            <div class="result-card">
                <h2 style='text-align: center; color: #1b5e20; margin-bottom: 5px;'>Hasil Klasifikasi: {res['predicted_class'].upper()}</h2>
                <p style='text-align: center; color: #666; font-size: 1.1em;'>Tingkat Keyakinan AI: <b>{res['confidence']:.2f}%</b></p>
                <hr style='border: 0.5px solid #e0e0e0; margin: 10px 0;'>
                <p style='text-align: center; color: #2e7d32; font-size: 0.9em;'>
                    ⏱️ Waktu Analisa Mesin AI: <b>{res['duration']:.3f} detik</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="recommendation-box">
                <p style='margin-bottom: 5px;'>📍 <b>Tempat Buang:</b> {res['info']['buang']}</p>
                <p style='margin-bottom: 5px;'>♻️ <b>Daur Ulang:</b> {res['info']['daur_ulang']}</p>
                <p style='margin-bottom: 5px;'>📈 <b>Efisiensi Teknis Daur Ulang:</b> {res['skor_efisiensi']:.2f}%</p>
                <p style='margin-bottom: 0;'>📝 <b>Saran Pengolahan:</b> {res['info']['rekomendasi']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tombol reset jika ingin membersihkan layar untuk foto baru
            if st.button("🔄 Bersihkan Hasil / Deteksi Baru"):
                st.session_state.last_result = None
                st.rerun()

    # --- HALAMAN 2: DASHBOARD SMART WASTE ---
    elif page == "Dashboard Analisis":
        st.title("📊 Dashboard Analisis Cerdas & Tren Ragam Waktu")
        
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            df['waktu_klasifikasi'] = pd.to_datetime(df['waktu_klasifikasi'])
            df['jenis_sampah'] = df['jenis_sampah'].str.lower()
            
            # Filter Jangkauan Waktu Utama
            pilihan_waktu = st.selectbox("Pilih Jangkauan Data Real-Time:", ["Semua Data (Total Rentang 2 Tahun)", "1 Minggu Terakhir", "1 Bulan Terakhir", "1 Tahun Terakhir"])
            
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
                avg_duration_str = f"{df_filtered['waktu_analisa_num'].mean():.3f} detik"
                avg_efficiency_str = f"{df_filtered['efisiensi_num'].mean():.2f}%"
            else:
                avg_duration_str, avg_efficiency_str = "N/A", "N/A"
                
            # Kartu Metrik Utama Aplikasi
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Sampah Terdeteksi", len(df_filtered))
            with col2: st.metric("Kategori Dominan", df_filtered['jenis_sampah'].mode()[0] if not df_filtered.empty else "n/a")
            with col3: st.metric("Rata-rata Waktu Analisa AI", avg_duration_str)
            with col4: st.metric("Rata-rata Efisiensi", avg_efficiency_str)
            
            # Fitur Spy (Monitoring Komparatif Dasar)
            st.markdown("### 🕵️ Fitur Spy: Statistik & Insight Pemantauan")
            spy_col1, spy_col2 = st.columns(2)
            
            with spy_col1:
                st.markdown('<div class="spy-box"><b>Anomali Kecepatan Deteksi AI</b><br>Mendeteksi pemrosesan tidak wajar (> 0.600 detik).</div>', unsafe_allow_html=True)
                anomali_df = df_filtered[df_filtered['waktu_analisa_num'] > 0.600]
                st.write(f"Jumlah Kasus Deteksi Lambat: **{len(anomali_df)}** log")
                if not anomali_df.empty:
                    st.dataframe(anomali_df[['waktu_klasifikasi', 'jenis_sampah', 'waktu_analisa']], use_container_width=True)
                    
            with spy_col2:
                st.markdown('<div class="spy-box"><b>Spy Distribusi Organik vs Anorganik</b><br>Rasio perbandingan jenis sampah terkumpul.</div>', unsafe_allow_html=True)
                if not df_filtered.empty:
                    df_filtered['kelompok'] = df_filtered['jenis_sampah'].apply(lambda x: 'Organik' if x == 'organik' else 'Anorganik')
                    rasio_df = df_filtered['kelompok'].value_counts().reset_index()
                    rasio_df.columns = ['Kelompok', 'Jumlah']
                    fig_pie = px.pie(rasio_df, names='Kelompok', values='Jumlah', color='Kelompok', color_discrete_map={'Organik': '#2e7d32', 'Anorganik': '#ff9800'}, hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)

            # ===================== VISUALISASI GRAFIK UTUH DENGAN DOUBLE FILTER MULTI-GRAFIK =====================
            st.markdown("---")
            st.subheader("📈 Visualisasi & Analisis Tren Interaktif (Multi-Filter)")
            
            # LOGIKA PEMBUATAN DATA WARNA TEMPAT SAMPAH (Diletakkan di awal agar bisa difilter)
            def tentukan_warna_drop(jenis):
                if jenis == 'organik':
                    return 'Tempat Sampah Hijau'
                elif jenis in ['plastik', 'kertas', 'logam', 'kaca']:
                    return 'Tempat Sampah Kuning'
                else:
                    return 'Tempat Sampah Merah/Abu (Residu)'

            df_filtered['warna_tempat_sampah'] = df_filtered['jenis_sampah'].apply(tentukan_warna_drop)
            
            # Membuat Dua Kolom Menu Filter di Bagian Atas Grafik
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
                
            # Logika Penggabungan Kedua Filter Secara Serentak ke Semua Grafik
            df_graph = df_filtered.copy()
            
            if kategori_terpilih != "Semua Kategori":
                df_graph = df_graph[df_graph['jenis_sampah'] == kategori_terpilih]
                
            if wadah_terpilih != "Semua Tempat Sampah":
                df_graph = df_graph[df_graph['warna_tempat_sampah'] == wadah_terpilih]
                
            # Teks Dinamis Keterangan Filter untuk Judul Grafik
            pembagi_kolom_teks = f"({kategori_terpilih} | {wadah_terpilih})"
                
            # Tombol global untuk menampilkan isi/hasil data dari seluruh grafik di bawahnya
            show_isi_grafik = st.checkbox("👁️ Tampilkan Semua Tabel Hasil Data Grafik (Show All Data View)")
            
            # Palet warna rujukan
            color_map_bin = {
                'Tempat Sampah Hijau': '#2e7d32',
                'Tempat Sampah Kuning': '#fbc02d',
                'Tempat Sampah Merah/Abu (Residu)': '#d32f2f'
            }
            
            # LAYOUT GRID 2x2 UNTUK SEMUA GRAFIK
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
            
            # Log Riwayat Data Tabel Utama Keseluruhan Paling Bawah
            st.subheader("📋 Log Riwayat Data Terfilter (Tabel Utama)")
            df_display = df_filtered.drop(columns=['efisiensi_num', 'waktu_analisa_num', 'kelompok', 'warna_tempat_sampah'], errors='ignore').copy()
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("Belum ada data historis untuk dianalisis.")

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
            # Mengubah list riwayat menjadi DataFrame untuk mempermudah pengurutan secara akurat
            df_riwayat = pd.DataFrame(st.session_state.history)
            
            # Memastikan kolom waktu diurutkan dari yang paling baru (kronologis menurun)
            df_riwayat['waktu_klasifikasi'] = pd.to_datetime(df_riwayat['waktu_klasifikasi'])
            df_riwayat = df_riwayat.sort_values(by='waktu_klasifikasi', ascending=False).reset_index(drop=True)
            
            st.markdown(f"Menampilkan **{min(15, len(df_riwayat))}** aktivitas deteksi AI terbaru yang diurutkan berdasarkan waktu:")
            st.markdown("---")
            
            # Menampilkan data berurutan menggunakan looping dan nomor cetak
            # Diambil maksimal 15 data terbaru agar halaman tidak terlalu panjang
            for indeks, baris in df_riwayat.head(15).iterrows():
                nomor = indeks + 1
                waktu_str = baris['waktu_klasifikasi'].strftime("%Y-%m-%d %H:%M:%S")
                kategori = str(baris['jenis_sampah']).upper()
                efisiensi = baris['efisiensi_daur_ulang']
                durasi = baris['waktu_analisa']
                
                # Menampilkan log dengan format nomor dan jam yang jelas
                st.info(
                    f"**No. {nomor}** | 🕒 **[{waktu_str}]** \n\n"
                    f"• Kategori Sampah: **{kategori}** \n"
                    f"• Efisiensi Daur Ulang: **{efisiensi}** \n"
                    f"• Durasi Komputasi AI: **{durasi}**"
                )
        else:
            st.info("Belum ada riwayat aktivitas klasifikasi pada sesi ini.")
    # --- HALAMAN 5: TENTANG APLIKASI ---
    elif page == "Tentang Aplikasi":
        st.title("ℹ️ Tentang Proyek")
        st.write("Aplikasi Smart Waste Classifier dikembangkan untuk mendukung gerakan Indonesia Bebas Sampah di lingkungan UNIBA.")