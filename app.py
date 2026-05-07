# --- app.py ---
import streamlit as st
import pandas as pd
import math
import io
import requests # <--- IMPORT BARU UNTUK MENGAMBIL GAMBAR STEAM
import matplotlib.pyplot as plt

from scraper import crawl_steam
from nlp_core import clean_text, run_topic_modeling
from ai_helper import generate_ai_insight, generate_topic_labels_with_ai 

# --- FUNGSI BARU UNTUK MENGAMBIL INFO GAME DARI STEAM ---
@st.cache_data
def get_game_info(app_id):
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    try:
        response = requests.get(url).json()
        if response[str(app_id)]['success']:
            data = response[str(app_id)]['data']
            return data['name'], data['header_image']
    except Exception:
        pass
    # Fallback jika gagal/App ID salah
    return "Unknown Game", "https://community.akamai.steamstatic.com/public/shared/images/header/globalheader_logo.png"

# --- 1. PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="Steam AI Analytics", page_icon="🎮", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #00E5FF;}
    .stButton>button {
        width: 100%; border-radius: 8px;
        background-color: #00E5FF; color: #0E1117; font-weight: bold;
    }
    .stButton>button:hover {background-color: #00B3CC; color: white;}
    .st-emotion-cache-1v0mbdj {border-radius: 10px; border: 1px solid #1E2329;}
    </style>
""", unsafe_allow_html=True)

# --- 2. SETUP SESSION STATE ---
if 'data_diproses' not in st.session_state:
    st.session_state.data_diproses = False

# --- 3. SIDEBAR (KONFIGURASI) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/83/Steam_icon_logo.svg", width=60)
    st.title("⚙️ Konfigurasi")
    
    app_id = st.text_input("Steam App ID", value="3551340", help="Contoh: 3551340 (Marvel Rivals)")
    
    col1, col2 = st.columns(2)
    with col1:
        review_type = st.selectbox("Sentimen", ["negative", "positive", "all"])
    with col2:
        language = st.selectbox("Bahasa", ["english", "indonesian"])
        
    num_topics = st.slider("Jumlah Topik (LDA)", 2, 8, 4)
    
    with st.expander("🛠️ Advanced Settings (TF-IDF)"):
        min_df_val = st.number_input("Min Document Frequency", value=5)
        max_df_val = st.slider("Max Document Frequency", 0.5, 1.0, 0.85)
        
    btn_proses = st.button("🚀 Mulai Analisis Data")

# --- 4. TAMPILAN UTAMA (HEADER) ---
st.title("🎮 Steam Review AI Analytics")
st.markdown("Ubah ribuan ulasan pemain menjadi **strategi bisnis yang dapat dieksekusi** menggunakan algoritma *Machine Learning (LDA)* dan *Generative AI*.")
st.divider()

# --- 5. LOGIKA PEMROSESAN ---
if btn_proses:
    try:
        with st.status("Memproses Data Steam...", expanded=True) as status:
            # 1. Mengambil Info Game (GAMBAR & NAMA)
            st.write("🌍 Mengambil metadata game...")
            g_name, g_image = get_game_info(app_id)

            # 2. Mengambil Ulasan
            st.write("⬇️ Mengunduh ulasan dari Steam...")
            df_raw = crawl_steam(app_id, review_type, language)
            
            # 3. NLP
            st.write("🔄 Membersihkan teks & Tokenisasi...")
            df_clean = clean_text(df_raw)
            
            # 4. LDA Topic Modeling
            st.write("🧠 Melatih Model Topic Modeling (LDA)...")
            lda_components, nama_fitur = run_topic_modeling(df_clean, num_topics, min_df_val, max_df_val)
            
            # 5. Label AI
            st.write("🤖 AI sedang membaca konteks & memberi nama topik...")
            topics_for_ai = {idx + 1: [nama_fitur[i] for i in topic.argsort()[-8:][::-1]] for idx, topic in enumerate(lda_components)}
            ai_labels = generate_topic_labels_with_ai(topics_for_ai, language)
            
            status.update(label="Analisis Selesai!", state="complete", expanded=False)
            
            # SIMPAN SEMUA KE STATE (TERMASUK NAMA & GAMBAR)
            st.session_state.update({
                'game_name': g_name, 'game_image': g_image, # INFO GAME BARU
                'df_raw': df_raw, 'df_clean': df_clean, 'lda_components': lda_components,
                'nama_fitur': nama_fitur, 'app_id': app_id, 'review_type': review_type,
                'num_topics': num_topics, 'topic_labels': ai_labels, 'data_diproses': True
            })
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# --- 6. MENAMPILKAN HASIL (UI MODERN TABS) ---
if st.session_state.data_diproses:
    
    # --- UI BARU: MENAMPILKAN PROFIL GAME ---
    with st.container(border=True):
        col_img, col_info = st.columns([1, 3])
        with col_img:
            st.image(st.session_state.game_image, use_container_width=True)
        with col_info:
            st.subheader(f"📊 Analisis Game: {st.session_state.game_name}")
            st.write(f"**App ID:** `{st.session_state.app_id}` | **Tipe Ulasan:** `{st.session_state.review_type.upper()}` | **Bahasa:** `{st.session_state.language.title()}`")
            st.write(f"Berhasil memproses **{len(st.session_state.df_clean)}** ulasan pemain.")
            
    st.write("") # Spasi kosong
    
    # MEMBUAT TABS
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard Topik", "🤖 AI Strategist", "📂 Database Ulasan"])

    # TAB 1: VISUALISASI & KATA KUNCI
    with tab1:
        st.subheader("Distribusi Topik Pembicaraan")
        col_topics = st.columns(math.ceil(st.session_state.num_topics / 2))
        
        for idx, topic in enumerate(st.session_state.lda_components):
            top_idx = topic.argsort()[-6:][::-1]
            words = [st.session_state.nama_fitur[i] for i in top_idx]
            judul_ai = st.session_state.topic_labels[idx]
            
            with st.container(border=True):
                st.markdown(f"#### {judul_ai}")
                st.write(f"*{', '.join(words)}*")

        st.divider()
        st.subheader("Visualisasi Bobot Kata")
        cols = 2
        rows = math.ceil(st.session_state.num_topics / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), facecolor='#0E1117')
        axes_flat = axes.flatten() if st.session_state.num_topics > 1 else [axes]

        for i, topic in enumerate(st.session_state.lda_components):
            top_f_ind = topic.argsort()[-8:] 
            top_f = [st.session_state.nama_fitur[j] for j in top_f_ind]
            weights = topic[top_f_ind]

            ax = axes_flat[i]
            ax.set_facecolor('#0E1117')
            ax.barh(top_f, weights, color='#00E5FF')
            ax.set_title(st.session_state.topic_labels[i], color='white', fontweight='bold')
            ax.tick_params(colors='white')
            ax.invert_yaxis()

        for j in range(i + 1, len(axes_flat)): 
            fig.delaxes(axes_flat[j])
            
        plt.tight_layout()
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', facecolor='#0E1117')
        byte_im = buf.getvalue()

        st.download_button(
            label="🖼️ Download Grafik (PNG)",
            data=byte_im,
            file_name=f"visualisasi_topik_{st.session_state.app_id}.png",
            mime="image/png",
            use_container_width=True
        )

    # TAB 2: AI INSIGHT
    with tab2:
        st.subheader("Saran Strategis dari AI")
        if st.button("✨ Generate Analisis Mendalam", type="primary", use_container_width=True):
            with st.spinner("AI sedang memformulasikan strategi..."):
                all_topics = [f"{st.session_state.topic_labels[idx]}: {', '.join([st.session_state.nama_fitur[i] for i in topic.argsort()[-5:][::-1]])}" for idx, topic in enumerate(st.session_state.lda_components)]
                
                # SEKARANG KITA MEMBERITAHU NAMA GAMENYA KE AI!
                prompt_untuk_ai = f"NAMA GAME: {st.session_state.game_name}\n\nTOPIK:\n" + "\n".join(all_topics)
                
                insight = generate_ai_insight(prompt_untuk_ai, st.session_state.review_type, st.session_state.app_id)
                
                with st.container(border=True):
                    st.markdown(insight)

    # TAB 3: DATA RAW & DOWNLOAD
    with tab3:
        st.subheader("Tabel Data Ulasan")
        st.dataframe(st.session_state.df_clean[['Review_ID', 'Playtime_Minutes', 'Review_Text', 'Cleaned_Review']], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download Raw CSV", st.session_state.df_raw.to_csv(index=False).encode('utf-8'), f"raw_{st.session_state.app_id}.csv", "text/csv", use_container_width=True)
        with col2:
            st.download_button("📥 Download Clean CSV", st.session_state.df_clean.to_csv(index=False).encode('utf-8'), f"clean_{st.session_state.app_id}.csv", "text/csv", use_container_width=True)
