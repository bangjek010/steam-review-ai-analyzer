import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import json
import numpy as np


from scraper import crawl_steam
from nlp_core import clean_text, run_topic_modeling


from ai_helper import generate_ai_insight, generate_topic_labels_with_ai, is_ai_connected 


def get_saved_analyses(current_app_id):
    cache_root = os.path.join("data", "analysis_results")
    if not os.path.exists(cache_root):
        return []
    try:
        folders = [f for f in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, f))]
    except Exception:
        return []
    
    safe_current_app_id = "".join([c for c in str(current_app_id) if c.isalnum() or c in ('_', '-')])
    
    options = []
    for f in folders:
        if f.startswith("analysis_"):
            parts = f.split("_")
            if len(parts) >= 7:
                app_id_val = parts[1]
                

                if app_id_val != safe_current_app_id:
                    continue
                
                rev_type_val = parts[2]
                lang_val = parts[3]
                topics_val = parts[4]
                min_df_val = parts[5]
                max_df_val = parts[6]
                

                display_name = f"📝 {rev_type_val} | {lang_val} | Tpc: {topics_val} | df: {min_df_val}"
                options.append({
                    "dir_name": f,
                    "display": display_name,
                    "params": {
                        "app_id": app_id_val,
                        "review_type": rev_type_val,
                        "language": lang_val,
                        "num_topics": int(topics_val),
                        "min_df_val": int(min_df_val) if min_df_val.isdigit() else 5,
                        "max_df_val": float(max_df_val) if max_df_val.replace('.', '', 1).isdigit() else 0.85
                    }
                })
    return options

def get_steam_game_details(app_id):
    import urllib.request
    import json
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            if str(app_id) in data and data[str(app_id)]['success']:
                app_data = data[str(app_id)]['data']
                return {
                    'name': app_data.get('name', f"Steam App {app_id}"),
                    'image': app_data.get('header_image', f"https://shared.fastly.steamstatic.com/store_images_api/v2/50/apps/{app_id}/header.jpg")
                }
    except Exception:
        pass
    return {
        'name': f"Steam App {app_id}",
        'image': f"https://shared.fastly.steamstatic.com/store_images_api/v2/50/apps/{app_id}/header.jpg"
    }

def get_all_saved_analyses():
    cache_root = os.path.join("data", "analysis_results")
    if not os.path.exists(cache_root):
        return []
    try:
        folders = [f for f in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, f))]
    except Exception:
        return []
    
    options = []
    for f in folders:
        if f.startswith("analysis_"):
            parts = f.split("_")
            if len(parts) >= 7:
                app_id_val = parts[1]
                rev_type_val = parts[2]
                lang_val = parts[3]
                topics_val = parts[4]
                min_df_val = parts[5]
                max_df_val = parts[6]
                
                json_path = os.path.join(cache_root, f, "analysis_data.json")
                game_name = f"Steam App {app_id_val}"
                game_image = f"https://shared.fastly.steamstatic.com/store_images_api/v2/50/apps/{app_id_val}/header.jpg"
                total_reviews = 0
                
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as file_json:
                            cache_data = json.load(file_json)
                            game_name = cache_data.get("game_name", game_name)
                            game_image = cache_data.get("game_image", game_image)
                    except Exception:
                        pass
                
                clean_csv_path = os.path.join(cache_root, f, "clean.csv")
                if os.path.exists(clean_csv_path):
                    try:
                        with open(clean_csv_path, 'r', encoding='utf-8') as f_csv:
                            total_reviews = sum(1 for _ in f_csv) - 1
                    except Exception:
                        pass
                
                options.append({
                    "dir_name": f,
                    "game_name": game_name,
                    "game_image": game_image,
                    "total_reviews": total_reviews,
                    "params": {
                        "app_id": app_id_val,
                        "review_type": rev_type_val,
                        "language": lang_val,
                        "num_topics": int(topics_val),
                        "min_df_val": int(min_df_val) if min_df_val.isdigit() else 5,
                        "max_df_val": float(max_df_val) if max_df_val.replace('.', '', 1).isdigit() else 0.85
                    }
                })
    return options

def load_saved_analysis_from_disk(dir_name, p):
    cache_dir = os.path.join("data", "analysis_results", dir_name)
    raw_csv_path = os.path.join(cache_dir, "raw.csv")
    clean_csv_path = os.path.join(cache_dir, "clean.csv")
    json_path = os.path.join(cache_dir, "analysis_data.json")
    
    if os.path.exists(raw_csv_path) and os.path.exists(clean_csv_path) and os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        st.session_state.df_raw = pd.read_csv(raw_csv_path)
        st.session_state.df_clean = pd.read_csv(clean_csv_path)
        st.session_state.lda_components = np.array(cache_data["lda_components"])
        st.session_state.nama_fitur = np.array(cache_data["nama_fitur"])
        st.session_state.topic_labels = cache_data["topic_labels"]
        st.session_state.ai_insight = cache_data.get("ai_insight")
        
        game_name = cache_data.get("game_name")
        game_image = cache_data.get("game_image")
        if not game_name or not game_image:
            details = get_steam_game_details(p["app_id"])
            game_name = details['name']
            game_image = details['image']
            cache_data["game_name"] = game_name
            cache_data["game_image"] = game_image
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=4)
        st.session_state.game_name = game_name
        st.session_state.game_image = game_image
        
        st.session_state.app_id = p["app_id"]
        st.session_state.review_type = p["review_type"]
        st.session_state.language = p["language"]
        st.session_state.num_topics = p["num_topics"]
        st.session_state.min_df_val = p["min_df_val"]
        st.session_state.max_df_val = p["max_df_val"]
        
        st.session_state.cache_dir = cache_dir
        st.session_state.data_diproses = True
        return True
    return False



st.set_page_config(page_title="Steam Review Topic Analyzer", layout="wide")


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Sidebar Styling */
    div[data-testid="stSidebar"] .stSelectbox label {
        font-size: 13px !important;
        font-weight: bold;
    }
    div[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        font-size: 11px !important;
    }
    
    /* ================= LIGHT MODE (DEFAULT) ================= */
    .stat-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .badge-id { background-color: #f1f5f9; color: #1e293b; border: 1px solid #e2e8f0; }
    .badge-count { background-color: rgba(37, 99, 235, 0.12); color: #2563eb; border: 1px solid rgba(37, 99, 235, 0.25); }
    .badge-type { background-color: rgba(219, 39, 119, 0.12); color: #db2777; border: 1px solid rgba(219, 39, 119, 0.25); }
    .badge-lang { background-color: rgba(5, 150, 105, 0.12); color: #059669; border: 1px solid rgba(5, 150, 105, 0.25); }

    .game-detail-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        min-height: 160px;
        color: #1e293b;
    }

    .topic-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .topic-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.15);
    }
    .topic-card-title {
        font-size: 14px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 10px;
    }
    .keyword-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .keyword-tag {
        background-color: #ffffff;
        color: #334155;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 500;
        border: 1px solid #cbd5e1;
    }

    div[data-testid="stMarkdownContainer"] blockquote {
        background-color: #f1f5f9 !important;
        border-left: 5px solid #818cf8 !important;
        color: #1e293b !important;
        padding: 20px !important;
        border-radius: 8px !important;
        margin: 15px 0 !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }
    
    /* Button Customization */
    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 16px -4px rgba(79, 70, 229, 0.4) !important;
    }

    /* ================= DARK MODE OVERRIDES ================= */
    @media (prefers-color-scheme: dark) {
        .badge-id { background-color: #1e293b; color: #cbd5e1; border: 1px solid #334155; }
        .game-detail-card {
            background-color: #111827;
            border: 1px solid #1f2937;
            color: #cbd5e1;
        }
        .topic-card {
            background-color: #111827;
            border: 1px solid #1f2937;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        .topic-card-title {
            color: #f1f5f9;
        }
        .keyword-tag {
            background-color: #1f2937;
            color: #cbd5e1;
            border: 1px solid #374151;
        }
        div[data-testid="stMarkdownContainer"] blockquote {
            background-color: #111827 !important;
            color: #cbd5e1 !important;
        }
    }

    /* Target Streamlit's Dark Theme Class Elements explicitly */
    [data-theme="dark"] .badge-id,
    .stApp[data-theme="dark"] .badge-id {
        background-color: #1e293b; color: #cbd5e1; border: 1px solid #334155;
    }
    [data-theme="dark"] .game-detail-card,
    .stApp[data-theme="dark"] .game-detail-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        color: #cbd5e1;
    }
    [data-theme="dark"] .topic-card,
    .stApp[data-theme="dark"] .topic-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    [data-theme="dark"] .topic-card-title,
    .stApp[data-theme="dark"] .topic-card-title {
        color: #f1f5f9;
    }
    [data-theme="dark"] .keyword-tag,
    .stApp[data-theme="dark"] .keyword-tag {
        background-color: #1f2937;
        color: #cbd5e1;
        border: 1px solid #374151;
    }
    [data-theme="dark"] div[data-testid="stMarkdownContainer"] blockquote,
    .stApp[data-theme="dark"] div[data-testid="stMarkdownContainer"] blockquote {
        background-color: #111827 !important;
        color: #cbd5e1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if 'data_diproses' not in st.session_state:
    st.session_state.data_diproses = False
if 'ai_insight' not in st.session_state:
    st.session_state.ai_insight = None
if 'topic_labels' not in st.session_state:
    st.session_state.topic_labels = []
if 'cache_dir' not in st.session_state:
    st.session_state.cache_dir = ""


st.sidebar.header("⚙️ Konfigurasi Analisis")


default_app_id = st.session_state.get("app_id", "3551340")
app_id = st.sidebar.text_input("Steam App ID", value=default_app_id)

default_review_type = st.session_state.get("review_type", "negative")
review_types = ["negative", "positive", "all"]
try:
    rev_index = review_types.index(default_review_type)
except ValueError:
    rev_index = 0
review_type = st.sidebar.selectbox("Jenis Ulasan", review_types, index=rev_index)

default_language = st.session_state.get("language", "english")
languages = ["english", "indonesian"]
try:
    lang_index = languages.index(default_language)
except ValueError:
    lang_index = 0
language = st.sidebar.selectbox("Bahasa", languages, index=lang_index)

default_num_topics = st.session_state.get("num_topics", 2)
num_topics = st.sidebar.slider("Jumlah Topik LDA", 2, 4, int(default_num_topics))

st.sidebar.markdown("---")
st.sidebar.subheader("TF-IDF Settings")
default_min_df = st.session_state.get("min_df_val", 5)
min_df_val = st.sidebar.number_input("Min Document Frequencay (min_df)", value=int(default_min_df))

default_max_df = st.session_state.get("max_df_val", 0.85)
max_df_val = st.sidebar.slider("Max Document Frequency (max_df)", 0.5, 1.0, float(default_max_df))


overwrite_analysis = st.sidebar.checkbox("Overwrite Analisis Sebelumnya", value=False)

btn_proses = st.sidebar.button("🚀 Mulai Analisis")


saved_analyses = get_saved_analyses(app_id)
if saved_analyses:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Riwayat Analisis Tersimpan")
    

    saved_displays = [opt["display"] for opt in saved_analyses]
    selected_display = st.sidebar.selectbox("Pilih Analisis Sebelumnya:", saved_displays)
    

    selected_opt = next(opt for opt in saved_analyses if opt["display"] == selected_display)
    
    if st.sidebar.button("📂 Muat Analisis"):
        if load_saved_analysis_from_disk(selected_opt["dir_name"], selected_opt["params"]):
            st.success(f"Analisis untuk App ID {selected_opt['params']['app_id']} berhasil dimuat!")
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
        else:
            st.sidebar.error("File analisis yang dipilih tidak lengkap atau rusak.")


st.title("🎮 Steam Review Topic Analyzer")
st.markdown("Aplikasi ini membantu kamu mengekstraksi topik keluhan/pujian dari ulasan Steam menggunakan algoritma LDA yang dipadukan dengan Generative AI.")


if btn_proses:
    try:
       
        safe_app_id = "".join([c for c in str(app_id) if c.isalnum() or c in ('_', '-')])
        dir_name = f"analysis_{safe_app_id}_{review_type}_{language}_{num_topics}_{min_df_val}_{max_df_val}"
        cache_dir = os.path.join("data", "analysis_results", dir_name)
        
        raw_csv_path = os.path.join(cache_dir, "raw.csv")
        clean_csv_path = os.path.join(cache_dir, "clean.csv")
        json_path = os.path.join(cache_dir, "analysis_data.json")
        
        has_cache = os.path.exists(raw_csv_path) and os.path.exists(clean_csv_path) and os.path.exists(json_path)
        

        if has_cache and not overwrite_analysis:
            with st.spinner("Memuat hasil analisis dari penyimpanan lokal..."):
                with open(json_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                st.session_state.df_raw = pd.read_csv(raw_csv_path)
                st.session_state.df_clean = pd.read_csv(clean_csv_path)
                st.session_state.lda_components = np.array(cache_data["lda_components"])
                st.session_state.nama_fitur = np.array(cache_data["nama_fitur"])
                st.session_state.topic_labels = cache_data["topic_labels"]
                st.session_state.ai_insight = cache_data.get("ai_insight")
                
     
                game_name = cache_data.get("game_name")
                game_image = cache_data.get("game_image")
                if not game_name or not game_image:
                    details = get_steam_game_details(app_id)
                    game_name = details['name']
                    game_image = details['image']
                    cache_data["game_name"] = game_name
                    cache_data["game_image"] = game_image
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=4)
                st.session_state.game_name = game_name
                st.session_state.game_image = game_image
                
                st.session_state.app_id = app_id
                st.session_state.review_type = review_type
                st.session_state.num_topics = num_topics
                st.session_state.language = language
                st.session_state.min_df_val = min_df_val
                st.session_state.max_df_val = max_df_val
                st.session_state.cache_dir = cache_dir
                st.session_state.data_diproses = True
                st.success("🔄 Berhasil memuat analisis dari cache lokal (menghemat token)!")
        else:

            ai_ok = is_ai_connected()
            

            if has_cache and overwrite_analysis and not ai_ok:
                st.warning("⚠️ Koneksi AI/API terputus. Tidak dapat melakukan overwrite analisis. Memuat data cache sebelumnya.")
                with st.spinner("Memuat hasil analisis dari penyimpanan lokal..."):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    st.session_state.df_raw = pd.read_csv(raw_csv_path)
                    st.session_state.df_clean = pd.read_csv(clean_csv_path)
                    st.session_state.lda_components = np.array(cache_data["lda_components"])
                    st.session_state.nama_fitur = np.array(cache_data["nama_fitur"])
                    st.session_state.topic_labels = cache_data["topic_labels"]
                    st.session_state.ai_insight = cache_data.get("ai_insight")
                    

                    game_name = cache_data.get("game_name")
                    game_image = cache_data.get("game_image")
                    if not game_name or not game_image:
                        details = get_steam_game_details(app_id)
                        game_name = details['name']
                        game_image = details['image']
                        cache_data["game_name"] = game_name
                        cache_data["game_image"] = game_image
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(cache_data, f, ensure_ascii=False, indent=4)
                    st.session_state.game_name = game_name
                    st.session_state.game_image = game_image
                    
                    st.session_state.app_id = app_id
                    st.session_state.review_type = review_type
                    st.session_state.num_topics = num_topics
                    st.session_state.language = language
                    st.session_state.min_df_val = min_df_val
                    st.session_state.max_df_val = max_df_val
                    st.session_state.cache_dir = cache_dir
                    st.session_state.data_diproses = True
            else:

                with st.spinner("Sedang memproses data... Mohon tunggu sebentar."):
                    st.info("⬇️ Mengunduh ulasan dari Steam...")
                    df_raw = crawl_steam(app_id, review_type, language)
                    
                    st.info("🎮 Mengambil detail game...")
                    details = get_steam_game_details(app_id)
                    game_name = details['name']
                    game_image = details['image']
                    
                    st.info("🔄 Membersihkan teks...")
                    df_clean = clean_text(df_raw)
                    
                    st.info("🧠 Menghitung TF-IDF & Melatih model LDA...")
                    lda_components, nama_fitur = run_topic_modeling(df_clean, num_topics, min_df_val, max_df_val)
                    

                    st.info("🤖 AI sedang membaca konteks untuk memberikan nama Topik...")

                    topics_for_ai = {}
                    for idx, topic in enumerate(lda_components):
                        top_idx = topic.argsort()[-8:][::-1] 
                        words = [nama_fitur[i] for i in top_idx]
                        topics_for_ai[idx + 1] = words
                    

                    ai_labels = generate_topic_labels_with_ai(topics_for_ai, language)
                
               
                    os.makedirs(cache_dir, exist_ok=True)
                    df_raw.to_csv(raw_csv_path, index=False)
                    df_clean.to_csv(clean_csv_path, index=False)
                    
                    cache_data = {
                        "topic_labels": ai_labels,
                        "nama_fitur": list(nama_fitur),
                        "lda_components": lda_components.tolist() if hasattr(lda_components, 'tolist') else lda_components,
                        "ai_insight": None,
                        "game_name": game_name,
                        "game_image": game_image
                    }
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=4)
                    
                  
                    st.session_state.df_raw = df_raw
                    st.session_state.df_clean = df_clean
                    st.session_state.lda_components = lda_components
                    st.session_state.nama_fitur = nama_fitur
                    st.session_state.app_id = app_id
                    st.session_state.review_type = review_type
                    st.session_state.num_topics = num_topics
                    st.session_state.language = language
                    st.session_state.min_df_val = min_df_val
                    st.session_state.max_df_val = max_df_val
                    st.session_state.topic_labels = ai_labels
                    st.session_state.ai_insight = None
                    st.session_state.game_name = game_name
                    st.session_state.game_image = game_image
                    st.session_state.cache_dir = cache_dir
                    st.session_state.data_diproses = True
                    st.success("✨ Analisis Baru berhasil dijalankan dan disimpan!")
            
    except ValueError:
        st.error("Error pada TF-IDF: Ulasan terlalu sedikit. Coba turunkan nilai 'min_df' menjadi 1 atau 2 di Sidebar.")
    except Exception as e:
        st.error(f"Terjadi kesalahan sistem: {e}")


if st.session_state.data_diproses:

    game_name_display = st.session_state.get("game_name", f"Steam App {st.session_state.app_id}")
    game_image_display = st.session_state.get("game_image", "")
    total_data_display = len(st.session_state.df_raw) if "df_raw" in st.session_state else 0
    
    st.markdown("### 🎮 Game Metadata & Stats")
    info_col1, info_col2 = st.columns([1, 2])
    with info_col1:
        if game_image_display:
            st.image(game_image_display, use_column_width=True)
    with info_col2:
        st.markdown(
            f"""
            <div class="game-detail-card">
                <h3 style="margin: 0 0 12px 0; color: #3b82f6; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 24px; font-weight: 800;">{game_name_display}</h3>
                <div style="margin-bottom: 15px;">
                    <span class="stat-badge badge-id">🆔 App ID: {st.session_state.app_id}</span>
                    <span class="stat-badge badge-count">📊 Total Ulasan: {total_data_display:,}</span>
                </div>
                <div style="font-size: 13px; font-weight: 500;">
                    <b>Jenis Ulasan:</b> <span class="stat-badge badge-type" style="margin: 0 6px 0 0; padding: 2px 8px; font-size: 11px;">{st.session_state.review_type.upper()}</span>
                    &nbsp;|&nbsp;
                    <b>Bahasa:</b> <span class="stat-badge badge-lang" style="margin: 0 0 0 6px; padding: 2px 8px; font-size: 11px;">{st.session_state.language.upper()}</span>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    st.markdown("---")
    
    st.success("✨ Analisis Selesai!")
    
    st.subheader("📋 Preview Data Hasil")
    st.dataframe(st.session_state.df_clean.head(10))

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💡 Kata Kunci per Topik")
        for idx, topic in enumerate(st.session_state.lda_components):
            top_idx = topic.argsort()[-8:][::-1]
            words = [st.session_state.nama_fitur[i] for i in top_idx]
            judul_ai = st.session_state.topic_labels[idx]
            
            # Buat bubble tag HTML untuk 6 kata kunci teratas
            tags_html = "".join([f'<span class="keyword-tag">{w}</span>' for w in words[:6]])
            
            st.markdown(
                f"""
                <div class="topic-card">
                    <div class="topic-card-title">Topik {idx+1} | {judul_ai}</div>
                    <div class="keyword-container">
                        {tags_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            

    with col2:
        st.markdown("### 📂 Download Hasil (CSV)")
        st.write("Unduh data ulasan mentah hasil crawling atau data ulasan bersih setelah prapemrosesan.")
        
        csv_raw = st.session_state.df_raw.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Raw Data", data=csv_raw, file_name=f"raw_{st.session_state.app_id}.csv", mime="text/csv")
        
        st.write("")
        
        csv_clean = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Clean Data", data=csv_clean, file_name=f"clean_{st.session_state.app_id}.csv", mime="text/csv")


    st.markdown("---")
    st.subheader("📊 Visualisasi Kata per Topik")
    

    def check_light_theme():
        try:
            bg = st.get_option("theme.backgroundColor")
            if bg:
                bg = bg.strip().lstrip('#')
                if len(bg) == 3:
                    bg = ''.join([c*2 for c in bg])
                if len(bg) == 6:
                    r = int(bg[0:2], 16)
                    g = int(bg[2:4], 16)
                    b = int(bg[4:6], 16)
                    brightness = (r * 299 + g * 587 + b * 114) / 1000
                    return brightness > 128
        except Exception:
            pass
        try:
            return st.get_option("theme.base") == "light"
        except Exception:
            return False

    is_light = check_light_theme()

    tab_bar, tab_wordcloud = st.tabs(["📊 Distribusi Kata (Bar Chart)", "☁️ Word Cloud per Topik"])

    with tab_bar:
        if is_light:
            text_color = "#1e293b"      # Dark slate-800
            grid_color = "#cbd5e1"      # Slate-300
            axis_color = "#94a3b8"      # Slate-400
            bar_label_color = "#475569" # Slate-600
        else:
            text_color = "#f1f5f9"      # Light grey
            grid_color = "#334155"      # Dark slate
            axis_color = "#475569"      # Medium slate
            bar_label_color = "#94a3b8" # Light slate

        cols = 2
        rows = math.ceil(st.session_state.num_topics / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows))
        axes_flat = axes.flatten() if st.session_state.num_topics > 1 else [axes]

        colors_palette = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b']

        for i, topic in enumerate(st.session_state.lda_components):
            top_f_ind = topic.argsort()[-8:] 
            top_f = [st.session_state.nama_fitur[j] for j in top_f_ind]
            weights = topic[top_f_ind]

            ax = axes_flat[i]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(axis_color)
            ax.spines['bottom'].set_color(axis_color)
            ax.xaxis.grid(True, linestyle='--', alpha=0.5, color=grid_color)
            ax.set_axisbelow(True)
            
            chart_color = colors_palette[i % len(colors_palette)]
            bars = ax.barh(top_f, weights, color=chart_color, edgecolor=None, height=0.6)
            
            try:
                ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8, color=bar_label_color)
            except AttributeError:
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, f' {width:.3f}', 
                            va='center', ha='left', fontsize=8, color=bar_label_color)
            
            judul_grafik = st.session_state.topic_labels[i]
            ax.set_title(f'Topik {i+1}: {judul_grafik}', fontweight='bold', fontsize=10, color=text_color, pad=10)
            ax.tick_params(axis='both', which='major', labelsize=8, colors=bar_label_color)
            ax.invert_yaxis()

        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.patch.set_alpha(0.0)
        for axis in axes_flat:
            axis.set_facecolor('none')

        plt.tight_layout()
        st.pyplot(fig)

    with tab_wordcloud:

        from wordcloud import WordCloud
        
        cols = 2
        rows = math.ceil(st.session_state.num_topics / cols)
        fig_wc, axes_wc = plt.subplots(rows, cols, figsize=(10, 3.2 * rows))
        axes_wc_flat = axes_wc.flatten() if st.session_state.num_topics > 1 else [axes_wc]
        

        wc_text_color = "#1e293b" if is_light else "#f1f5f9"
        
        for i, topic in enumerate(st.session_state.lda_components):
            top_f_ind = topic.argsort()[-30:]
            top_f = [st.session_state.nama_fitur[j] for j in top_f_ind]
            weights = topic[top_f_ind]
            

            word_freqs = dict(zip(top_f, weights))
            

            wc = WordCloud(
                background_color=None,
                mode="RGBA",
                colormap="Blues", 
                width=500,
                height=350,
                prefer_horizontal=0.8,
                random_state=42
            ).generate_from_frequencies(word_freqs)
            
            ax = axes_wc_flat[i]
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            
            judul_grafik = st.session_state.topic_labels[i]
            ax.set_title(f'Topik {i+1}: {judul_grafik}', fontweight='bold', fontsize=10, color=wc_text_color, pad=8)
            
        for j in range(i + 1, len(axes_wc_flat)):
            fig_wc.delaxes(axes_wc_flat[j])
            
        fig_wc.patch.set_alpha(0.0)
        for axis in axes_wc_flat:
            axis.set_facecolor('none')
            
        plt.tight_layout()
        st.pyplot(fig_wc)

    
    

    st.markdown("---")
    st.subheader("🤖 AI Strategic Insights (Beta)")


    all_topics = []
    for idx, topic in enumerate(st.session_state.lda_components):
        top_idx = topic.argsort()[-5:][::-1]
        words = [st.session_state.nama_fitur[i] for i in top_idx]
        judul_ai = st.session_state.topic_labels[idx]
        all_topics.append(f"Topik {idx+1} ({judul_ai}): {', '.join(words)}")
    
    topics_string = "\n".join(all_topics)

    if st.session_state.ai_insight:
        st.markdown(f"> {st.session_state.ai_insight}")
        st.warning("⚠️ **Rekomendasi Prioritas:** Fokus pada perbaikan yang paling sering muncul pada topik dengan bobot tertinggi terlebih dahulu. Jangan lupa cek juga apakah masalahnya lebih ke teknis, desain, atau gameplay untuk menentukan tim yang harus turun tangan.")
        

        if overwrite_analysis:
            if st.button("🔄 Regenerate Analisis Perbaikan"):
                with st.spinner("AI sedang membaca konteks game dan menyusun ulang saran strategis..."):
                    if is_ai_connected():
                        insight = generate_ai_insight(topics_string, st.session_state.review_type, st.session_state.app_id)
                        st.session_state.ai_insight = insight
                        

                        json_path = os.path.join(st.session_state.cache_dir, "analysis_data.json")
                        if os.path.exists(json_path):
                            with open(json_path, 'r', encoding='utf-8') as f:
                                cache_data = json.load(f)
                            cache_data["ai_insight"] = insight
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(cache_data, f, ensure_ascii=False, indent=4)
                        
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                    else:
                        st.error("Koneksi AI/API terputus. Tidak dapat melakukan regenerasi.")
    else:
        if st.button("✨ Generate Analisis Perbaikan"):
            with st.spinner("AI sedang membaca konteks game dan menyusun saran strategis..."):
                if is_ai_connected():
                    insight = generate_ai_insight(topics_string, st.session_state.review_type, st.session_state.app_id)
                    st.session_state.ai_insight = insight
                    

                    json_path = os.path.join(st.session_state.cache_dir, "analysis_data.json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                        cache_data["ai_insight"] = insight
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(cache_data, f, ensure_ascii=False, indent=4)
                    
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
                else:
                    st.error("Koneksi AI/API terputus. Silakan coba lagi ketika API terhubung.")

elif not btn_proses:
    st.info("👈 Silakan atur konfigurasi di sebelah kiri, lalu klik tombol 'Mulai Analisis' untuk memulai analisis baru.")
    
    st.markdown("---")
    st.subheader("📂 Riwayat Analisis Tersimpan")
    
    all_saved = get_all_saved_analyses()
    if all_saved:
        # Urutkan atau batasi jika perlu, tampilkan dalam grid 3 kolom
        cols = st.columns(3)
        for idx, opt in enumerate(all_saved):
            col_idx = idx % 3
            with cols[col_idx]:
                with st.container(border=True):
                    # Tampilkan gambar game
                    st.image(opt["game_image"], use_container_width=True)
                    
                    # Nama & App ID
                    st.markdown(f"##### **{opt['game_name']}**")
                    
                    # Tampilkan badge info
                    st.markdown(f"""
                    <div style="margin-top: 5px; margin-bottom: 10px; line-height: 1.8;">
                        <span class="stat-badge badge-id">ID: {opt['params']['app_id']}</span>
                        <span class="stat-badge badge-count">📝 {opt['total_reviews']} Ulasan</span>
                        <br/>
                        <span class="stat-badge badge-type">{opt['params']['review_type'].upper()}</span>
                        <span class="stat-badge badge-lang">{opt['params']['language'].title()}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tombol Muat Analisis
                    if st.button("📂 Muat Analisis", key=f"load_card_{opt['dir_name']}", use_container_width=True):
                        with st.spinner("Memuat analisis..."):
                            if load_saved_analysis_from_disk(opt["dir_name"], opt["params"]):
                                st.success(f"Analisis untuk {opt['game_name']} berhasil dimuat!")
                                if hasattr(st, "rerun"):
                                    st.rerun()
                                else:
                                    st.experimental_rerun()
                            else:
                                st.error("Gagal memuat analisis.")
    else:
        st.markdown("*Belum ada riwayat analisis yang tersimpan. Silakan lakukan analisis pertama Anda di sebelah kiri.*")