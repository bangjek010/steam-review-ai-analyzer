# --- ai_helper.py ---
import google.generativeai as genai
import streamlit as st

def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

def generate_ai_insight(game_name, topics_list, review_type, app_id):
    API_KEY = get_api_key()
    if not API_KEY: return "Gagal: API Key tidak ada."
    
    genai.configure(api_key=API_KEY)
    
    # 🌟 SISTEM KASTA: Jika model atas limit (429), otomatis turun ke model bawahnya
    daftar_model = [
        'gemini-2.5-flash',       # Pintar, tapi limit harian cuma 20
        'gemini-3.1-flash-lite',  # Penyelamat! Limit harian 500
        'gemini-3-flash'          # Cadangan terakhir
    ]

    prompt = f"""
    Anda adalah seorang Lead Game Analyst. 
    Tugas Anda adalah menganalisis ulasan pemain untuk game: **{game_name}** (Steam App ID: {app_id}).
    Sentimen ulasan yang sedang Anda analisis saat ini adalah ulasan '{review_type}'. 

    Berikut adalah hasil klastering Machine Learning (Topik & Kata Kunci) dari ribuan ulasan pemain:
    {topics_list}

    INSTRUKSI PENTING:
    Hubungkan kata kunci di atas dengan konteks genre dan fitur asli dari game {game_name}. JANGAN berhalusinasi.

    Buatlah laporan analisis strategis dengan format markdown berikut:
    1. 🎯 **Ringkasan Sentimen {game_name}:** Jelaskan secara spesifik poin utama pemain.
    2. 💡 **3 Rekomendasi Prioritas untuk Developer:** Langkah konkret minggu ini.
    3. 📊 **Akar Masalah / Kekuatan Utama:** Analisis apakah dominan teknis, UI/UX, gameplay, atau monetisasi.
    
    Gunakan bahasa Indonesia yang profesional, tajam, namun mudah dipahami.
    """
    
    # Mencoba model satu per satu
    for nama_model in daftar_model:
        try:
            model = genai.GenerativeModel(nama_model)
            response = model.generate_content(prompt)
            return response.text # Jika berhasil, langsung kembalikan hasil
        except Exception as e:
            error_msg = str(e).lower()
            # Jika Error karena kuota habis (429), Lanjut coba model berikutnya!
            if "429" in error_msg or "quota" in error_msg:
                continue
            
    # Jika semua model di daftar sudah dicoba dan tetap gagal
    return "⚠️ Server AI sedang penuh. Silakan coba beberapa saat lagi."

def generate_topic_labels_with_ai(game_name, topics_dict, language):
    API_KEY = get_api_key()
    if not API_KEY: return [f"Topik {i+1}" for i in range(len(topics_dict))]
    
    genai.configure(api_key=API_KEY)
    
    # 🌟 SISTEM KASTA UNTUK LABEL TOPIK
    daftar_model = [
        'gemini-2.5-flash',
        'gemini-3.1-flash-lite'
    ]

    formatted_topics = "\n".join([f"Topik {k}: {', '.join(v)}" for k, v in topics_dict.items()])

    prompt = f"""
    Berikan judul kategori singkat (maks 3 kata) untuk Topik ulasan game "{game_name}".
    Bahasa: {language}.

    Kata Kunci:
    {formatted_topics}

    PILIH DARI KATEGORI BERIKUT: UI/UX, Gameplay, Server/Bug, Cerita, Audio, Komunitas, Visual, Monetisasi, Fitur Baru, atau Sentimen Umum.
    Gunakan 1 Emoji di awal.

    FORMAT BALASAN WAJIB SEPERTI INI (Tanpa teks tambahan):
    Topik 1: 🎮 Fitur Gameplay
    Topik 2: 🛠️ Masalah Server
    """

    # FUNGSI PEMOTONG TEKS (SUDAH DIPERBAIKI)
    def parse_labels(response_text):
        clean_text = response_text.replace("```text", "").replace("```", "").strip()
        lines = clean_text.split('\n')
        labels = []
        for line in lines:
            # SEKARANG MENCARI TITIK DUA (:), BUKAN GARIS (|)
            if "Topik" in line and ":" in line:
                nama_topik = line.split(':', 1)[1].strip()
                labels.append(nama_topik)
        return labels

    # Mencoba model satu per satu
    for nama_model in daftar_model:
        try:
            model = genai.GenerativeModel(nama_model)
            response = model.generate_content(prompt)
            labels = parse_labels(response.text)
            
            # Pastikan jumlah label sama dengan jumlah topik
            if len(labels) == len(topics_dict):
                return labels
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg:
                continue # Lanjut ke model berikutnya jika limit
                
    # Jika gagal total, kembalikan ke nama default (Topik 1, Topik 2)
    return [f"Topik {i+1}" for i in range(len(topics_dict))]
