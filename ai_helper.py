# --- ai_helper.py ---
import google.generativeai as genai
import streamlit as st

def get_api_key():
    # Mengambil API key dari Streamlit Secrets (aman)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error("API Key tidak ditemukan. Pastikan sudah mengatur secrets di Streamlit.")
        return None

# Menerima tambahan parameter 'game_name'
def generate_ai_insight(game_name, topics_list, review_type, app_id):
    API_KEY = get_api_key()
    if not API_KEY: return "Gagal: API Key tidak ada."
    
    genai.configure(api_key=API_KEY)
    # Menggunakan Gemini 1.5 Flash (Model terbaru & lebih cerdas dari 2.5 flash yang belum stabil)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    Anda adalah seorang Lead Game Analyst. 
    Tugas Anda adalah menganalisis ulasan pemain untuk game: **{game_name}** (Steam App ID: {app_id}).
    Sentimen ulasan yang sedang Anda analisis saat ini adalah ulasan '{review_type}'. 

    Berikut adalah hasil klastering Machine Learning (Topik & Kata Kunci) dari ribuan ulasan pemain:
    {topics_list}

    INSTRUKSI PENTING:
    Karena Anda sudah tahu ini adalah game {game_name}, hubungkan kata kunci di atas dengan konteks genre dan fitur asli dari game ini. JANGAN berhalusinasi menyebutkan fitur yang tidak ada di dalam game ini.

    Buatlah laporan analisis strategis dengan format markdown berikut:
    1. 🎯 **Ringkasan Sentimen {game_name}:** Jelaskan secara spesifik apa poin utama yang difokuskan pemain dari kata kunci di atas.
    2. 💡 **3 Rekomendasi Prioritas untuk Developer:** Langkah konkret apa yang harus dilakukan tim dev minggu ini untuk merespons topik tersebut?
    3. 📊 **Akar Masalah / Kekuatan Utama:** Analisis apakah topik di atas dominan ke masalah teknis (bug/crash/server), desain UI/UX, fundamental gameplay, atau kebijakan monetisasi.
    
    Gunakan bahasa Indonesia yang profesional, tajam, namun mudah dipahami (actionable).
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gagal memuat AI Insight: {e}"

# Menerima tambahan parameter 'game_name'
def generate_topic_labels_with_ai(game_name, topics_dict, language):
    API_KEY = get_api_key()
    if not API_KEY: return [f"Topik {i+1}" for i in range(len(topics_dict))]
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')

    formatted_topics = "\n".join([f"Topik {k}: {', '.join(v)}" for k, v in topics_dict.items()])

    prompt = f"""
    Act as a Game Data Analyst. You are analyzing Topic Modeling results from player reviews for the game: "{game_name}".
    The keywords are in this language: {language}.

    Here are the topics and their top keywords:
    {formatted_topics}

    Task: Provide a short, specific, and highly relevant category title (max 4 words) for each topic based on the keywords AND the context of the game "{game_name}". 
    Include 1 relevant emoji at the beginning of the title.
    If language is 'indonesian', write titles in Indonesian. If 'english', write in English.

    OUTPUT FORMAT STRICTLY LIKE THIS EXAMPLE (Do not add any other text):
    Topik 1: 🎮 Fitur Gameplay
    Topik 2: 🛠️ Masalah Server
    """

    try:
        response = model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        
        labels = [line.split('|')[1].strip() for line in lines if '|' in line]
        if len(labels) != len(topics_dict):
            return [f"Topik {i+1}" for i in range(len(topics_dict))]
        return labels
    except Exception:
        return [f"Topik {i+1}" for i in range(len(topics_dict))]
