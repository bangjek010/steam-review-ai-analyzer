import google.generativeai as genai
import streamlit as st
import os

def get_api_key():
    try:
        # Coba ambil dari Streamlit secrets
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    
    # Coba ambil dari Environment Variable
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key
        
    return None

def generate_ai_insight(topics_list, review_type, app_id):
    API_KEY = get_api_key()
    if not API_KEY: 
        return "Gagal memuat AI Insight: API Key tidak ditemukan. Silakan konfigurasi GEMINI_API_KEY di Streamlit Secrets."

        
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    Anda adalah seorang Analis Game Profesional. 
    Saat ini Anda sedang menganalisis ulasan untuk sebuah game di Steam. 
    Referensi URL game tersebut adalah: https://store.steampowered.com/app/{app_id}

    Berdasarkan hasil ekstraksi topik dari ulasan yang bertipe '{review_type}', 
    berikut adalah topik dan kata kunci utama yang sedang ramai dibicarakan pemain:
    {topics_list}

    INSTRUKSI PENTING:
    1. Coba kenali judul game ini berdasarkan URL dan kata kunci ulasan di atas (contoh: jika kata kuncinya seputar 'tactic', 'football', 'manager', ini pasti Football Manager). 
    2. JANGAN MENGARANG (halusinasi) nama game jika Anda tidak yakin 100%. Jika ragu, cukup sebutkan genre game-nya saja berdasarkan kata kunci.
    
    Tolong berikan laporan analisis dengan format berikut:
    1. 🎯 **Ringkasan Sentimen:** Sebutkan nama gamenya (jika tahu), lalu jelaskan apa yang sebenarnya dirasakan pemain.
    2. 💡 **3 Saran Konkret untuk Developer (Improvement Priority):** Apa yang harus diperbaiki minggu ini juga berdasarkan topik di atas?
    3. 📊 **Akar Masalah:** Analisis apakah ini murni masalah teknis (bug/crash), desain UI/UX, fundamental gameplay, atau kebijakan monetisasi.
    
    Gunakan bahasa Indonesia yang profesional, tajam, namun tetap santai.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gagal memuat AI Insight: {e}"


# --- FUNGSI BARU UNTUK MEMBERI NAMA TOPIK ---
def generate_topic_labels_with_ai(topics_dict, language):
    API_KEY = get_api_key()
    if not API_KEY:
        return [f"Topik {i+1}" for i in range(len(topics_dict))]
        
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Format kata kunci agar mudah dibaca AI
    formatted_topics = "\n".join([f"Topik {k}: {', '.join(v)}" for k, v in topics_dict.items()])

    prompt = f"""
    Act as a Professional Data Analyst. I have LDA Topic Modeling results from game reviews.
    The reviews are in this language: {language}.

    Here are the topics and their top keywords:
    {formatted_topics}

    Task: Provide a very short, specific category title (max 4-5 words) for each topic based on the keywords. 
    Include 1 relevant emoji at the beginning of the title.
    If the language is 'indonesian', write the titles in Indonesian. If 'english', write in English.

    OUTPUT FORMAT STRICTLY LIKE THIS (Do not add any other text):
    1| 🛠️ Title Here
    2| 🎨 Title Here
    3| ⚽ Title Here
    """

    try:
        response = model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        
        labels = []
        for line in lines:
            if '|' in line:
                # Mengambil teks setelah tanda '|'
                labels.append(line.split('|')[1].strip())
                
        # Jika AI gagal mengikuti format, kembalikan nama default
        if len(labels) != len(topics_dict):
            return [f"Topik {i+1}" for i in range(len(topics_dict))]
            
        return labels
    except Exception as e:
        # Fallback jika error API
        return [f"Topik {i+1}" for i in range(len(topics_dict))]

def is_ai_connected():
    API_KEY = get_api_key()
    if not API_KEY:
        return False
    try:
        genai.configure(api_key=API_KEY)
        # Check connection by listing models (very fast and doesn't consume generation tokens)
        next(iter(genai.list_models()))
        return True
    except Exception:
        return False
