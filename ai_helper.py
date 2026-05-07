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

def generate_ai_insight(topics_list, review_type, app_id):
    API_KEY = get_api_key()
    if not API_KEY: return "Gagal: API Key tidak ada."
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    Anda adalah seorang Analis Game Profesional. 
    Menganalisis game di Steam: https://store.steampowered.com/app/{app_id}
    Ulasan tipe: '{review_type}'. 
    Topik & Kata kunci: {topics_list}

    Berikan laporan analisis dengan format markdown:
    1. 🎯 **Ringkasan Sentimen:** Sebutkan nama gamenya (jika tahu) & perasaan pemain.
    2. 💡 **3 Saran Konkret untuk Developer:** Prioritas perbaikan.
    3. 📊 **Akar Masalah:** Analisis murni (teknis/desain/UI-UX/monetisasi).
    Gunakan bahasa Indonesia profesional dan menarik.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gagal memuat AI Insight: {e}"

def generate_topic_labels_with_ai(topics_dict, language):
    API_KEY = get_api_key()
    if not API_KEY: return [f"Topik {i+1}" for i in range(len(topics_dict))]
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')

    formatted_topics = "\n".join([f"Topik {k}: {', '.join(v)}" for k, v in topics_dict.items()])

    prompt = f"""
    Act as a Data Analyst. Topic Modeling results in {language}.
    Keywords:
    {formatted_topics}

    Task: Provide a short, specific category title (max 4 words) with 1 relevant emoji at the beginning.
    OUTPUT FORMAT STRICTLY LIKE THIS:
    1| 🛠️ Title Here
    2| 🎨 Title Here
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
