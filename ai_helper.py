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
    if not API_KEY: return "API_ERROR"
    genai.configure(api_key=API_KEY)
    
    # 🌟 SISTEM ANTRIAN MODEL (Urutan dari yang Paling Pintar -> Paling Aman)
    daftar_model = [
        'gemini-2.5-flash',  
        'gemini-1.5-pro',       # Kasta 1: Paling pintar, tapi rawan limit
        'gemini-1.5-flash',     # Kasta 3: Sangat stabil, limit harian besar (1500x)
        'gemini-1.5-flash-8b'   # Kasta 4: Paling ringan, anti limit
    ]

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
    
    # Looping mencoba model satu per satu
    for nama_model in daftar_model:
        try:
            model = genai.GenerativeModel(nama_model)
            response = model.generate_content(prompt)
            return response.text # Jika sukses, langsung stop dan kembalikan hasil
        except Exception as e:
            error_msg = str(e).lower()
            # Jika Error Limit (429), CONTINUE (Lanjut ke model berikutnya di daftar_model)
            if "429" in error_msg or "quota" in error_msg:
                continue
            else:
                # Jika error karena masalah lain (misal internet mati), stop!
                return "GENERAL_ERROR"
    
    # Jika SEMUA model di dalam daftar sudah habis terkena limit:
    return "LIMIT_ERROR"

def generate_topic_labels_with_ai(game_name, topics_dict, language):
    API_KEY = get_api_key()
    if not API_KEY: return [f"Topik {i+1}" for i in range(len(topics_dict))]
    genai.configure(api_key=API_KEY)
    
    # 🌟 SISTEM ANTRIAN MODEL
    daftar_model = [
        'gemini-2.5-flash',
        'gemini-1.5-pro',       # Kasta 1: Paling pintar, tapi rawan limit
        'gemini-1.5-flash',     # Kasta 3: Sangat stabil, limit harian besar (1500x)
        'gemini-1.5-flash-8b'   # Kasta 4: Paling ringan, anti limit
    ]
    
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

    def parse_labels(response_text):
        clean_text = response_text.replace("```text", "").replace("```", "").strip()
        lines = clean_text.split('\n')
        labels = []
        for line in lines:
            if "Topik" in line and ":" in line:
                nama_topik = line.split(':', 1)[1].strip()
                labels.append(nama_topik)
        return labels

    # Looping mencoba model satu per satu
    for nama_model in daftar_model:
        try:
            model = genai.GenerativeModel(nama_model)
            response = model.generate_content(prompt)
            labels = parse_labels(response.text)
            
            if len(labels) == len(topics_dict):
                return labels
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg:
                continue # Coba model berikutnya
            
    # Jika semua gagal/limit, kembalikan ke default
    return [f"Topik {i+1}" for i in range(len(topics_dict))]
