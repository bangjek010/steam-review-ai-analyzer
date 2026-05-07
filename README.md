# 🎮 Steam Review AI Analyzer (SteamPulse)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![AI Models](https://img.shields.io/badge/Gemini_AI-2.5_Flash_%7C_3.1_Lite-orange?style=for-the-badge&logo=google&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Steam Review AI Analyzer** adalah dashboard analitik data *end-to-end* yang dirancang untuk pengembang game dan pengambil keputusan teknis. Aplikasi ini mengubah ribuan ulasan pemain yang tidak terstruktur menjadi **strategi bisnis berbasis data** dalam hitungan detik.

Alih-alih membaca feedback secara manual, aplikasi ini menggunakan **Natural Language Processing (NLP)** dan **Latent Dirichlet Allocation (LDA)** untuk menemukan topik tersembunyi dan bug yang sering muncul. Terakhir, integrasi **Google Gemini AI** akan menerjemahkan kata kunci teknis menjadi wawasan strategis yang mudah dipahami.

---

## ✨ Fitur Utama

*   **🕸️ Live Steam API Integration:** Mengambil metadata game (gambar & judul) secara otomatis dan melakukan *scraping* ulasan mentah menggunakan App ID.
*   **🧹 Robust NLP Pipeline:** Pembersihan teks, tokenisasi, dan lemmatization menggunakan `NLTK` untuk menyaring gangguan (noise).
*   **🧠 Unsupervised Machine Learning (LDA):** Mengelompokkan feedback pemain ke dalam topik-topik logis menggunakan TF-IDF Vectorizer dan algoritma LDA.
*   **🤖 Smart AI Categorization:** Menggunakan **Gemini AI** untuk memberikan judul kategori berbasis emoji yang relevan (misal: *🎮 Gameplay*, *🛠️ Server Bugs*).
*   **💡 AI Strategic Consultant:** Menghasilkan rencana aksi kontekstual dan analisis akar masalah berdasarkan topik yang diekstrak.
*   **🛡️ Multi-Model AI Fallback System:** Sistem cadangan otomatis. Jika model utama (`gemini-2.5-flash`) mencapai limit, sistem berpindah ke `gemini-3.1-flash-lite` untuk memastikan aplikasi tetap berjalan.
*   **📊 Modern UI:** Dashboard mode gelap yang interaktif dengan grafik Matplotlib, kartu metrik dinamis, dan fitur ekspor CSV.

---

## 🏗️ Arsitektur & Tech Stack

| Komponen | Teknologi |
| :--- | :--- |
| **Frontend UI** | Streamlit (Python) |
| **Data Manipulation** | Pandas, Numpy |
| **Web Scraping** | Steam API, `steamreviews`, `requests` |
| **NLP & ML** | NLTK, Scikit-Learn (TF-IDF, LDA) |
| **Visualization** | Matplotlib, Streamlit Metrics |
| **Generative AI** | Google Gemini (2.5 Flash & 3.1 Flash Lite) |

---

## 🚀 Cara Menjalankan Secara Lokal

1. **Clone Repository**
   ```bash
   git clone [https://github.com/dzakwan-rafi/steam-review-ai-analyzer.git](https://github.com/dzakwan-rafi/steam-review-ai-analyzer.git)
   cd steam-review-ai-analyzer

2. **Create a Virtual Environment (Recommended)**
   ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install Dependencies**
   ```bash
     pip install -r requirements.txt

4. **Setup Google Gemini API Key**
   This app requires a free Google Gemini API key to run the AI features.

  1. Get your API Key from Google AI Studio.
  2. In the root directory of this project, create a hidden folder named .streamlit.
  3. Inside that folder, create a file named secrets.toml.
  4. Add your key inside the file:
     
       ```bash
     GEMINI_API_KEY = "your_actual_api_key_here"

5. **Launch the App**
   ```bash
   streamlit run app.py
   
---

📸 Screenshots
<img width="1918" height="965" alt="image" src="https://github.com/user-attachments/assets/e9732f18-0e02-4a7c-beba-00b96a2f9724" />
<img width="1474" height="665" alt="image" src="https://github.com/user-attachments/assets/d4db9bbf-2969-4a51-b623-a1685065b448" />
<img width="1576" height="882" alt="image" src="https://github.com/user-attachments/assets/28b45941-f3ac-4b9c-bfc7-f6fa2721a9c3" />
<img width="1508" height="530" alt="image" src="https://github.com/user-attachments/assets/78f91ee5-e0d8-4a13-bc3f-40a8a0dbbb44" />

---

📝 License
Distributed under the MIT License. See LICENSE for more information.

---

## 👤 Connect with Me

Dibuat dengan ☕ oleh **Dzakwan Rafi**

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bangjek010)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/dzkwanrf)

---


