import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Setup NLTK (Hanya dipanggil saat file ini di-import)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Import custom stopwords jika ada
try:
    from daftar_stopwords import custom_stopwords
except ImportError:
    custom_stopwords = set()

def get_stopwords():
    return set(stopwords.words('english')).union(custom_stopwords)

def clean_text(df):
    stop_words = get_stopwords()
    lemmatizer = WordNetLemmatizer()
    
    def process(teks):
        if pd.isna(teks): return ""
        teks = str(teks).lower()
        teks = re.sub(r'http\S+|www\S+|https\S+', '', teks)
        teks = re.sub(r'[^a-z\s]', '', teks)
        tokens = word_tokenize(teks)
        return " ".join([lemmatizer.lemmatize(k) for k in tokens if lemmatizer.lemmatize(k) not in stop_words])

    df['Cleaned_Review'] = df['Review_Text'].apply(process)
    return df

def run_topic_modeling(df_clean, num_topics, min_df_val, max_df_val):
    # Proses TF-IDF
    tfidf_vec = TfidfVectorizer(max_features=1000, min_df=min_df_val, max_df=max_df_val)
    tfidf_matrix = tfidf_vec.fit_transform(df_clean['Cleaned_Review'].fillna(''))
    nama_fitur = tfidf_vec.get_feature_names_out()
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=nama_fitur)
    
    # Proses LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(df_tfidf)
    
    return lda.components_, nama_fitur