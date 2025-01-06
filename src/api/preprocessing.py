import pandas as pd
import numpy as np
import string
import re
import nltk
import swifter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Download stopwords untuk Bahasa Indonesia dari nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')


def clean_text(text):
    # Menghilangkan tanda baca dan karakter khusus
    text = re.sub(r'ï', 'i', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\b(?:\d+|[^\w\s])\b', ' ', text)
    text = re.sub(r'[\[\]\{\}\_\-\=\"]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip()
    return text


singkatan_dict = {
    " ai ": " artificial intelligence ",
    " ml ": " machine learning ",
    " dl ": " deep learning ",
    " ui ": " user interface ",
    " ux ": " user experience ",
    " nlp ": " pengolahan bahasa alami ",
    " api ": " antarmuka pemrograman aplikasi",
    " data science ": " ilmu data ",
    " database ": " basis data ",
    "pengembangan": " perancangan ",
    "software": "perangkat lunak"
}


def casefolding_replace(text):
    text = text.lower()
    for singkatan, full_form in singkatan_dict.items():
        text = text.replace(singkatan, full_form)
        text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    return nltk.word_tokenize(text)


list_stopwords_id = stopwords.words('indonesian')
list_stopwords = set(list_stopwords_id)
list_stopwords.update(['mahasiswa', 'siswa', 'peserta', 'anda', 'chapter', 'terkait', 'kesiapan', 'belajar', 'mempelajari', 'mengerti', 'meliputi',
                      'tentang', 'batasan', 'menguasai', 'konsep', 'campus', 'bangkit', 'academy', 'path', 'tugas', 'berbasis', 'skill', 'kritis', 'detail'])

translation_dict = {
    "mining": "penambangan",
    "machine": "mesin",
    "deep": "mendalam",
    "learning": "pemelajaran",
    "computer": "komputer",
    "vision": "visi",
    "natural": "alami",
    "language": "bahasa",
    "processing": "pengolahan",
    "business": "bisnis",
    "artificial": "buatan",
    "intelligence": "kecerdasan",
    "expert": "pakar",
    "system": "sistem",
    "cloud": "awan",
    "computing": "komputasi",
    "grid": "jaringan",
    "security": "keamanan",
    "cyber": "siber",
    "interface": "antarmuka",
    "design": "desain",
    "code": "kode",
    "testing": "pengujian",
    "semantic": "semantik",
    "process": "proses",
    "pemrosesan": "proses",
    "service": "layanan"
}


def filter_and_translate(tokens):
    filtered_tokens = [word for word in tokens if word not in list_stopwords]
    translated_tokens = []
    for token in filtered_tokens:
        if token in translation_dict:
            translated_tokens.append(translation_dict[token])
        else:
            translated_tokens.append(token)
    return translated_tokens


# Buat Stemmer dan Fungsi untuk Stemming Indo
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def stemmed_wrapper(term):
    stemmed = stemmer.stem(term)
    return stemmed


def build_term_dict(data):
    """Build a dictionary of terms from the data"""
    term_dict = {}
    # Handle if data is a string (column name) or list of tokens
    if isinstance(data, str):
        # If data is a column name, create empty dict
        term_dict = {}
    else:
        # If data is list of documents, build term dict
        for document in data:
            for term in document:
                if term not in term_dict:
                    term_dict[term] = ' '
    return term_dict


def apply_stemming(data, term_dict):
    """Apply stemming to documents using term dictionary"""
    # Update term dictionary with stemmed terms
    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)

    # Apply stemming to each document
    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    return data.apply(get_stemmed_term)


lemmatizer = WordNetLemmatizer()


def lemmatize_text(tokens):
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


list_stopwords.update(['mata', 'kuliah', 'paham', 'papar', 'erti', 'selesai', 'jelas', 'implementasi', 'kenal', 'milik', 'sesuai', 'nya', 'evaluasi', 'pilar', 'versi', 'butuh', 'beda', 'ambil', 'perhati', 'isu', 'bekal', 'fondasi', 'nilai', 'suasana', 'cakup', 'demonstrasi', 'dampak', 'bata',
                       'bagaimana', 'lain', 'strategis', 'bahas', 'topik', 'pilih', 'terap', 'bantu', 'kelola', 'final', 'awas', 'laksana', 'materi', 'terampil', 'manfaat', 'peran', 'langkah', 'bidang', 'atur', 'jalan', 'terima', 'jenis', 'tipe', 'ak', 'hidup', 'kuat', 'lengkap', 'cari', 'komunikasi', 'dunia', 'solid', 'sederhana', 'e'
                       'teori', 'artikel', 'karakteristik', 'hadap', 'definisi', 'luas', 'khusus', 'hubung', 'buruk', 'isi', 'identifikasi', 'organisasi', 'solusi', 'tuju', 'studio', 'aktivitas', 'nyata', 'pecah', 'ruang', 'lingkup', 'utama', 'hasil', 'hitung', 'fundamental', 
                       'studi', 'prinsip', 'aspek', 'gagal', 'kait', 'ulas', 'teliti', 'urai', 'ilmu', 'tahap', 'ranah', 'tanggap', 'capai', 'tingkat', 'parah', 'arah', 'mekanisme', 'kendali', 'lancar', 'konseptual', 'konvensional'])

new_dict = {
    "alam": "alami",
    "prose": "proses",
    "analis": "analisis",
    "lingkung": "lingkungan",
    "jaring": "jaringan",
    "layan": "layanan",
    "bangun": "rancang",
    "ancang": "rancang",
    "pemrograman": "program",
    "putus": "keputusan",
    "responsif": "respons",
    "komputasional": "komputasi",
    "fungsional": "fungsi",
    "operasional": "operasi"
}


def ubah(tokens):
    ubah_tokens = []
    for token in tokens:
        if token in new_dict:
            ubah_tokens.append(new_dict[token])
        else:
            ubah_tokens.append(token)
    return ubah_tokens


def filter_stopwords(text):
    text = [word for word in text if word not in list_stopwords]
    text = [word for word in text if word != '']
    return text
