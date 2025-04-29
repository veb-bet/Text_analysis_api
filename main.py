import string
from collections import Counter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.corpus import stopwords

# Загружаем ресурсы для анализа текста (один раз при запуске)
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Инициализация FastAPI
app = FastAPI()

# Входная модель
class TextInput(BaseModel):
    text: str

# Фильтрация стоп-слов
def filter_stopwords(text: str, language: str):
    try:
        stop_words = stopwords.words(language)
    except:
        stop_words = []
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

# Основная функция анализа
def analyze_text(text: str):
    try:
        language = detect(text)
    except:
        language = "en"  # fallback

    text_lower = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text_lower.translate(translator)
    clean_text = filter_stopwords(clean_text, language)
    words = clean_text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(5)

    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)

    return {
        "language": language,
        "word_counts": word_counts,
        "most_common_words": most_common_words,
        "sentiment": sentiment
    }

# Роут анализа
@app.post("/analyze_text/")
def analyze(text_input: TextInput):
    if not text_input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = analyze_text(text_input.text)
    return result
