import json
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd


russian_stopwords = stopwords.words('russian')
def preprocess_text(text):
    stemmer = SnowballStemmer("russian")
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    tokens = word_tokenize(text, language='russian')
    tokens = [stemmer.stem(word) for word in tokens if word not in russian_stopwords]
    return ' '.join(tokens)


def get_article_content(article_url):
    response = requests.get(article_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text_blocks = soup.find_all('div', class_='block-text')
        if text_blocks:
            full_text = ' '.join(block.get_text(strip=True) for block in text_blocks)
            return preprocess_text(full_text)
        else:
            return None
    else:
        return None


def load_articles_from_csv(csv_file):
    df = pd.read_csv(csv_file, header=None)
    article_links = df[0].tolist()
    article_tokens = {}
    for link in article_links:
        tokens = get_article_content(link)
        if tokens:
            article_tokens[link] = tokens
    return article_tokens

# Обработка статей и сохранение токенов
def save_tokens_to_file(article_tokens, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(article_tokens, file, ensure_ascii=False, indent=4)

# Загрузка и обработка статей
articles = load_articles_from_csv("links.csv")
processed_articles = {link: preprocess_text(text) for link, text in articles.items()}

# Сохранение токенов в файл
save_tokens_to_file(processed_articles, "tokens.json")
