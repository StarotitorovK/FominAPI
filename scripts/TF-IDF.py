import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


def load_tokens_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return json.load(file)


russian_stopwords = stopwords.words('russian')


processed_articles = load_tokens_from_file("tokens.json")
article_links = list(processed_articles.keys())
article_texts = list(processed_articles.values())


vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=russian_stopwords)
X = vectorizer.fit_transform(article_texts)


user_query = ""


def preprocess_text(text):
    stemmer = SnowballStemmer("russian")
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    tokens = word_tokenize(text, language='russian')
    tokens = [stemmer.stem(word) for word in tokens if word not in russian_stopwords]
    return ' '.join(tokens)

processed_query = preprocess_text(user_query)

query_vector = vectorizer.transform([processed_query])
n_components = 156
lsa = TruncatedSVD(n_components=n_components)
X_lsa = lsa.fit_transform(X)
query_vector_lsa = lsa.transform(query_vector)


similarities_lsa = cosine_similarity(query_vector_lsa, X_lsa)


for idx, similarity in enumerate(similarities_lsa[0]):
    if similarity > 0.4:
        print(f"Статья {article_links[idx]} с совпадением {similarity:.4f}")
