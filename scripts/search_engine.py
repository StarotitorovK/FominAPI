import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


class ArticleSearch:
    def __init__(self, filename: str, n_components: int):
        self.processed_articles = load_tokens_from_file(filename)
        self.russian_stopwords = stopwords.words('russian')
        self.vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=self.russian_stopwords)
        self.vectorized_articles = self.vectorizer.fit_transform(list(self.processed_articles.values()))
        self.lsa_processor = TruncatedSVD(n_components=n_components)
        self.lsa_processed_articles = self.lsa_processor.fit_transform(self.vectorized_articles)

    def find_articles(self, user_query):
        answer = []
        processed_query = preprocess_text(user_query, self.russian_stopwords)

        query_vector = self.vectorizer.transform([processed_query])
        query_vector_lsa = self.lsa_processor.transform(query_vector)
        similarities = cosine_similarity(query_vector_lsa, self.lsa_processed_articles)

        for idx, similarity in enumerate(similarities[0]):
            if similarity > 0.4:
                answer.append(list(self.processed_articles.keys())[idx])
        print(answer)
        return answer


def load_tokens_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return json.load(file)


# processed_articles = load_tokens_from_file("../tokens.json")
# article_links = list(processed_articles.keys())
# article_texts = list(processed_articles.values())
#
# vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=russian_stopwords)
# X = vectorizer.fit_transform(article_texts)

# user_query = ""


def preprocess_text(text, russian_stopwords):
    stemmer = SnowballStemmer("russian")
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    tokens = word_tokenize(text, language='russian')
    tokens = [stemmer.stem(word) for word in tokens if word not in russian_stopwords]
    return ' '.join(tokens)

#
# processed_query = preprocess_text(user_query)
#
# query_vector = vectorizer.transform([processed_query])
#
# similarities = cosine_similarity(query_vector, X)
#
# print("Статьи с высоким уровнем совпадения (> 0.08):")
# for idx, similarity in enumerate(similarities[0]):
#     if similarity > 0.08:
#         print(f"Статья {article_links[idx]} с совпадением {similarity:.4f}")
