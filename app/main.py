from fastapi import FastAPI

from scripts.search_engine import ArticleSearch

app = FastAPI()
search_engine = ArticleSearch("C:/Users/Пользователь/OneDrive/Рабочий стол/FominAPI/tokens.json")

@app.get("/health")
def get_article(question: str):
    answer = search_engine.find_articles(question)
    return {"answer": answer}


@app.get("/faq")
def get_article(question: str):
    return {"answer": "I'm alive! You asked: " + question}
