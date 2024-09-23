from fastapi import FastAPI

from faq_rag.rag_model import qa_chain
from scripts.search_engine import ArticleSearch

app = FastAPI()
search_engine = ArticleSearch("C:/Users/Пользователь/OneDrive/Рабочий стол/FominAPI/tokens.json", 156)


@app.get("/health")
def get_article(question: str):
    answer = search_engine.find_articles(question)
    return {"answer": answer}


@app.get("/faq")
def get_article(question: str):
    answer = qa_chain.invoke(question)
    return {"answer": answer}
