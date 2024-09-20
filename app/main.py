from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def get_article(question: str):
    return {"answer": "I'm alive! You asked: " + question}


@app.get("/faq")
def get_article(question: str):
    return {"answer": "I'm alive! You asked: " + question}
