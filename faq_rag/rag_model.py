from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from decouple import config

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # UNSAFE!!!!
OPENAI_API_KEY = config('OPENAI_API_KEY')
BASE_URL = config('BASE_URL')


def load_data(path: str, big_chunk_size: int = 1000, big_chunk_overlap: int = 0,
              small_chunk_size: int = 1000, small_chunk_overlap: int = 0, ) -> [str]:
    """
    Load data from the directory and split it on chunks
    :param path:
    :param chunk_overlap:
    :param chunk_size:
    :return:
    """
    directory = os.fsencode(path + '/big')
    chunks = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            chunks.extend(chunkify_txt(path + '/big/' + filename, big_chunk_size, big_chunk_overlap, mode='big'))
        elif filename.endswith(".pdf"):
            chunks.extend(chunkify_pdf(path + '/big/' + filename, big_chunk_size, big_chunk_overlap, mode='big'))
        else:
            continue

    directory = os.fsencode(path + '/small')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            chunks.extend(
                chunkify_txt(path + '/small/' + filename, small_chunk_size, small_chunk_overlap, mode='small'))
        elif filename.endswith(".pdf"):
            chunks.extend(
                chunkify_pdf(path + '/small/' + filename, small_chunk_size, small_chunk_overlap, mode='small'))
        else:
            continue
    return chunks


def chunkify_pdf(filename: str, chunk_size: int = 1000, chunk_overlap: int = 0, mode: str = 'big') -> [str]:
    """
    Load PDF-file and splits it on chunks
    :param filename:
    :param chunk_overlap:
    :param chunk_size:
    :return:
    """
    if mode == 'big':
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n\n", "\n\n", "\n", " ", ""]
        )
    elif mode == 'small':
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    loader = PyPDFLoader(filename)
    pages = loader.load()

    chunks = text_splitter.split_documents(pages)
    return chunks


def chunkify_txt(filename: str, chunk_size: int = 1000, chunk_overlap: int = 0, mode: str = 'big') -> [str]:
    """
    Load TXT-file and splits it on chunks
    :param filename:
    :param chunk_overlap:
    :param chunk_size:
    :return:
    """
    if mode == 'big':
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n\n", "\n\n", "\n", " ", ""]
        )
    elif mode == 'small':
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    loader = TextLoader(filename, encoding="utf-8")
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    return chunks


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,
                              base_url=BASE_URL)

texts = load_data('C:/Users/Пользователь/OneDrive/Рабочий стол/FominAPI/FAQ', big_chunk_size=2000,
                  big_chunk_overlap=300, small_chunk_size=500, small_chunk_overlap=100)

# Создание векторной базы данных для хранения текстов и соответствующих им векторных представлений
# db = FAISS.from_documents(texts, embeddings)
# db.save_local('faiss_index')
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

""" Настройка ретривера (системы поиска по схожести векторных представлений документов)
   Здесь параметр k в search_kwargs отвечает за количество наиболее релевантных документов в выдаче"""
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Создаем llm_chain со встроенным Retrieval из langchain для удобства использования


# print(db.similarity_search_with_score('у меня будет операция в большой гинекологии, какие мне нужно сдать анализы?'))

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY,
                   base_url=BASE_URL),
    chain_type='stuff', retriever=retriever)

# qa = RetrievalQA.from_chain_type(
#     llm=model, chain_type="rag", retriever=retriever, return_source_documents=True)
#
# # Формулировка запроса и получение ответа на вопрос
# query = "Где мне искать информацию по инвентаризации?"
# result = qa({"query": query})
#
# query = 'как я могу связаться с клиникой фомина в Сочи?'
#
# print(qa_chain.run(query))
