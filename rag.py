from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DB_PATH = "chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 4
    }
)


def retrieve_documents(question: str):
    """
    Retrieve the top matching document chunks.
    """
    return retriever.invoke(question)