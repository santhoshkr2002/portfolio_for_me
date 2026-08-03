from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCUMENT_FOLDER = "documents"
CHROMA_DB_PATH = "chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = []

pdf_files = list(Path(DOCUMENT_FOLDER).glob("*.pdf"))

if not pdf_files:
    raise Exception("No PDF files found inside the documents folder.")

for pdf in pdf_files:

    print(f"Processing: {pdf.name}")

    loader = PyPDFLoader(str(pdf))

    pages = loader.load()

    chunks = splitter.split_documents(pages)

    for index, chunk in enumerate(chunks):

        chunk.metadata.update({
            "source": pdf.name,
            "chunk": index
        })

    documents.extend(chunks)

print(f"Total chunks: {len(documents)}")

Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=CHROMA_DB_PATH
)

print("Chroma database created successfully.")