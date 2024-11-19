from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"  # Path to the folder containing your PDF files

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    documents = []
    # Iterate through each file in the DATA_PATH directory
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):  # Only process PDF files
            pdf_path = os.path.join(DATA_PATH, filename)
            loader = PyPDFLoader(pdf_path)
            document = loader.load()
            documents.extend(document)
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print(f"Creating Chroma DB at: {os.path.abspath(CHROMA_PATH)}")

    # Create a new DB from the documents
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
