import argparse
import shutil
import json
import os

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from get_embedding_function import get_embedding_function
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

CHROMA_PATH = "./chroma"
DATA_PATH = "./data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    chunks = calculate_chunk_ids(chunks)  # Ensure IDs are assigned
    add_to_chroma(chunks)

def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        
        # Case 1: txt file
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                qa_pairs = content.split('\n\n')  # Assuming each Q&A pair is separated by two newlines
                for qa_pair in qa_pairs:
                    if qa_pair.strip():  # Avoid empty Q&A pairs
                        # Each Q&A pair is a document
                        documents.append(Document(page_content=qa_pair.strip(), metadata={"source": filename}))

        # Case 2: jsonl file
        if filename.endswith('.jsonl'):
            # Read the JSON Lines file and process each line
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip(): # ensure the line is not empty
                        json_line = json.loads(line.strip())
                        prompt = json_line.get('prompt', '')
                        completion = json_line.get('completion', '')
                        content = prompt + " " + completion
                        documents.append(Document(page_content=content, metadata={"source": filename}))

        # Implement other cases if possible/necessary

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False)
    chunks = []
    for doc in documents:
        # Ensure to use the correct attribute 'page_content'
        doc_chunks = text_splitter.split_text(doc.page_content)
        for chunk in doc_chunks:
            chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return chunks


def add_to_chroma(chunks):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    existing_items = db.get(include=[])  # This returns a dictionary with different categories

    # Directly access the 'ids' key, which contains a list of existing IDs
    existing_ids = set(existing_items['ids']) if existing_items.get('ids') is not None else set()

    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]
    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        batch_size = 166
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            db.add_documents(batch)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        current_page_id = f"{chunk.metadata['source']}:{chunk.metadata.get('page', 'N/A')}" # the page number is for PDFs
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            last_page_id = current_page_id
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
