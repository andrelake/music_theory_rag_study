import re
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def load_document(filepath: str) -> List:
    print(f'Loading {filepath}')
    loader = PyPDFLoader(filepath)
    data = loader.load()
    for page in data:
        page.page_content = re.sub(r' +', ' ', page.page_content)
        page.page_content = re.sub(r'(?: \n)+(?<! )', ' \n', page.page_content)
    print(f'Data Loaded Successfully. Total pages: {len(data)}')
    return data


def chunk_data(data: List, chunk_size: int = 4000) -> List:
    print(f'Starting chunk context')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=400
    )
    chunks = text_splitter.split_documents(data)
    return chunks
