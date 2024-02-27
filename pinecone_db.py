from typing import List

from langchain_community.vectorstores import Pinecone as PineconeVectorstore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from data_handler import chunk_data, load_document
from env import PINECONE_API_KEY, PINECONE_ENVIRONMENT, FILE_PATH, OPENAI_API_KEY, PINECONE_INDEX_NAME
from llm_client import get_encoding


def get_client(pinecone_api_key) -> Pinecone:
    return Pinecone(api_key=pinecone_api_key)


def get_vectorstore(index_name, encoding) -> Pinecone:
    print(f'Getting info from index: {index_name}')
    return PineconeVectorstore.from_existing_index(index_name, encoding)


def create_index(chunks: List, encoding: OpenAIEmbeddings,
                 index_name: str, pinecone: Pinecone) -> PineconeVectorstore:
    print(f'Creating index: {index_name}')
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
    )
    print(f'Index created : {index_name}')
    return PineconeVectorstore.from_documents(chunks, encoding, index_name=index_name)


def delete_pinecone_index(index_name: str, pinecone: Pinecone = get_client(PINECONE_API_KEY)) -> None:
    indexes = pinecone.list_indexes()
    if index_name in indexes.names():
        pinecone.delete_index(index_name)
        print(f'Index Deleted Successfully.')


def get_retriever(vectorstore) -> VectorStoreRetriever:
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


if __name__ == '__main__':
    pinecone_client = get_client(PINECONE_API_KEY)
    chunks = chunk_data(load_document(FILE_PATH))
    encoding = get_encoding(OPENAI_API_KEY)
    create_index(chunks, encoding,  PINECONE_INDEX_NAME, pinecone_client)
