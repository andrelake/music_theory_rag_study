from time import perf_counter, sleep

from env import PINECONE_INDEX_NAME, OPENAI_API_KEY
from llm_client import get_encoding, ask_something
from pinecone_db import get_vectorstore, get_retriever


def chat_handler(retriever, session_id):
    i = 1
    print("Digite Sair ou Fim para encerrar.")
    while True:
        question = input(f"\nPergunta #{i}: ")
        if question.lower() in ["sair", "fim"]:
            print("Tchau!")
            sleep(2)
            break

        time_start = perf_counter()
        ask_something(question, retriever, session_id)
        time_end = perf_counter()
        print(f"\nTime taken: {time_end - time_start:.2f} seconds")
        i += 1


def main():
    encoding = get_encoding(OPENAI_API_KEY)
    vs = get_vectorstore(PINECONE_INDEX_NAME, encoding)
    retriever = get_retriever(vs)
    session_id = 'session_music1'

    chat_handler(retriever, session_id)


if __name__ == '__main__':
    main()
