from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from env import OPENAI_MODEL_NAME, OPENAI_API_KEY, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN


def get_encoding(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key)


def ask_something(question: str, retriever: VectorStoreRetriever, session_id: str):
    # prompt
    prompt = build_chat_prompt_template()

    # chat_history
    chat_history = UpstashRedisChatMessageHistory(
        url=UPSTASH_REDIS_REST_URL,
        token=UPSTASH_REDIS_REST_TOKEN,
        session_id=session_id
    )

    # memory
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        input_key='input',
        output_key='output',
        chat_memory=chat_history,
        return_messages=True)

    # tool
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_about_music_theory",
        description="Searches and returns content about music theory. "
                    "For any question about music theory, "
                    "please use the search_about_music_theory tool.",
    )

    # tool list
    tools = [retriever_tool]

    # llm
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                     model_name=OPENAI_MODEL_NAME,
                     temperature=0.5,
                     streaming=True,
                     callbacks=[StreamingStdOutCallbackHandler()])

    # agent
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # agent_executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

    resp = agent_executor.invoke({"input": question, "chat_history": chat_history,
                                  'context': retriever.get_relevant_documents(question)})
    return resp.get('output')


def build_chat_prompt_template() -> ChatPromptTemplate:
    system_template = (
        """
        You are a helpful music teacher assistant that helps a student.
        You can only use Portuguese, from Brazil, to read and answer all the questions.
        You should only using the agent tool if asked about something related to music theory.
        You also can use the following context and chat history to answer the questions. 
        Combine chat history and the last user's question to create a new independent question. 
        Context: {context}
        Chat history: {chat_history} 
        Question: {input}
        """
    )

    return ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'chat_history', 'input'],
                                                           template=system_template)),
         MessagesPlaceholder(variable_name='chat_history', optional=True),
         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
         MessagesPlaceholder(variable_name='agent_scratchpad')]
    )
