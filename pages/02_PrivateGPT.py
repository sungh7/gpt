from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables import RunnableLambda
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title='PrivateGPT',
    page_icon="📃"
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

# memory = ConversationSummaryBufferMemory(
# #     llm=llm,
# #     max_token_limit=100,
# #     memory_key="history",
# # )


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f'./.cache/private/files/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/private/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator='\n',
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(splitter)
    embeddings = OllamaEmbeddings(
        model="mistral:latest"
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"role": role, "content": message})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["content"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        Answer the question using ONLY the following context and history. If you don't know the answer 
        just say you don't know. DON'T make anything up.
        
        Context: {context}
  
        """
        ),
        ("human", "{question}")
    ]
)


# def load_memory(_):
#     return memory.load_memory_variables({})['history']


st.title("PrivateGPT")

st.markdown("""
            Welcome!
            Use this private chatbot to ask questions to an AI about your files!        
            Upload your files on the sidebar.
            """
            )


with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=['pdf', 'txt', 'docx']
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", role="ai", save=False)
    paint_history()
    message = st.chat_input("Ask me anything about your file!")
    if message:
        send_message(message, role="human")
        chain = (
            {
                # "history": RunnablePassthrough.assign(history=load_memory),
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            response = chain.invoke(message)
            # memory.save_context(
            #     {"input": message},
            #     {"output": response.content})


else:
    st.session_state["messages"] = []
