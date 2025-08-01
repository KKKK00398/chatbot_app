import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ベクトル保存ファイル名
VECTORSTORE_PATH = "faiss_index"

# ベクトルストアの再利用 or 作成
def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        # allow_dangerous_deserialization=True を追加
        return FAISS.load_local(VECTORSTORE_PATH, OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
    else:
        loader = TextLoader("docs/knowledge.txt", encoding="utf-8")
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_split = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(docs_split, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        return vectorstore

# Streamlit UI
st.title("📄 ドキュメントチャットボット")
query = st.text_input("質問を入力してください:")

if query:
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query)
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    st.write("🧠 回答:")
    st.write(response)
