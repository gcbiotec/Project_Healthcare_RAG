import os
import streamlit as st

#import rag_engine
#st.write("Import rag_engine OK")

from config import DATA_DIR
from rag_engine import RAGEngine

st.set_page_config(page_title="Chat RAG com Arquivos", layout="wide")

st.title("Chat com RAG baseado em arquivos")
st.caption("Faça perguntas com base nos documentos carregados na base.")

if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

rag = st.session_state.rag

with st.sidebar:
    st.header("Base de conhecimento")

    st.write(f"Pasta monitorada: `{DATA_DIR}`")

    if st.button("Gerar/Regerar índice"):
        with st.spinner("Lendo arquivos e gerando embeddings..."):
            try:
                rag.load_models()
                rag.build_knowledge_base(DATA_DIR)
                st.success("Índice gerado com sucesso.")
            except Exception as e:
                st.error(f"Erro ao gerar índice: {e}")

    st.markdown("---")
    st.write("Tipos suportados: PDF, DOCX e TXT")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Digite sua pergunta")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando contexto e gerando resposta..."):
            try:
                rag.load_models()
                answer, contexts = rag.answer(user_prompt)
                st.markdown(answer)
            except Exception as e:
                st.error(f"Erro: {e}")