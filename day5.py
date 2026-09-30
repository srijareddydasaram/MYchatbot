import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
st.set_page_config(page_title="Mini RAG", page_icon=":books:")
st.title("Mini RAG :books: Document Store + Retrieval")
st.caption("PDF -> Chunks -> Enbeddings -> Chromedb -> Retrieval -> Ollama")
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
model = load_embedding_model()
client=chromadb.PersistentClient(path="./chroma_db")
collection=client.get_or_create_collection(name="documents")
st.sidebar.header("Settings")
ollama model=st.sidebar.text_input("Ollama Model", value="llama2")
chunk_size=st.sidebar.slider("Chunks size", 200, 1500, 500, 100)
top_k=st.sidebar.slider("Chunks to retrieve", 1, 2, 3)
st.header(" Build Document Store")
uploaded_file=st.file_uploader("Upload a text-based PDF, type=["pdf"])
if uploaded_file and st.button(" Process & Store PDF"):
    reader=Pdfreader(uploaded_file)
    text="" 
    for page_number,page in enumerate(reader.page, start=1):
        page_text=page.extract_text()
        if page_text:
            text+=page_text+"\n"
    if not text.strip():
        st.error("No readable text was found. Try a text-based PDF.")
        st.stop()
    chunks=[]
    for i in range(0, len(text), chunk_size):
        chunk=text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)