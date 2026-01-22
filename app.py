import streamlit as st
import os
import tempfile
import shutil
from loaders import build_corpus
from embed_store import build_vectorstore
from rag import rag_chat
from PIL import Image
# Page Config
im=Image.open("logo.png")
st.set_page_config(page_title="Multimodal RAG", page_icon="im",layout="wide")
st.logo("logo.png",size="large")

def save_uploaded_files(uploaded_files):
    """
    Saves uploaded Streamlit files to a temporary directory 
    so loaders.py can read them from disk.
    """
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return temp_dir

def main():
    st.title("📄 Multimodal RAG Chatbot")
    st.markdown("Chat with your PDFs, Images, Videos, and Text files.")

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize Vector Store State
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # --- Sidebar: File Upload ---
    with st.sidebar:
        st.header("1. Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files (PDF, TXT, PNG, JPG, MP4)", 
            accept_multiple_files=True,
            type=["pdf", "txt", "png", "jpg", "mp4"]
        )
        
        process_btn = st.button("Process Files")

        if process_btn and uploaded_files:
            with st.spinner("Processing files... (OCR & Transcription might take a moment)"):
                try:
                    # 1. Save files to temp folder
                    temp_folder = save_uploaded_files(uploaded_files)
                    
                    # 2. Build Corpus (Loaders)
                    # Note: This uses your loaders.py logic
                    corpus = build_corpus(temp_folder)
                    
                    # 3. Build Vector Store (Embeddings)
                    st.session_state.vectorstore = build_vectorstore(corpus)
                    
                    st.success(f"Successfully processed {len(uploaded_files)} files!")
                    
                    # Cleanup: Remove temp folder after loading into memory
                    shutil.rmtree(temp_folder)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # --- Main Chat Interface ---
    
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # 1. Display User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Generate Response
        if st.session_state.vectorstore is None:
            st.error("Please upload and process documents first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Prepare memory object for your rag.py
                        memory = {"messages": st.session_state.messages}
                        
                        # Call your rag_chat function
                        response = rag_chat(prompt, st.session_state.vectorstore, memory)
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
