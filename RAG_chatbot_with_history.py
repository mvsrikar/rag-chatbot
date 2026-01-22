import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model

# Use environment variables, fallback to config if available
google_api_key = os.getenv("GOOGLE_API_KEY")
google_model = os.getenv("GOOGLE_MODEL")

if not os.getenv("GOOGLE_API_KEY") and google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key


# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon=":speaking_head:", # Optional: adds a favicon (emoji or icon)
    layout="centered",    # Optional: other options include "wide"
    initial_sidebar_state="auto" # Optional
)

# Upload pdf files to chatbot
st.header("RAG Chatbot with Multiple PDF Upload")

def upload_pdfs():
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        st.success(f"{len(uploaded_files)} PDF file(s) uploaded successfully!")
        for file in uploaded_files:
            st.write(f"Uploaded: {file.name} - {len(file.read())} bytes")
            file.seek(0)  # Reset file pointer
        return uploaded_files
    return []

with st.sidebar:
    st.title("Your Documents")
    files = upload_pdfs()
    if files:
        pass
    else:
        st.info("Please upload PDF files to get started.")
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Extract text from PDFs

if files:
    all_chunks = []
    all_metadatas = []
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    for file in files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
        all_metadatas.extend([{"source": file.name} for _ in chunks])

    # Generating embeddings 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Storing in vector database
    vector_store = FAISS.from_texts(all_chunks, embedding=embeddings, metadatas=all_metadatas)

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Get user query
    user_query = st.chat_input("Enter your query here:")

    if user_query:
        # Append user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Do similarity search and retrieve relevant chunks (unchanged)
        docs = vector_store.similarity_search(user_query, k=3)
        
        # Generate response using Gemini model with conversation history
        def summarize_conversation(history):
            if not history:
                return ""
            summary_prompt = f"Summarize the following conversation between user and assistant, keeping key points and context for future reference:\n\n" + "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
            model = init_chat_model(f"google_genai:{google_model}")
            response = model.invoke([{"role": "user", "content": summary_prompt}])
            return response.content

        def generate_response_with_history(conversation_history, model=init_chat_model(f"google_genai:{google_model}")):
            # Combine PDF context with the conversation
            combined_context = "\n\n".join([doc.page_content for doc in docs])
            system_message = f"You are a helpful assistant. Use the following context from the uploaded PDF to answer questions: {combined_context}"
            
            # Summarize previous history (excluding the current user query)
            previous_history = conversation_history[:-1]
            summary = summarize_conversation(previous_history)
            
            # Build conversation: system with summary + current user query
            full_conversation = [
                {"role": "system", "content": system_message + ("\n\nConversation summary: " + summary if summary else "")},
                {"role": "user", "content": conversation_history[-1]['content']}
            ]
            
            response = model.invoke(full_conversation)
            return response.content        # Generate and display response
        response = generate_response_with_history(st.session_state.messages)
        
        # Collect sources
        sources = set(doc.metadata['source'] for doc in docs)
        cited_response = response + f"\n\n**Sources:** {', '.join(sources)}"
        
        # Append assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": cited_response})
        
        # Display assistant message
        with st.chat_message("assistant"):
            st.write(cited_response)

