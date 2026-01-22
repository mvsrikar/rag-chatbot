# RAG Chatbot with History

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot that allows uploading multiple PDFs, maintaining conversation history, and citing sources.

## Features
- Upload multiple PDF documents
- Conversational AI with history and summarization
- Source citation for responses
- Guardrails to stay within document context
- Dockerized for easy deployment

## Local Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables:
   - `GOOGLE_API_KEY`: Your Google Gemini API key
   - `GOOGLE_MODEL`: Model name (e.g., `gemini-1.5-flash`)
3. Run: `streamlit run RAG_chatbot_with_history.py`

## Docker Setup
1. Build: `docker build -t rag-chatbot .`
2. Run: `docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key -e GOOGLE_MODEL=gemini-1.5-flash rag-chatbot`
3. Access at `http://localhost:8501`

## Docker Hub
The Docker image is automatically built and pushed to Docker Hub on pushes to `main`.

Pull the image: `docker pull mvsrikar/rag-chatbot:latest`

Run as above.

## Usage
- Upload PDFs in the sidebar.
- Chat with the bot—responses are based on the documents with sources cited.
- Clear history with the button in the sidebar.