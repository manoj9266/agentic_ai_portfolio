# RAG PDF Assistant

A production-ready Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and ask questions about their contents. This application uses Google's Gemini 2.5 Flash, Google Generative AI Embeddings, FAISS vector store, and Gradio for the UI.

## Features
- **Concurrent User Support**: Uses Gradio's state management to ensure that multiple users can upload and query their own PDFs independently without interfering with each other's sessions.
- **Optimized Embedding Loading**: Embedding models are loaded efficiently on startup to prevent redundant downloading or reloading upon subsequent PDF uploads.
- **Robust Error Handling**: Handles empty PDFs, unreadable files, and API errors gracefully.
- **Professional UI**: Employs `gr.Blocks` and `gr.ChatInterface` for a clean, intuitive layout.

## Setup Instructions

1. **Clone the repository and navigate to the project folder:**
   ```bash
   cd rag_pdf_assistant
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Rename `.env.example` to `.env` and add your Google Gemini API Key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Run the Application:**
   ```bash
   python rag_pdf_assistant.py
   ```
   Open your browser to the local URL provided by Gradio (usually http://127.0.0.1:7860).

## Tech Stack
- [LangChain](https://python.langchain.com/) for LCEL and orchestration.
- [Google Generative AI](https://ai.google.dev/) (Gemini) for both the LLM and Embeddings.
- [FAISS](https://faiss.ai/) for the local vector database.
- [Gradio](https://www.gradio.app/) for the web interface.

---
*This application is automatically deployed to HuggingFace Spaces via GitHub Actions.*


