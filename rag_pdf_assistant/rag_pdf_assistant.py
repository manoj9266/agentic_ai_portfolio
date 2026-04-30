"""
RAG PDF Assistant. It helps you query your PDF documents using Google Gemini and LangChain.
"""
import os
import logging

import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load Environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

class RagPDFAssistant:
    """A RAG assistant that loads a PDF and answers questions about it."""
    
    def __init__(self):
        self._validate_env()
        self.rag_chains = {}
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME, 
            temperature=0,
            google_api_key=GEMINI_API_KEY
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        self.prompt = self._load_prompt_template()
        
        # Initialize embeddings once to avoid downloading/loading on every upload
        logger.info("Loading embedding model...")
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2"
        # )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_API_KEY,
        )

    def _validate_env(self) -> None:
        """Validates that necessary environment variables are set."""
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY environment variable is not set.")
            raise ValueError("Missing GEMINI_API_KEY. Please check your .env file.")  
    
    def _load_prompt_template(self):
        """Loads the QA prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", "You are a precise assistant. Answer only from the provided context.\n\nContext:\n{context}"),
            ("human", "{question}")                                                   
        ])

    def load_document(self, file, request: gr.Request) -> str:
        """Loads, chunks, embeds, and builds the RAG chain for the uploaded PDF."""
        if file is None:
            return "No file provided."
            
        try:
            logger.info(f"Loading document: {file.name}")
            loader = PyPDFLoader(file.name)
            docs = loader.load()

            if not docs:
                return "The uploaded PDF appears to be empty or cannot be read."
            
            # Chunk document
            chunks = self.splitter.split_documents(docs)
            logger.info(f"Document split into {len(chunks)} chunks.")

            # Generate Vector Store
            logger.info("Building vector store...")
            vs = FAISS.from_documents(chunks, self.embeddings)
            retriever = vs.as_retriever(search_kwargs={"k": 5})

            # Build LCEL Chain
            rag_chain = (
                 {"context": retriever, "question": RunnablePassthrough()}
                 | self.prompt
                 | self.llm
                 | StrOutputParser()
            )
            logger.info("RAG chain successfully built.")
            self.rag_chains[request.session_hash] = rag_chain
            return f"✅ Successfully loaded and indexed {len(chunks)} chunks! You can now ask questions in the chat."
            
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return f"❌ Error loading document: {str(e)}"

    def query(self, message: str, history: list, request: gr.Request) -> str:
        """Queries the RAG chain. Signature matches Gradio ChatInterface."""
        rag_chain = self.rag_chains.get(request.session_hash)
        if not rag_chain:
            return "⚠️ Please upload and process a PDF document first."
            
        try:
            return rag_chain.invoke(message)
        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return f"❌ An error occurred while generating the answer: {str(e)}"


def create_ui() -> gr.Blocks:
    """Creates the Gradio Blocks UI."""
    try:
        assistant = RagPDFAssistant()
    except Exception as e:
        logger.error(e)
        error_msg = str(e)
        return gr.Interface(
            fn=lambda *args: f"Configuration Error: {error_msg}",
            inputs=gr.Textbox(),
            outputs=gr.Textbox(),
            title="📄 RAG PDF Assistant (Error)"
        )

    with gr.Blocks(title="📄 AI RAG PDF Assistant") as interface:
        gr.Markdown("# 📄 RAG PDF Assistant\nUpload a PDF document, and then ask the AI questions about its contents!")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload Document")
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_status = gr.Textbox(label="Status", interactive=False)
                
            with gr.Column(scale=2):
                gr.Markdown("### 2. Chat with your Document")
                chatbot = gr.ChatInterface(
                    fn=assistant.query,
                )
        
        # Link the file upload action to the backend
        pdf_input.upload(
            fn = assistant.load_document,
            inputs = [pdf_input],
            outputs = [upload_status]
        )
        
    return interface


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)