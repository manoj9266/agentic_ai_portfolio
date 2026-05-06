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
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

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
class SafeGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    """Wrapper to fix the 500 INTERNAL error bug in embed_query for gemini-embedding-001."""
    def embed_query(self, text: str) -> list[float]:
        # Route embed_query through embed_documents to avoid task_type='RETRIEVAL_QUERY' crashes
        return self.embed_documents([text])[0]

class RagPDFAssistant:
    """A RAG assistant that loads a PDF and answers questions about it."""
    
    def __init__(self):
        self._validate_env()
        self.rag_chains_with_history = {}
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME, 
            temperature=0,
            google_api_key=GEMINI_API_KEY
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        self.store = {}
        self.contextualize_q_prompt, self.qa_prompt = self._load_prompt_templates()
        
        # Initialize embeddings once to avoid downloading/loading on every upload
        logger.info("Loading embedding model...")
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2"
        # )
        self.embeddings = SafeGoogleEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_API_KEY,
        )

    def _validate_env(self) -> None:
        """Validates that necessary environment variables are set."""
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY environment variable is not set.")
            raise ValueError("Missing GEMINI_API_KEY. Please check your .env file.")  

    def _load_prompt_templates(self):
        """Loads the QA prompt templates for history-aware retrieval."""
        
        # 1. Contextualize Question Prompt:
        # If the user asks a follow-up question like "Can you explain that?", 
        # this prompt uses the chat history to rephrase it into a standalone question 
        # like "Can you explain [specific topic]?".
        # This ensures the document retriever knows exactly what to search for.
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])
        
        # 2. Question Answering Prompt:
        # This is the main prompt that actually answers the user's question.
        # It takes the standalone question, the chat history, and the retrieved PDF context.
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise assistant. Answer only from the provided context.\n\nContext:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])
        return contextualize_q_prompt, qa_prompt         

    def get_session_history(self,session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]    


    def load_document(self, file, request: gr.Request) -> str:
        """Loads, chunks, embeds, and builds the RAG chain for the uploaded PDF."""
        if file is None:
            return "No file provided."
            
        try:
            # Gradio 4+ passes the filepath as a string by default
            file_path = file if isinstance(file, str) else file.name
            logger.info(f"Loading document: {file_path}")
            loader = PyPDFLoader(file_path)
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
            rewrite_chain = (
                    {
                        "question": lambda x: x["question"],
                        "chat_history": lambda x: x.get("chat_history", [])
                    }
                    | self.contextualize_q_prompt
                    | self.llm
                    | StrOutputParser()
                )
            rag_chain = (
                 RunnablePassthrough.assign(rewritten_question=rewrite_chain)
                 .assign(
                     context=lambda x: retriever.invoke(x["rewritten_question"])
                     )
                 | self.qa_prompt
                 | self.llm
                 | StrOutputParser()
            )
            rag_chain_with_history = RunnableWithMessageHistory(rag_chain,
                                    self.get_session_history,
                                    input_messages_key = 'question',
                                    history_messages_key='chat_history'
                                    )
            logger.info("RAG chain successfully built.")
            self.rag_chains_with_history[request.session_hash] = rag_chain_with_history
            return f"✅ Successfully loaded and indexed {len(chunks)} chunks! You can now ask questions in the chat."
            
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return f"❌ Error loading document: {str(e)}"

    def query(self, message: str, history: list, request: gr.Request) -> str:
        """Queries the RAG chain. Signature matches Gradio ChatInterface."""
        rag_chain = self.rag_chains_with_history.get(request.session_hash)
        if not rag_chain:
            return "⚠️ Please upload and process a PDF document first."
            
        try:
            return rag_chain.invoke({"question": message},config={"configurable":{"session_id":request.session_hash}})
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