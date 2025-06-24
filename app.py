"""
RFP Chat Application with Document Caching

This application allows users to chat with their RFP (Request for Proposal) documents
using advanced language models. It supports PDF and DOCX file formats and provides
multiple search methods for document retrieval.

Key features:
- Document caching to avoid reprocessing the same document
- Multiple language model options
- Various search methods for document retrieval
- Conversation history tracking
"""

import os
import re
import tempfile
import warnings
import hashlib
import pickle
import streamlit as st
from typing import List, Optional, Dict, Tuple
from pathlib import Path

from huggingface_hub import InferenceClient
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Document processing imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

# Configuration
HUGGINGFACE_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_TOKEN

# Constants
DEFAULT_PERSIST_DIRECTORY = "chromadb/"
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_CHUNK_OVERLAP = 1000
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
CACHE_DIR = "document_cache"


class DocumentCache:
    """Handles document caching to avoid reprocessing the same document."""
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        """
        Initialize the document cache.
        
        Args:
            cache_dir: Directory to store cached documents
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_file_hash(self, file_content: bytes) -> str:
        """
        Generate a hash for the file content.
        
        Args:
            file_content: Binary content of the file
            
        Returns:
            Hash string representing the file content
        """
        return hashlib.md5(file_content).hexdigest()
        
    def get_cache_path(self, file_hash: str) -> Path:
        """
        Get the cache file path for a given file hash.
        
        Args:
            file_hash: Hash of the file content
            
        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{file_hash}.pkl"
        
    def is_cached(self, file_hash: str) -> bool:
        """
        Check if a document is already cached.
        
        Args:
            file_hash: Hash of the file content
            
        Returns:
            True if the document is cached, False otherwise
        """
        return self.get_cache_path(file_hash).exists()
        
    def save_to_cache(self, file_hash: str, doc_splits: List[Document], vectordb: FAISS) -> None:
        """
        Save document chunks and vector database to cache.
        
        Args:
            file_hash: Hash of the file content
            doc_splits: List of document chunks
            vectordb: Vector database
        """
        cache_data = {
            "doc_splits": doc_splits,
            "vectordb": vectordb
        }
        
        with open(self.get_cache_path(file_hash), 'wb') as f:
            pickle.dump(cache_data, f)
            
    def load_from_cache(self, file_hash: str) -> Tuple[List[Document], FAISS]:
        """
        Load document chunks and vector database from cache.
        
        Args:
            file_hash: Hash of the file content
            
        Returns:
            Tuple of document chunks and vector database
        """
        try:
            with open(self.get_cache_path(file_hash), 'rb') as f:
                cache_data = pickle.load(f)
                return cache_data["doc_splits"], cache_data["vectordb"]
        except Exception as e:
            st.error(f"Error loading from cache: {str(e)}")
            return [], None


class DocumentProcessor:
    """Handles document processing operations including parsing and chunking."""

    @staticmethod
    def parse_text_data(text_data: str) -> List[Document]:
        """
        Parse structured text data from document processing output.
        
        Args:
            text_data: The structured text data from document processing
            
        Returns:
            List of Document objects
        """
        # Matches tags like <text>, <section_header_level_1>, etc. and captures inner text
        pattern = re.compile(
            r"<(text|section_header_level_1|list_item)>(.*?)</\1>", re.DOTALL)

        # Collect all contents into a single page
        contents = []
        for match in pattern.finditer(text_data):
            raw_content = match.group(2)
            # Remove all remaining <...> tags like <loc_...>
            cleaned = re.sub(r"<.*?>", "", raw_content).strip()
            if cleaned:
                contents.append(cleaned)

        # If nothing matched, return an empty list
        if not contents:
            return []

        # Create one document assuming all content belongs to a single page
        document = Document(
            page_content="\n".join(contents),
            metadata={"page": 1}
        )

        return [document]

    @staticmethod
    def load_doc(file_path: str) -> List[Document]:
        """
        Load and process a document file (PDF or DOCX).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks
        """
        try:
            # Configure pipeline options
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True

            # Configure document converter
            doc_converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=PyPdfiumDocumentBackend
                    ),
                    InputFormat.DOCX: WordFormatOption(
                        pipeline_cls=SimplePipeline
                    ),
                },
            )
            
            # Convert document
            result = doc_converter.convert(file_path)
            text_data = result.document.export_to_document_tokens()
            structured_document = DocumentProcessor.parse_text_data(text_data)
            
            # Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP
            )
            doc_splits = text_splitter.split_documents(structured_document)
            return doc_splits
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return []


class FileHandler:
    """Handles file upload and saving operations."""
    
    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """
        Save the uploaded file to a temporary directory.
        
        Args:
            uploaded_file: The uploaded file object from Streamlit
            
        Returns:
            Path to the saved file
        """
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        return file_path


class DocumentRetriever:
    """Handles document retrieval operations using various search methods."""
    
    @staticmethod
    def retrieve_context(chunks: List[Document], query: str, method: str = "similarity_search", 
                        vectordb: Optional[FAISS] = None) -> List[Document]:
        """
        Retrieve relevant document chunks based on the query and search method.
        
        Args:
            chunks: List of document chunks
            query: User query
            method: Search method to use
            vectordb: Vector database for similarity search
            
        Returns:
            List of relevant document chunks
        """
        if not chunks:
            return []
            
        if method == "similarity_search" and vectordb:
            return vectordb.similarity_search(query, k=4)
        elif method == "keyword_match":
            return [doc for doc in chunks if query.lower() in doc.page_content.lower()][:4]
        elif method == "hybrid_search" and vectordb:
            keyword_matches = [
                doc for doc in chunks if query.lower() in doc.page_content.lower()][:2]
            similarity_matches = vectordb.similarity_search(query, k=2)
            return keyword_matches + similarity_matches
        elif method == "basic_top_chunks":
            return chunks[:4]
        else:
            return chunks[:4]


class ChatApp:
    """Main chat application class."""
    
    def __init__(self):
        """Initialize the chat application."""
        self.setup_page()
        self.initialize_session_state()
        self.document_cache = DocumentCache()
        self.setup_sidebar()
        self.setup_client()
        
    def setup_page(self):
        """Configure the Streamlit page."""
        st.set_page_config(page_title="Chat With Your RFP Files")
        st.title("📄AI-based automation for RFP Files")
        st.subheader("This utility enables you to upload an RFP document and extract relevant information by asking questions like Title, Scope of Work, Technical Requirements, and more.")
        
    def initialize_session_state(self):
        """Initialize session state variables."""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "file_name" not in st.session_state:
            st.session_state.file_name = None
        if "file_hash" not in st.session_state:
            st.session_state.file_hash = None
        if "vectordb" not in st.session_state:
            st.session_state.vectordb = None
        if "doc_splits" not in st.session_state:
            st.session_state.doc_splits = []
            
    def setup_sidebar(self):
        """Set up the sidebar with model selection and file upload."""
        llm_models = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "EleutherAI/gpt-neo-2.7B",
            "tiiuae/falcon-40b-instruct",
            "google/flan-t5-xxl"
        ]
        
        search_methods = [
            "similarity_search",
            "keyword_match",
            "hybrid_search",
            "basic_top_chunks"
        ]
        
        self.llm_model = st.sidebar.selectbox(
            "Select LLM Model", llm_models, index=0)
        self.search_method = st.sidebar.selectbox(
            "Select Search Method", search_methods, index=0)
            
        self.uploaded_file = st.sidebar.file_uploader(
            "Upload a PDF or DOCX file", type=["pdf", "docx"])
            
        # Display chat history in sidebar
        if st.session_state.chat_history:
            st.sidebar.subheader("Conversation History")
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:  # User message
                    st.sidebar.text_area(f"You:", value=message["content"], height=68, disabled=True, key=f"hist_{i}")
                else:  # Assistant message
                    st.sidebar.text_area(f"Assistant:", value=message["content"][:100] + "...", height=68, disabled=True, key=f"hist_{i}")
            
    def setup_client(self):
        """Set up the Hugging Face Inference client."""
        try:
            self.client = InferenceClient(
                model=self.llm_model, token=os.environ["HUGGINGFACEHUB_API_TOKEN"])
        except Exception as e:
            st.error(f"Error initializing model client: {str(e)}")
            self.client = None
            
    def process_uploaded_file(self):
        """Process the uploaded file and create vector database."""
        if self.uploaded_file:
            # Get file content for hashing
            file_content = self.uploaded_file.getvalue()
            file_hash = self.document_cache.get_file_hash(file_content)
            file_name = self.uploaded_file.name

            # Check if this is a new file
            if file_hash != st.session_state.file_hash:
                st.session_state.file_name = file_name
                st.session_state.file_hash = file_hash
                st.session_state.chat_history = []
                
                # Check if the document is already cached
                if self.document_cache.is_cached(file_hash):
                    with st.spinner(f"Loading {file_name} from cache..."):
                        doc_splits, vectordb = self.document_cache.load_from_cache(file_hash)
                        if doc_splits and vectordb:
                            st.session_state.doc_splits = doc_splits
                            st.session_state.vectordb = vectordb
                            st.success(f"✅ Document loaded from cache. Total chunks: {len(doc_splits)}")
                        else:
                            st.error("Failed to load document from cache. Processing document...")
                            self._process_and_cache_document(file_content)
                else:
                    # Process and cache the document
                    self._process_and_cache_document(file_content)
    
    def _process_and_cache_document(self, file_content):
        """Process the document and cache the results."""
        file_path = FileHandler.save_uploaded_file(self.uploaded_file)
        file_hash = self.document_cache.get_file_hash(file_content)
        
        with st.spinner(f"Processing {self.uploaded_file.name}..."):
            try:
                doc_splits = DocumentProcessor.load_doc(file_path)
                if doc_splits:
                    st.session_state.doc_splits = doc_splits
                    
                    # Create vector database
                    embedding_model = HuggingFaceEmbeddings(
                        model_name=DEFAULT_EMBEDDING_MODEL)
                    vectordb = FAISS.from_documents(doc_splits, embedding_model)
                    st.session_state.vectordb = vectordb
                    
                    # Cache the processed document
                    self.document_cache.save_to_cache(file_hash, doc_splits, vectordb)
                    
                    st.success(f"✅ Document processed and cached. Total chunks: {len(doc_splits)}")
                else:
                    st.error("Failed to extract content from the document.")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                
    def display_chat_history(self):
        """Display the chat history."""
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    def handle_user_input(self):
        """Handle user input and generate response."""
        user_input = st.chat_input("Ask a question about your document...")
        
        if user_input and st.session_state.vectordb and self.client:
            # Add user message to history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking... 🧠"):
                    try:
                        # Retrieve context
                        retrieved_docs = DocumentRetriever.retrieve_context(
                            st.session_state.doc_splits, 
                            user_input, 
                            method=self.search_method, 
                            vectordb=st.session_state.vectordb
                        )
                        
                        if retrieved_docs:
                            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                            
                            prompt = f"""Use the following context to answer the question at the end.
                                If you don't know the answer, just say you don't know.

                                Context:
                                {context}

                                Question: {user_input}
                                Answer:"""

                            # LLM Response
                            response = self.client.chat_completion(
                                messages=[{"role": "user", "content": prompt}])
                            answer = response.choices[0].message["content"]
                        else:
                            answer = "I couldn't find relevant information in the document to answer your question."
                            
                        # Add LLM response to history
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer})
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        
    def run(self):
        """Run the chat application."""
        self.process_uploaded_file()
        
        if st.session_state.vectordb:
            self.display_chat_history()
            self.handle_user_input()
        else:
            st.info("Please upload a PDF or DOCX file to start chatting.")


if __name__ == "__main__":
    app = ChatApp()
    app.run()
