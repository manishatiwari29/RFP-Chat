# RFP Chat

A conversational AI application that allows you to chat with your RFP (Request for Proposal) documents using advanced language models.

## Features

- Upload and process PDF and DOCX files
- Document caching to avoid reprocessing the same document
- Fast document processing with Docling and PyPdfium
- Interactive chat interface with Streamlit
- Multiple language model options
- Various search methods for document retrieval
- Conversation history tracking
- OCR support for scanned documents
- Table structure recognition

## Installation Guide

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

#### Windows

1. **Create a virtual environment**:
   ```cmd
   # Navigate to your project directory
   cd path\to\rfp-chat

   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face API token**:
   - Create an account on [Hugging Face](https://huggingface.co/settings/tokens)
   - Generate a new token with read access
   - Open `app.py` and replace the API token:
     ```python
     HUGGINGFACE_API_TOKEN = "your_hugging_face_api_token"
     ```

#### macOS/Linux

1. **Create a virtual environment**:
   ```bash
   # Navigate to your project directory
   cd path/to/rfp-chat

   # Create a virtual environment
   python3 -m venv venv

   # Activate the virtual environment
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face API token**:
   - Create an account on [Hugging Face](https://huggingface.co/settings/tokens)
   - Generate a new token with read access
   - Open `app.py` and replace the API token:
     ```python
     HUGGINGFACE_API_TOKEN = "your_hugging_face_api_token"
     ```

### Troubleshooting Common Installation Issues

#### PyTorch/torchvision Issues

If you encounter errors related to PyTorch or torchvision (e.g., "operator torchvision::nms does not exist"), try installing them separately:

```bash
# For CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8 (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Dependency Conflicts

If you encounter dependency conflicts, try installing with the `--no-deps` flag and then manually install required dependencies:

```bash
pip install --no-deps transformers sentence-transformers
pip install torch==2.0.1 torchvision==0.15.2
```

## Running the Application

After installation, run the application with:

```bash
# Make sure your virtual environment is activated
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

1. **Select Model and Search Method**:
   - Choose a language model from the sidebar
   - Select a search method (similarity search, keyword match, hybrid search, or basic top chunks)

2. **Upload Document**:
   - Click "Upload a PDF or DOCX file" in the sidebar
   - Select your RFP document

3. **Chat with Your Document**:
   - Type your questions in the chat input
   - View responses based on document content
   - Check conversation history in the sidebar

## Available Models

- mistralai/Mistral-7B-Instruct-v0.2
- EleutherAI/gpt-neo-2.7B
- tiiuae/falcon-40b-instruct
- google/flan-t5-xxl

## Search Methods

- **Similarity Search**: Vector-based similarity search using FAISS
- **Keyword Match**: Simple keyword matching in document chunks
- **Hybrid Search**: Combines vector similarity and keyword matching
- **Basic Top Chunks**: Returns the first few chunks of the document

## Technical Details

- Document processing using Docling library
- OCR and table structure recognition for PDF files
- Document chunking with RecursiveCharacterTextSplitter
- Vector embeddings with sentence-transformers/all-MiniLM-L6-v2
- Vector storage with FAISS
- MD5 hashing for document caching
