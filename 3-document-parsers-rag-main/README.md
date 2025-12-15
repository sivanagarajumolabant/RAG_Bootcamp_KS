# Document Parsers for RAG

A comprehensive collection of Jupyter notebooks demonstrating four document parsing frameworks for building RAG (Retrieval-Augmented Generation) systems.

## Overview

This project provides hands-on tutorials for parsing documents using:

| Framework | Type | Best For |
|-----------|------|----------|
| **Docling** | Open-source (IBM) | Scientific documents, tables |
| **Unstructured** | Open-source | General purpose, fast processing |
| **LlamaParse** | Cloud service | Complex layouts, multimodal |
| **Vectorize.io** | Cloud platform | End-to-end RAG pipelines |

## Project Structure

```
document-parsers-rag/
├── docling_document_parsing.ipynb          # Docling tutorial
├── unstructured_document_parsing.ipynb     # Unstructured tutorial
├── llamaparse_document_parsing.ipynb # LlamaParse basics
├── llamaparse_document_parsing.ipynb # LlamaParse advanced + RAG
├── vectorize_document_parsing.ipynb  # Vectorize.io basics
├── vectorize_document_parsing.ipynb  # Vectorize.io pipelines + RAG
├── sample_documents/                       # Test documents (PDF, DOCX, etc.)
├── requirements.txt                        # Python dependencies
└── .env.example                            # Environment variables template
```

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd document-parsers-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Notebooks

```bash
jupyter notebook
```

## Environment Variables

```bash
# Required for RAG examples
OPENAI_API_KEY=your-openai-api-key

# LlamaParse (https://cloud.llamaindex.ai/)
LLAMA_CLOUD_API_KEY=llx-your-api-key

# Vectorize.io (https://platform.vectorize.io/)
VECTORIZE_ORGANIZATION_ID=your-org-id
VECTORIZE_API_KEY=your-api-key

# Optional
HF_TOKEN=your-huggingface-token
PINECONE_API_KEY=your-pinecone-key
```

## Notebooks Summary

### Docling (`docling_document_parsing.ipynb`)

IBM's open-source document parser with advanced table recognition.

**Features:**
- PDF, DOCX, PPTX, HTML, image parsing
- Multiple OCR engines (RapidOCR, EasyOCR, Tesseract)
- TableFormer for table structure recognition
- VLM (Vision-Language Model) pipeline support
- LangChain integration via DoclingLoader

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")
markdown = result.document.export_to_markdown()
```

### Unstructured (`unstructured_document_parsing.ipynb`)

Fast, general-purpose document processing library.

**Features:**
- Auto-detection with `partition()` function
- Multiple strategies: fast, hi_res, ocr_only
- Element type filtering (Title, Table, NarrativeText, etc.)
- Built-in chunking strategies
- LangChain UnstructuredLoader integration

```python
from unstructured.partition.auto import partition

elements = partition(filename="document.pdf", strategy="hi_res")
tables = [e for e in elements if e.category == "Table"]
```

### LlamaParse (Part 1 & 2)

LlamaIndex's GenAI-native cloud parser for complex documents.

**Features:**
- Presets: fast, balanced, premium, scientific, invoice, slides
- Multimodal extraction (charts, screenshots, OCR)
- Structured output with JSON schemas
- Custom parsing instructions
- Async batch processing

```python
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex

parser = LlamaParse(result_type="markdown", preset="scientific")
documents = parser.load_data("document.pdf")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
```

**Pricing:** Free tier 1,000 pages/day, Paid $0.003/page

### Vectorize.io (Part 1 & 2)

All-in-one RAG platform with built-in vector database.

**Features:**
- Iris AI-powered extraction (100+ languages)
- Built-in RAG Pipeline Builder (free vector DB)
- Source connectors: File Upload, S3, Google Drive, SharePoint
- Semantic search with re-ranking
- Deep Research for AI-generated reports

```python
import vectorize_client as v

# Create pipeline
pipeline = pipelines_api.create_pipeline(
    org_id,
    v.CreatePipelineRequest(
        name="my_pipeline",
        extraction=v.ExtractionConfig(strategy="IRIS"),
        embedder=v.EmbedderConfig(provider="OPENAI"),
        vector_database=v.VectorDatabaseConfig(type="VECTORIZE")
    )
)

# Retrieve documents
response = pipelines_api.retrieve_documents(
    org_id, pipeline_id,
    v.RetrieveDocumentsRequest(question="your query", rerank=True)
)
```

**Pricing:** Free tier 1,000 pages/day for Iris extraction

## Framework Comparison

| Feature | Docling | Unstructured | LlamaParse | Vectorize.io |
|---------|---------|--------------|------------|--------------|
| **Cost** | Free | Free | 1K pages/day free | 1K pages/day free |
| **OCR** | Multiple engines | Tesseract | Built-in | AI-powered |
| **Tables** | TableFormer | Built-in | AI-powered | AI-powered |
| **Speed** | Medium | Fast | Fast | Fast |
| **Built-in Vector DB** | No | No | No | Yes (free) |
| **Deep Research** | No | No | No | Yes |

## RAG Pipeline Pattern

All notebooks follow a similar pattern for building RAG systems:

```python
# 1. Parse documents
documents = parser.load_data("document.pdf")

# 2. Setup ChromaDB
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=collection)

# 3. Create index
from llama_index.core import VectorStoreIndex, StorageContext

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# 4. Query
query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
```

## Sample Documents

The `sample_documents/` directory contains test files:
- `attention_paper.pdf` - "Attention Is All You Need" paper
- `sample.html` - HTML with tables, lists, code blocks
- `sample.md` - Markdown with various formatting
- `sample.docx`, `sample.pptx`, `sample.xlsx` - Office documents
- `scan.pdf`, `scanned.pdf` - Scanned documents for OCR testing

## Requirements

- Python 3.10-3.12
- See `requirements.txt` for full dependencies

### Framework-specific installations:

```bash
# Docling with extras
pip install 'docling[easyocr]'     # EasyOCR support
pip install 'docling[vlm]'          # Vision-Language Models

# Unstructured with all formats
pip install "unstructured[all-docs]"

# LlamaParse
pip install llama-parse llama-index

# Vectorize.io
pip install vectorize-client
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API Key Errors | Ensure `.env` file exists with valid keys |
| Timeout on Large PDFs | Increase `job_timeout_in_seconds` |
| OCR Quality | Use `high_res_ocr=True` or `strategy="hi_res"` |
| Memory Issues | Process documents in batches |
| Table Extraction | Use `premium_mode` or structured presets |

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
