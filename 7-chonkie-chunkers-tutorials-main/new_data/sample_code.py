"""
Chonkie RAG System Implementation
A production-ready implementation of a Retrieval-Augmented Generation system using Chonkie chunkers.
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkerType(Enum):
    """Enumeration of available chunker types."""
    TOKEN = "token"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    CODE = "code"
    TABLE = "table"
    NEURAL = "neural"
    LATE = "late"
    SLUMBER = "slumber"


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    token_count: int
    chunk_id: str
    source_document: str
    start_index: int
    end_index: int
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Validate chunk after initialization."""
        if not self.text:
            raise ValueError("Chunk text cannot be empty")
        if self.token_count <= 0:
            raise ValueError("Token count must be positive")

    def to_dict(self) -> Dict:
        """Convert chunk to dictionary representation."""
        return {
            "text": self.text,
            "token_count": self.token_count,
            "chunk_id": self.chunk_id,
            "source_document": self.source_document,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "metadata": self.metadata or {}
        }


class DocumentProcessor:
    """Handles document loading and preprocessing."""

    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize document processor.

        Args:
            supported_formats: List of supported file extensions (default: txt, md, py)
        """
        self.supported_formats = supported_formats or ['.txt', '.md', '.py', '.json']
        logger.info(f"DocumentProcessor initialized with formats: {self.supported_formats}")

    def load_document(self, filepath: str) -> str:
        """
        Load document from file.

        Args:
            filepath: Path to document file

        Returns:
            Document text content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Document not found: {filepath}")

        ext = os.path.splitext(filepath)[1]
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {ext}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded document: {filepath} ({len(content)} chars)")
            return content
        except UnicodeDecodeError:
            logger.error(f"Failed to decode {filepath}")
            raise

    def preprocess(self, text: str) -> str:
        """
        Preprocess text before chunking.

        Args:
            text: Raw text content

        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Normalize line endings
        text = text.replace('\r\n', '\n')
        return text


class ChunkerFactory:
    """Factory for creating chunker instances."""

    @staticmethod
    def create_chunker(chunker_type: ChunkerType, **kwargs):
        """
        Create a chunker instance.

        Args:
            chunker_type: Type of chunker to create
            **kwargs: Chunker-specific parameters

        Returns:
            Chunker instance

        Raises:
            ValueError: If chunker type is invalid
        """
        try:
            if chunker_type == ChunkerType.TOKEN:
                from chonkie import TokenChunker
                return TokenChunker(**kwargs)
            elif chunker_type == ChunkerType.SENTENCE:
                from chonkie import SentenceChunker
                return SentenceChunker(**kwargs)
            elif chunker_type == ChunkerType.RECURSIVE:
                from chonkie import RecursiveChunker
                return RecursiveChunker(**kwargs)
            elif chunker_type == ChunkerType.SEMANTIC:
                from chonkie import SemanticChunker
                return SemanticChunker(**kwargs)
            elif chunker_type == ChunkerType.CODE:
                from chonkie import CodeChunker
                return CodeChunker(**kwargs)
            elif chunker_type == ChunkerType.TABLE:
                from chonkie import TableChunker
                return TableChunker(**kwargs)
            elif chunker_type == ChunkerType.NEURAL:
                from chonkie import NeuralChunker
                return NeuralChunker(**kwargs)
            elif chunker_type == ChunkerType.LATE:
                from chonkie import LateChunker
                return LateChunker(**kwargs)
            elif chunker_type == ChunkerType.SLUMBER:
                from chonkie import SlumberChunker
                return SlumberChunker(**kwargs)
            else:
                raise ValueError(f"Unknown chunker type: {chunker_type}")
        except ImportError as e:
            logger.error(f"Failed to import chunker: {e}")
            raise


class EmbeddingManager:
    """Manages embedding generation and caching."""

    def __init__(self, model_name: str = "gemini-embedding-exp-03-07"):
        """
        Initialize embedding manager.

        Args:
            model_name: Name of embedding model to use
        """
        self.model_name = model_name
        self.cache = {}
        self.embedding_dim = None
        logger.info(f"EmbeddingManager initialized with model: {model_name}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # Check cache
        cache_key = hash(text)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate embedding (placeholder - would use actual model)
        embedding = np.random.randn(768)  # Example dimension
        self.cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return [self.embed_text(text) for text in texts]

    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")


class VectorStore:
    """Simple in-memory vector store for chunk retrieval."""

    def __init__(self):
        """Initialize vector store."""
        self.chunks: List[Chunk] = []
        self.embeddings: List[np.ndarray] = []
        logger.info("VectorStore initialized")

    def add_chunks(self, chunks: List[Chunk], embeddings: List[np.ndarray]):
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks: List of text chunks
            embeddings: Corresponding embedding vectors
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        logger.info(f"Added {len(chunks)} chunks to vector store")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search for most similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.embeddings:
            return []

        # Compute cosine similarity
        similarities = []
        for chunk, embedding in zip(self.chunks, self.embeddings):
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, similarity))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def clear(self):
        """Clear all chunks and embeddings."""
        self.chunks.clear()
        self.embeddings.clear()
        logger.info("VectorStore cleared")


class RAGPipeline:
    """Complete RAG pipeline implementation."""

    def __init__(
        self,
        chunker_type: ChunkerType,
        embedding_manager: EmbeddingManager,
        chunker_kwargs: Optional[Dict] = None
    ):
        """
        Initialize RAG pipeline.

        Args:
            chunker_type: Type of chunker to use
            embedding_manager: Embedding manager instance
            chunker_kwargs: Parameters for chunker initialization
        """
        self.chunker = ChunkerFactory.create_chunker(
            chunker_type,
            **(chunker_kwargs or {})
        )
        self.embedding_manager = embedding_manager
        self.vector_store = VectorStore()
        self.doc_processor = DocumentProcessor()
        logger.info(f"RAGPipeline initialized with {chunker_type.value} chunker")

    def ingest_document(self, filepath: str):
        """
        Ingest document into RAG system.

        Args:
            filepath: Path to document file
        """
        # Load and preprocess
        text = self.doc_processor.load_document(filepath)
        text = self.doc_processor.preprocess(text)

        # Chunk document
        chunks = self.chunker.chunk(text)
        logger.info(f"Created {len(chunks)} chunks from {filepath}")

        # Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_manager.embed_batch(chunk_texts)

        # Add to vector store
        self.vector_store.add_chunks(chunks, embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """
        Retrieve relevant chunks for query.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant chunks
        """
        # Embed query
        query_embedding = self.embedding_manager.embed_text(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k)

        # Extract chunks
        chunks = [chunk for chunk, _ in results]
        logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
        return chunks

    def generate_answer(self, query: str, context_chunks: List[Chunk]) -> str:
        """
        Generate answer using retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks

        Returns:
            Generated answer
        """
        # Combine context
        context = "\n\n".join([chunk.text for chunk in context_chunks])

        # Generate answer (placeholder - would use LLM)
        answer = f"Based on the context, here's the answer to '{query}':\n{context[:200]}..."
        return answer

    def query(self, query: str, top_k: int = 5) -> str:
        """
        End-to-end query processing.

        Args:
            query: User query
            top_k: Number of chunks to retrieve

        Returns:
            Generated answer
        """
        chunks = self.retrieve(query, top_k)
        answer = self.generate_answer(query, chunks)
        return answer


def main():
    """Main entry point for RAG system demo."""
    # Initialize components
    embedding_manager = EmbeddingManager()

    # Create RAG pipeline with semantic chunker
    pipeline = RAGPipeline(
        chunker_type=ChunkerType.SEMANTIC,
        embedding_manager=embedding_manager,
        chunker_kwargs={
            "chunk_size": 512,
            "threshold": "auto"
        }
    )

    # Ingest documents
    documents = [
        "sample_technical_doc.txt",
        "sample_research_paper.txt"
    ]

    for doc in documents:
        if os.path.exists(doc):
            pipeline.ingest_document(doc)

    # Query system
    query = "What are the best practices for chunking in RAG systems?"
    answer = pipeline.query(query)
    print(f"\nQuery: {query}")
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
