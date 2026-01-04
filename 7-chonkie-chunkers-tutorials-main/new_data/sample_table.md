# Chonkie Chunkers Comparison Table

## Performance Metrics Across Different Chunkers

| Chunker Name | Use Case | Processing Speed (chunks/sec) | Semantic Quality (0-100) | Memory Usage (MB) | Token Consistency | Best For | Computational Cost |
|---|---|---|---|---|---|---|---|
| TokenChunker | General purpose | 12000 | 61 | 50 | Very High | High-throughput applications, real-time systems | Very Low |
| SentenceChunker | Q&A systems | 8500 | 71 | 65 | High | Question answering, semantic search | Low |
| RecursiveChunker | Structured docs | 7200 | 76 | 80 | Medium | Markdown, technical documentation | Low |
| TableChunker | Tabular data | 5500 | 68 | 95 | Medium | Data tables, spreadsheet content | Low |
| SemanticChunker | Multi-topic docs | 450 | 78 | 350 | Low | Multi-topic documents, topical coherence | Medium |
| LateChunker | RAG retrieval | 180 | 82 | 800 | Low | Maximum retrieval recall | High |
| CodeChunker | Source code | 3200 | 88 | 120 | Medium | API docs, code repositories | Medium |
| NeuralChunker | Complex documents | 320 | 85 | 650 | Low | Subtle topic variations | High |
| SlumberChunker | Premium quality | 8 | 92 | 400 | Very Low | Books, research papers, highest quality needs | Very High |

## Feature Comparison

| Feature | TokenChunker | SentenceChunker | RecursiveChunker | SemanticChunker | LateChunker | CodeChunker | NeuralChunker | TableChunker | SlumberChunker |
|---|---|---|---|---|---|---|---|---|---|
| Overlap Support | Yes | Yes | Yes | No | No | Limited | No | No | No |
| Preserves Sentences | No | Yes | Partial | Yes | Yes | N/A | Yes | N/A | Yes |
| Embedding Required | No | No | No | Yes | Yes | No | Yes | No | Yes |
| Language Agnostic | Yes | Partial | Yes | Yes | Yes | No | Partial | Yes | Partial |
| Thread Safe | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Limited |
| Batch Processing | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Limited |
| Configurable Threshold | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Custom Separators | No | No | Yes | No | No | Yes | No | No | No |
| AST Parsing | No | No | No | No | No | Yes | No | No | No |
| LLM Powered | No | No | No | No | No | No | No | No | Yes |

## Resource Requirements

| Chunker | CPU Usage (%) | GPU Required | RAM (GB) | API Key Required | Cost Per 1M Chunks | Latency (ms/chunk) |
|---|---|---|---|---|---|---|
| TokenChunker | 5-10 | No | 0.5 | No | $0 | 0.08 |
| SentenceChunker | 8-12 | No | 0.6 | No | $0 | 0.12 |
| RecursiveChunker | 10-15 | No | 0.8 | No | $0 | 0.14 |
| TableChunker | 12-18 | No | 1.0 | No | $0 | 0.18 |
| SemanticChunker | 25-35 | Optional | 2.5 | Yes (embeddings) | $2.50 | 2.22 |
| LateChunker | 45-60 | Recommended | 4.0 | Yes (embeddings) | $8.00 | 5.56 |
| CodeChunker | 18-25 | No | 1.2 | No | $0 | 0.31 |
| NeuralChunker | 40-55 | Recommended | 3.5 | Optional | $0 | 3.13 |
| SlumberChunker | 10-15 | No | 2.0 | Yes (LLM) | $45.00 | 125.00 |

## Supported Document Types

| Document Type | TokenChunker | SentenceChunker | RecursiveChunker | SemanticChunker | LateChunker | CodeChunker | NeuralChunker | TableChunker | SlumberChunker |
|---|---|---|---|---|---|---|---|---|---|
| Plain Text | Excellent | Excellent | Excellent | Excellent | Excellent | Poor | Excellent | Poor | Excellent |
| Markdown | Good | Good | Excellent | Good | Good | Poor | Good | Good | Excellent |
| HTML | Good | Good | Good | Good | Good | Poor | Good | Fair | Good |
| Source Code | Poor | Poor | Fair | Poor | Poor | Excellent | Poor | Poor | Good |
| JSON | Fair | Fair | Fair | Fair | Fair | Good | Fair | Poor | Good |
| CSV/Tables | Poor | Poor | Poor | Poor | Poor | Poor | Poor | Excellent | Fair |
| PDFs | Good | Good | Good | Good | Good | Fair | Good | Fair | Excellent |
| Research Papers | Good | Excellent | Good | Excellent | Excellent | Poor | Excellent | Fair | Excellent |
| API Docs | Good | Good | Excellent | Good | Good | Excellent | Good | Good | Excellent |
| Conversations | Good | Excellent | Fair | Good | Good | Poor | Excellent | Poor | Excellent |

## Version Compatibility

| Chunker | Chonkie Version | Python Version | Dependencies | Installation Command |
|---|---|---|---|---|
| TokenChunker | >=1.0.0 | >=3.8 | autotiktokenizer | pip install chonkie |
| SentenceChunker | >=1.0.0 | >=3.8 | autotiktokenizer | pip install chonkie |
| RecursiveChunker | >=1.0.0 | >=3.8 | autotiktokenizer | pip install chonkie |
| TableChunker | >=1.0.0 | >=3.8 | autotiktokenizer | pip install chonkie |
| SemanticChunker | >=1.2.0 | >=3.9 | model2vec, numpy | pip install chonkie[semantic] |
| LateChunker | >=1.2.0 | >=3.9 | model2vec, numpy | pip install chonkie[semantic] |
| CodeChunker | >=1.1.0 | >=3.8 | tree-sitter | pip install chonkie[code] |
| NeuralChunker | >=1.3.0 | >=3.9 | transformers, torch | pip install chonkie[semantic] |
| SlumberChunker | >=1.4.0 | >=3.9 | openai/gemini | pip install chonkie[all] |

## Benchmark Results on Standard Datasets

| Dataset | TokenChunker | SentenceChunker | RecursiveChunker | SemanticChunker | LateChunker | CodeChunker | NeuralChunker | TableChunker | SlumberChunker |
|---|---|---|---|---|---|---|---|---|---|
| Wikipedia (Recall@5) | 0.68 | 0.74 | 0.72 | 0.85 | 0.88 | N/A | 0.86 | N/A | 0.92 |
| TechDocs (Recall@5) | 0.71 | 0.76 | 0.82 | 0.84 | 0.87 | 0.91 | 0.85 | N/A | 0.93 |
| AcademicPapers (Recall@5) | 0.66 | 0.79 | 0.71 | 0.89 | 0.91 | N/A | 0.90 | N/A | 0.95 |
| SourceCode (Recall@5) | 0.52 | 0.58 | 0.64 | 0.61 | 0.67 | 0.94 | 0.63 | N/A | 0.82 |
| Conversations (Recall@5) | 0.73 | 0.81 | 0.68 | 0.79 | 0.82 | N/A | 0.84 | N/A | 0.89 |

## Recommended Chunk Sizes

| Chunker | Minimum Size (tokens) | Optimal Size (tokens) | Maximum Size (tokens) | Recommended Overlap (%) |
|---|---|---|---|---|
| TokenChunker | 128 | 512 | 2048 | 10-20% |
| SentenceChunker | 256 | 512 | 1024 | 10-15% |
| RecursiveChunker | 256 | 512 | 2048 | 10-20% |
| TableChunker | 128 | 512 | 4096 | 0% |
| SemanticChunker | 256 | 512 | 1024 | N/A |
| LateChunker | 512 | 1024 | 4096 | N/A |
| CodeChunker | 512 | 2048 | 8192 | 5-10% |
| NeuralChunker | 256 | 512 | 1024 | N/A |
| SlumberChunker | 256 | 512 | 2048 | N/A |
