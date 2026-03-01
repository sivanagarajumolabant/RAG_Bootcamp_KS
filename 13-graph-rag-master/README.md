# GraphRAG with LangChain + Neo4j

A PDF-to-Knowledge-Graph RAG pipeline using LangChain, LangGraph, and Neo4j.

---

## What It Does

Loads a PDF → Extracts a Knowledge Graph → Hybrid Retrieval → LLM Q&A

```
PDF → PyMuPDF Loader → Text Chunks
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     OpenAI Embeddings         LLMGraphTransformer
     Neo4j Vector Index         Neo4j Property Graph
              │                           │
              └─────────────┬─────────────┘
                            ▼
                   EnsembleRetriever
                            │
                   LangGraph Router
                   ├── simple   → Vector Search
                   ├── relational → Graph Traversal
                   └── complex  → Hybrid (both)
                            │
                       GPT-4o Answer
```

---

## Stack

| Component | Library |
|-----------|---------|
| Graph Extraction | `langchain-experimental` — `LLMGraphTransformer` |
| Graph Storage | `langchain-neo4j` — `Neo4jGraph` |
| Vector Index | `langchain-neo4j` — `Neo4jVector` (hybrid) |
| Workflow | `langgraph` — `StateGraph` |
| Cypher Q&A | `langchain-neo4j` — `GraphCypherQAChain` |
| Evaluation | `ragas` |
| LLM | `langchain-openai` — GPT-4o / GPT-4o-mini |

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure `.env`**
```env
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

**3. Start Neo4j**

- Local: [Neo4j Desktop](https://neo4j.com/download/) or Docker
- Cloud: [Neo4j AuraDB](https://neo4j.com/cloud/aura/) (free tier available)

**4. Run the notebook**
```
graphrag_langchain.ipynb
```

---

## Notebook Structure

| Section | What it does |
|---------|-------------|
| 1. Setup | Install deps, load env |
| 2. PDF Loading | PyMuPDF + RecursiveCharacterTextSplitter |
| 3. KG Construction | LLMGraphTransformer → Neo4j |
| 4. Vector Index | Neo4j hybrid (vector + BM25 keyword) |
| 5. LangGraph Workflow | Query classification + routing |
| 6. Querying | `ask_graphrag()` — natural language Q&A |
| 7. Evaluation | RAGAS metrics |

---

## Quick Usage

```python
# Ask anything about your PDF
ask_graphrag("What are the main causes of climate change?")
ask_graphrag("How does deforestation relate to greenhouse gases?")
ask_graphrag("What mitigation strategies are discussed?")
```

---

## Requirements

- Python ≥ 3.12
- Neo4j ≥ 5.11 (vector index support)
- OpenAI API key

---

*KrishAI Technologies — Senior AI Consultant | 2026*
