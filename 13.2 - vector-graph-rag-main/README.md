# Hybrid RAG: Vector Search + Knowledge Graph

> **A practical, hands-on demonstration of combining Pinecone vector search and Neo4j graph database to build a richer, more contextual AI retrieval system.**

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique used to make AI systems more accurate and grounded. Instead of relying solely on what a language model memorized during training, RAG:

1. **Retrieves** relevant documents from an external knowledge base at query time
2. **Augments** the LLM's prompt with that retrieved context
3. **Generates** an answer grounded in real, verifiable data

This eliminates hallucinations and allows AI to answer questions about *any* domain — not just what exists in training data.

---

## Why Hybrid? The Limitation of Standard RAG

Most RAG systems use only **vector search** — converting text into numerical embeddings and finding the closest matches to a query. This is powerful, but it has a fundamental blind spot: **it cannot follow relationships between entities**.

If Napoleon's "Death" section is retrieved, a pure vector search doesn't know to also pull in information about Talleyrand (who was historically connected) or the Battle of Waterloo (which led to Napoleon's exile). It retrieves semantically similar text, but misses the *structural context*.

**This project solves that** by combining two retrieval strategies:

| Strategy | Technology | What It Finds |
|----------|-----------|---------------|
| **Semantic Search** | Pinecone (vector DB) | Text chunks *semantically similar* to the query |
| **Graph Traversal** | Neo4j (graph DB) | *All entities and sections structurally connected* to the matched entity |

Together, every retrieval returns not just a relevant text chunk, but the full neighborhood of related entities — giving the LLM far richer context to reason from.

---

## Architecture

```
                   ┌──────────────────────────────────┐
                   │         JSON Data Files           │
                   │  Napoleon · Talleyrand · Waterloo │
                   └──────────────┬───────────────────┘
                                  │
           ┌──────────────────────┴────────────────────────┐
           │                                               │
           ▼                                               ▼
  ┌─────────────────────┐                    ┌──────────────────────────┐
  │    Neo4j Ingest      │                    │    Pinecone Ingest        │
  │                     │                    │                          │
  │  Person nodes       │                    │  Split text into chunks  │
  │  Event nodes        │                    │  Embed via OpenAI        │
  │  Section nodes      │                    │  Store vectors +         │
  │  RELATED_TO rels    │                    │  metadata in index       │
  │  HAS_SECTION rels   │                    │                          │
  └─────────────────────┘                    └──────────────────────────┘

                         ─── Query Time ───

                   ┌──────────────────────┐
                   │    User Question      │
                   │  "Who killed Napoleon?"
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │  Embed with OpenAI   │
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │  Pinecone Query      │  ──► top-k nearest chunks
                   │  (vector similarity) │      + metadata["name"]
                   └──────────┬───────────┘
                              │  name = "Napoleon_info"
                   ┌──────────▼───────────┐
                   │  Neo4j Cypher Query  │  ──► all connected nodes
                   │  MATCH (n)-[r]-(m)   │      & relationships
                   └──────────┬───────────┘
                              │
                   ┌──────────▼──────────────────────────┐
                   │  Combined Result                     │
                   │  {                                   │
                   │    score: 0.57,                      │
                   │    metadata: {                       │
                   │      text: "Napoleon died of...",    │
                   │      section: "Death"                │
                   │    },                                │
                   │    neo4j_nodes: [                    │
                   │      Napoleon ↔ Talleyrand,          │
                   │      Napoleon ↔ Battle of Waterloo,  │
                   │      Napoleon → "Death" section      │
                   │    ]                                 │
                   │  }                                   │
                   └──────────────────────────────────────┘
```

---

## The Jupyter Notebook (`main.ipynb`)

The notebook is the **main entry point** for understanding and running the system. It is written with detailed explanations for learners at every step.

### Step 1 — Connect to Databases
Establishes connections to both Neo4j (via LangChain's `Neo4jGraph` wrapper) and Pinecone using credentials from a `.env` file. This is the foundation everything else builds on.

### Step 2 — Define Data Sources
Specifies the three JSON files to process. Each file is structured as `{section_name: section_text}` — a dictionary mapping topic sections to their text content.

### Step 3 — Ingest into Neo4j: Create Nodes
For each entity, two node types are created in Neo4j:
- A **main node** labeled `Person` or `Event` (e.g. `Napoleon_info :Person`)
- **Section nodes** labeled `Section`, one per JSON key (e.g. `{type: "Death", parent_name: "Napoleon_info"}`)

Uses Cypher's `MERGE` command (like "INSERT IF NOT EXISTS") so re-running is always safe.

### Step 4 — Create Relationships in the Graph
Connects nodes with directed, typed edges:

| Relationship | Example |
|---|---|
| `Person RELATED_TO Person` | Napoleon ↔ Talleyrand |
| `Person RELATED_TO Event` | Napoleon ↔ Battle of Waterloo |
| `Person HAS_SECTION Section` | Napoleon → "Death" section |
| `Event HAS_SECTION Section` | Battle of Waterloo → "Commanders" section |

All relationships are bidirectional so graph traversal works from any starting node.

### Step 5 — Ingest into Pinecone: Vector Embeddings
For each JSON file:
1. Text is split into ~400-character chunks with 100-character overlap (to preserve context at boundaries)
2. Each chunk is embedded using OpenAI's `text-embedding-3-small` model (1536-dimensional vectors)
3. Vectors are upserted into Pinecone with metadata: entity name, section, chunk index, and original text

**Total vectors inserted:** 755 (121 Talleyrand + 316 Napoleon + 318 Waterloo)

### Step 6 — Define the Graph Traversal Query
A parameterized Cypher query that, given an entity name, returns its entire **1-hop graph neighborhood** — all directly connected nodes and their relationships. This is what enriches each vector search result with structural context.

### Step 7 — Run Hybrid Search
Brings everything together: embeds the user's question → finds top-k similar chunks in Pinecone → fetches the graph neighborhood of each matched entity from Neo4j → returns rich, combined results.

**Example output for `"Who killed Napoleon?"`:** Returns chunks from the "Death" section with similarity scores of ~0.57, plus all graph connections to Napoleon (Talleyrand, Battle of Waterloo, and all section nodes).

---

## Project Structure

```
vector-graph-rag/
├── main.ipynb                    # Annotated walkthrough notebook (start here)
├── data/
│   ├── Napoleon.json             # Napoleon Bonaparte (Person)
│   ├── Talleyrand.json           # Charles de Talleyrand (Person)
│   └── Battle_of_Waterloo.json  # Battle of Waterloo (Event)
├── config/
│   ├── neo4j.py                  # Neo4j connection factory
│   ├── pinecone_cfg.py           # Pinecone client factory
│   └── llm.py                    # OpenAI client setup
├── ingest/
│   ├── neo4j.py                  # create_nodes() — populates the graph
│   └── pinecone_ingest.py        # process_and_upsert_files() — chunks + embeds
├── retrieve/
│   └── neo4j_pinecone.py         # search_and_fetch() — hybrid retrieval
└── pyproject.toml                # Dependencies (managed by uv)
```

---

## Why This Approach Is Valuable

### For learners
- See RAG work end-to-end in a minimal, readable codebase
- Understand how vector databases and graph databases complement each other
- Learn to connect to and query real production-grade tools (Pinecone, Neo4j, OpenAI)
- Build intuition for when to use each retrieval strategy

### For practitioners
- A clean reference implementation of hybrid graph-RAG
- Easily adaptable: swap the JSON files for your own domain data
- The retrieval layer (`retrieve/neo4j_pinecone.py`) can plug into any LLM chain

### For researchers
- A concrete starting point for exploring graph-RAG architectures
- Natural extension paths: multi-hop traversal, entity extraction from raw text, re-ranking, answer generation with an LLM

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A **Neo4j** instance — [Neo4j Desktop](https://neo4j.com/download/) (local) or [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/) (free cloud tier)
- A **Pinecone** account with an index — [free tier available](https://www.pinecone.io/)
- An **OpenAI** API key for generating embeddings

---

## Setup

**1. Install dependencies:**
```bash
uv sync
```

**2. Create a `.env` file in the project root:**
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
PINECONE_API_KEY=your_pinecone_key
PINECONE_HOST=your_index_host_url
OPENAI_API_KEY=your_openai_key
```

**3. Open the notebook:**
```bash
uv run jupyter notebook main.ipynb
```

Run all cells from top to bottom. Each cell is annotated with explanations of what it does and why.

---

## Technology Stack

| Tool | Role |
|------|------|
| [Pinecone](https://www.pinecone.io/) | Vector database — stores and searches embeddings |
| [Neo4j](https://neo4j.com/) | Graph database — stores entities and their relationships |
| [OpenAI](https://platform.openai.com/) | Embedding model (`text-embedding-3-small`) |
| [LangChain](https://python.langchain.com/) | Neo4j graph client and text splitting utilities |
| [uv](https://docs.astral.sh/uv/) | Python package and project manager |
