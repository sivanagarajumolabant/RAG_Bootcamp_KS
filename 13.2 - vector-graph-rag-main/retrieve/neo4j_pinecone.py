from config import neo4j
from config.llm import client
from config.pinecone_cfg import get_pinecone
import os

graph = neo4j.load_neo4j_graph()
pc = get_pinecone()
index = pc.Index(host=os.getenv("PINECONE_HOST"))


def search_and_fetch(neo4j_query: str, query_text: str, top_k: int = 2) -> list[dict]:
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    ).data[0].embedding

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    final_results: list[dict] = []

    for match in results.matches:
        score = match.score
        metadata = match.metadata

        nodes = graph.query(neo4j_query, params={"name": metadata["name"]})

        final_results.append({
            "score": score,
            "metadata": metadata,
            "neo4j_nodes": nodes
        })

    return final_results
