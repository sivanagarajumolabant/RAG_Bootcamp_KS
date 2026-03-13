from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os
from config.llm import client
from config.pinecone_cfg import get_pinecone

pc = get_pinecone()
index = pc.Index(host=os.getenv("PINECONE_HOST"))


def process_and_upsert_files(
    file_names: list[str],
    chunk_size: int = 400,
    chunk_overlap: int = 100
) -> dict[str, int]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    summary: dict[str, int] = {}

    for name in file_names:
        file = f"data/{name}.json"
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        vectors: list[dict] = []

        for section_name, section_text in data.items():
            chunks = text_splitter.split_text(section_text)

            for i, chunk in enumerate(chunks):
                embedding = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                ).data[0].embedding

                vector_id = f"{name}_{section_name}_{i}"

                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "name": f"{name}_info",
                        "section": section_name,
                        "chunk_index": i,
                        "text": chunk
                    }
                })

        index.upsert(vectors=vectors)
        print(f"Inserted {len(vectors)} chunks from {file}")
        summary[name] = len(vectors)

    return summary
