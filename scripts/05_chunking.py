# scripts/05_chunking.py

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
BATCH_SIZE = 200
CHUNK_SIZE = 100
OVERLAP = 20

INDEX_NAME_FIXED = "arxiv-chunks-fixed"
INDEX_NAME_SEMANTIC = "arxiv-chunks-semantic"

# Підключення до Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# Вибираємо 30 найдовших abstract
df["abstract_length"] = df["abstract"].str.len()
longest_papers = df.nlargest(30, "abstract_length")

print(f"Відібрано статей: {len(longest_papers)}")
print(
    f"Середня довжина abstract: "
    f"{longest_papers['abstract_length'].mean():.0f} символів"
)


# Fixed-size chunking: розбиття за кількістю слів
def fixed_size_chunk(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks


fixed_chunks = []

for _, row in longest_papers.iterrows():
    chunks = fixed_size_chunk(row["abstract"])

    for i, chunk in enumerate(chunks):
        fixed_chunks.append({
            "arxiv_id": str(row["id"]),
            "title": row["title"],
            "chunk_id": i,
            "text": chunk,
            "year": int(row["year"]),
            "category": row["category"]
        })

print(f"\nКількість Fixed-size chunks: {len(fixed_chunks)}")
print(f"Приклад chunk:\n{fixed_chunks[0]['text']}")


# Semantic chunking: об'єднання цілих речень
def semantic_chunk(text, max_chunk_size=CHUNK_SIZE):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()

        if not sentence_words:
            continue

        if current_word_count + len(sentence_words) <= max_chunk_size:
            current_chunk.append(sentence)
            current_word_count += len(sentence_words)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            current_chunk = [sentence]
            current_word_count = len(sentence_words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


semantic_chunks = []

for _, row in longest_papers.iterrows():
    chunks = semantic_chunk(row["abstract"])

    for i, chunk in enumerate(chunks):
        semantic_chunks.append({
            "arxiv_id": str(row["id"]),
            "title": row["title"],
            "chunk_id": i,
            "text": chunk,
            "year": int(row["year"]),
            "category": row["category"]
        })

print(f"\nКількість Semantic chunks: {len(semantic_chunks)}")
print(f"Приклад semantic chunk:\n{semantic_chunks[0]['text']}")


# Створення індексів Pinecone
for index_name in [INDEX_NAME_FIXED, INDEX_NAME_SEMANTIC]:
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    print(f"Індекс {index_name} готовий до роботи")


# Векторизація та завантаження чанків
def upload_chunks(chunks, index_name):
    index = pc.Index(index_name)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    for i in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        desc=f"Завантаження {index_name}"
    ):
        batch = []

        for j in range(i, min(i + BATCH_SIZE, len(chunks))):
            batch.append({
                "id": f"chunk_{j}",
                "values": embeddings[j].tolist(),
                "metadata": {
                    "arxiv_id": chunks[j]["arxiv_id"],
                    "title": chunks[j]["title"],
                    "text": chunks[j]["text"],
                    "chunk_id": chunks[j]["chunk_id"],
                    "year": chunks[j]["year"],
                    "category": chunks[j]["category"]
                }
            })

        index.upsert(vectors=batch)

    print(
        f"Завантажено в {index_name}: "
        f"{index.describe_index_stats().total_vector_count}"
    )


upload_chunks(fixed_chunks, INDEX_NAME_FIXED)
upload_chunks(semantic_chunks, INDEX_NAME_SEMANTIC)


# Пошук по чанках
def search_chunks(query, index_name):
    index = pc.Index(index_name)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True
    )

    print(f"\nРезультати для {index_name}")
    print(f"Запит: {query}")

    for i, match in enumerate(results["matches"], 1):
        metadata = match["metadata"]

        print(f"\n{i}. {metadata['title']}")
        print(f"   Score: {match['score']:.4f}")
        print(f"   Chunk: {metadata['text'][:250]}")


# Три тестові запити
queries = [
    "machine learning methods for image classification",
    "neural networks and deep learning",
    "computer vision and object recognition"
]

for query in queries:
    search_chunks(query, INDEX_NAME_FIXED)
    search_chunks(query, INDEX_NAME_SEMANTIC)
