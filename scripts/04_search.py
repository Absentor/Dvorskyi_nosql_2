# scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract

# Функція кодування пошукового запиту
def encode_query(query):
    embedding = model.encode(
        query,
        normalize_embeddings=True
    )
    return embedding.tolist()

# Функція семантичного пошуку
def semantic_search(query):
    query_embedding = encode_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True
    )

    print(f"\nПошуковий запит: {query}")
    print("\nТоп-5 найбільш релевантних статей:\n")

    for i, match in enumerate(results["matches"], 1):
        metadata = match["metadata"]

        print(f"{i}. {metadata['title']}")
        print(f"   Категорія: {metadata['category']}")
        print(f"   Рік: {metadata['year']}")
        print(f"   Схожість: {match['score']:.4f}")
        print(f"   Abstract: {metadata['abstract'][:250]}...")
        print()

# Виконуємо семантичний пошук
query = "teaching machines to recognize objects in pictures"
semantic_search(query)

# Пошук із фільтрацією

# Функція пошуку з фільтрами
def filtered_search(query, filters):
    query_embedding = encode_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        filter=filters,
        include_metadata=True
    )

    print(f"\nЗапит: {query}")
    print(f"Фільтри: {filters}")
    print(f"Знайдено статей: {len(results['matches'])}\n")

    for i, match in enumerate(results["matches"], 1):
        metadata = match["metadata"]

        print(f"{i}. {metadata['title']}")
        print(f"   Категорія: {metadata['category']}")
        print(f"   Рік: {metadata['year']}")
        print(f"   Схожість: {match['score']:.4f}")
        print(f"   Abstract: {metadata['abstract'][:250]}...")
        print()


# Приклад A: reinforcement learning за останні 5 років, категорія cs.LG
filtered_search(
    "reinforcement learning",
    {
        "year": {"$gte": 2021},
        "category": {"$eq": "cs.LG"}
    }
)

# Приклад B: статті до 2015 року, будь-яка категорія
filtered_search(
    "machine learning",
    {
        "year": {"$lt": 2015}
    }
)

# Порівняння різних метрик схожості

embeddings = np.load("embeddings/embeddings.npy")

query = "teaching machines to recognize objects in pictures"
query_embedding = model.encode(query, normalize_embeddings=True)

# Cosine Similarity
cosine_scores = np.dot(embeddings, query_embedding)
cosine_top = np.argsort(cosine_scores)[::-1][:5]

# Dot Product
dot_scores = np.dot(embeddings, query_embedding)
dot_top = np.argsort(dot_scores)[::-1][:5]

# L2 Distance
l2_scores = np.linalg.norm(embeddings - query_embedding, axis=1)
l2_top = np.argsort(l2_scores)[:5]


def print_results(metric_name, top_indices, scores):
    print(f"\n{metric_name}")
    print("-" * 50)

    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. {df.iloc[idx]['title']}")
        print(f"   Score: {scores[idx]:.4f}")


print_results("Cosine Similarity", cosine_top, cosine_scores)
print_results("Dot Product", dot_top, dot_scores)
print_results("L2 Distance", l2_top, l2_scores)