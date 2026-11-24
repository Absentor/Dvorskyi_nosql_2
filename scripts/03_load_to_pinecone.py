# scripts/03_load_to_pinecone.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс (якщо не існує)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

print(f"Індекс {INDEX_NAME} готовий до роботи")

# Завантажуємо датасет та ембеддинги
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

# Завантажуємо вектори батчами
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження в Pinecone"):
    batch = []

    for j in range(i, min(i + BATCH_SIZE, len(df))):
        batch.append({
            "id": f"paper_{j}",
            "values": embeddings[j].tolist(),
            "metadata": {
                "arxiv_id": str(df.iloc[j]["id"]),
                "title": str(df.iloc[j]["title"]),
                "abstract": str(df.iloc[j]["abstract"])[:500],
                "authors": str(df.iloc[j]["authors"])[:200],
                "year": int(df.iloc[j]["year"]),
                "category": str(df.iloc[j]["category"])
            }
        })

    index.upsert(vectors=batch)

# Перевіряємо кількість векторів
print(f"\nКількість векторів в індексі: {index.describe_index_stats().total_vector_count}")
