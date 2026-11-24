# scripts/02_embed.py

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

# Завантажуємо датасет
df = pd.read_parquet(INPUT_FILE)

# Об'єднуємо назву та анотацію
texts = (
    df["title"] + " [SEP] " + df["abstract"]
).tolist()

# Завантажуємо модель
model = SentenceTransformer(MODEL_NAME)

# Створюємо ембеддинги
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Виводимо інформацію
print(f"\nКількість оброблених текстів: {len(texts)}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.4f}")

# Створюємо папку для збереження
os.makedirs("embeddings", exist_ok=True)

# Зберігаємо ембеддинги
np.save(OUTPUT_FILE, embeddings)

print(f"\nЕмбеддинги збережено в {OUTPUT_FILE}")

