from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import csv

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
print(" Загружаем модель...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print(" Модель загружена")
corpus_texts = []
corpus_metadata = []
with open("corpus.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader) 
    for row in reader:
        corpus_texts.append(row[1]) 
        corpus_metadata.append(row[2])  
print(f" Загружено {len(corpus_texts)} слов")
print(" Вычисляем эмбеддинги (1 раз)...")
corpus_embeddings = model.encode(corpus_texts, convert_to_numpy=True)
print(" Готово! Сервер запускается...")

class SearchQuery(BaseModel):
    query: str
    max_results: int = 5

@app.post("/search")
def search(query: SearchQuery):
    query_embedding = model.encode([query.query], convert_to_numpy=True)
    
    similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
    
    top_indices = np.argsort(similarities)[-query.max_results:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            "rank": len(results),
            "text": corpus_texts[idx],
            "metadata": corpus_metadata[idx],
            "similarity": float(similarities[idx])
        })
    
    return {"query": query.query, "results": results}
@app.get("/")
def root():
    return {"message": "TUV.WORLD Semantic Search API работает!", "words_count": len(corpus_texts)}
if __name__ == "__main__":
    import uvicorn
    print(" Сервер запущен на http://localhost:8001")
    print(" Документация API: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)