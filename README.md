# 🎬 ML Recommendation Service (Semantic + Personalized)

## 📌 Overview

This ML service powers a **movie recommendation system** using:

- Semantic search (text embeddings)
- Personalized recommendations (user embeddings)
- Vector similarity search (FAISS)

It is designed as a **scalable microservice** that integrates with a backend (Spring Boot) and can be extended to **multimodal systems (images, jobs, etc.)**.

---

# 🧠 Key Features

## 🔍 1. Semantic Search

- Converts user query → embedding
- Retrieves top-K similar movies using FAISS
- Supports natural language queries like: “movies like interstellar”, “horror movies like conjuring”


---

## 👤 2. Personalized Recommendations

- Builds user embeddings using:
  - Ratings
  - Clicks
  - Watch history
  - Recency
  - Genre preferences

- Combines:Interaction Vector + Preference Vector → User Embedding

-  Retrieves recommendations via vector similarity

---

## ⚖️ 3. Hybrid Ranking

- Semantic similarity (embedding-based)
- Content-type filtering (movies vs documentaries)
- Balanced retrieval (avoids bias in results)

---

# 🏗️ Architecture
Client (Spring Boot)
↓
Flask ML Service
↓
Embeddings Model (Sentence Transformers)
↓
FAISS Index (Vector Search)
↓
Movie Metadata (Pandas / CSV)

---

# ⚙️ Tech Stack

- Python 3.10+
- Flask
- Sentence Transformers (MiniLM)
- FAISS (vector search)
- NumPy / Pandas

---

# 🚀 API Endpoints

## 🔍 Semantic Search

### Endpoint:
POST /recommend/search
### Request:
```json
{
  "query": "movies like interstellar"
}
```
### Response:
```json
{
   "movieIds": [123, 456, 789]
}
```
👤 User Recommendations
Endpoint:
POST /recommend/user
## Request:
```json
{
  "user": {
    "interactions": [...],
    "preferences": {
      "genres": ["Sci-Fi", "Action"]
    }
  }
}
```
### Response:
```json
{
  "movieIds": [123, 456, 789]
}
```

🧠 Core Concepts

1. Embeddings
	•	Text → vector representation (384D)
	•	Similar meaning → similar vectors
2. Cosine Similarity
	•	Used via dot product (normalized vectors)
score = embedding @ query_vector

3. FAISS Index
	•	Stores embeddings
	•	Enables fast nearest neighbor search
index.search(query, k)

4. Interaction Vector

Weighted combination of user history:
User = Σ (movie_embedding × interaction_weight)

5. Preference Vector

Genre-based vector:
Preference = Σ (genre_weight × genre_centroid)

6. Final User Embedding
  User Embedding =
    α × interaction_vector
  + (1 - α) × preference_vector


🧪 Running the Service

1. Setup environment
  • python3 -m venv venv
  • source venv/bin/activate
  • pip install -r requirements.txt

2. Start Flask server
   python app.py
   
4. Server runs on:
   http://localhost:4400

🧠 Design Principles
	•	Load model + index once at startup
	•	Keep ML logic separate from backend
	•	Preserve ranking across services
	•	Use hybrid retrieval (semantic + rules)

