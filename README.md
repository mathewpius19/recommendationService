#  ML Recommendation Service (Semantic + Personalized)

##  Overview

This ML service powers a **movie recommendation system** using:

- Semantic search (text embeddings)
- Personalized recommendations (user embeddings)
- Vector similarity search (FAISS)

It is designed as a **scalable microservice** that integrates with a backend (Spring Boot) and can be extended to **multimodal systems (images, jobs, etc.)**.

This project was built to explore how modern recommendation systems
combine semantic search with user behavior to deliver personalized results.

Instead of relying on traditional collaborative filtering, this system uses
vector embeddings and hybrid ranking to simulate real-world ML pipelines.

---

# Key Features

## 1. Semantic Search

- Converts user query → embedding
- Retrieves top-K similar movies using FAISS
- Supports natural language queries like: “movies like interstellar”, “horror movies like conjuring”


---

## 2. Personalized Recommendations

- Builds user embeddings using:
  - Ratings
  - Clicks
  - Watch history
  - Recency
  - Genre preferences

- Combines:Interaction Vector + Preference Vector → User Embedding

-  Retrieves recommendations via vector similarity

---

## 3. Hybrid Ranking

- Semantic similarity (embedding-based)
- Content-type filtering (movies vs documentaries)
- Balanced retrieval (avoids bias in results)

---

#  Architecture
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

#  Tech Stack

- Python 3.10+
- Flask
- Sentence Transformers (MiniLM)
- FAISS (vector search)
- NumPy / Pandas

---

#  API Endpoints

##  Semantic Search

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
 User Recommendations
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

## Challenges solved

- Handled semantic drift (e.g. horror vs thriller overlap)

- Balanced recommendation bias across genres

- Preserved ranking across microservices (ML → backend → UI)

- Designed user embedding system combining behavior + preferences

