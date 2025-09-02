## 📌 Project Overview
This project clusters **unrecognized user requests** in goal‑oriented dialog systems. It was built as a **Natural Language Processing (NLP) final project** by **Omer Blau** and **Ofek Cohen**.

**Pipeline:**  
1) **Clean & preprocess text** (configurable)  
2) **Generate embeddings** with SentenceTransformer (`all-MiniLM-L6-v2`)  
3) **Cluster** by iterative thresholded assignment (Euclidean distance)  
4) **Name clusters** with KeyBERT (TF‑IDF fallback)  
5) **Pick Top‑K representative sentences** per cluster

Evaluated on two datasets: **Banking** and **COVID‑19**.


## ⚙️ Data Processing
Configurable cleaning parameters:
```python
params = {
  "expand contractions": True,
  "remove special": True,
  "tokenize": False,
  "remove stop words": False,
  "lemmatize": False
}
```
- **Embeddings:** `SentenceTransformer (all-MiniLM-L6-v2)` + normalization  
- **Stopwords:** NLTK (optional)  
- **Contractions:** `contractions` lib


## 🔎 Clustering
Threshold‑based, iterative approach:
- Start with **no clusters**; assign a request to an existing cluster if distance ≤ **threshold** (used: **0.87**); otherwise create a new cluster.
- Recompute **centroids** after each pass (mean embedding).
- Stop when max iters reached or changes < **early_stop_ratio** (used: **0.0001**).

**Core structures:**  
`assignments` (request → cluster id), `clusters` (id → indices), `centroids` (id → mean vector).


## 🏷️ Cluster Naming
Tried multiple strategies; final choice is **KeyBERT** with **TF‑IDF** fallback:
- Extract short, meaningful keyphrases with KeyBERT.  
- If none fit constraints, back off to TF‑IDF top terms.


## 📝 Top‑K Representative Sentences
- Compute cluster **centroid**.  
- Rank sentences by **Euclidean distance** to centroid.  
- Enforce **diversity** via cosine‑similarity cap (**0.93**).  
- Pick **K closest** that aren’t near‑duplicates.


## 📊 Results (runtime)
- **COVID‑19:** ~**42.79s**  
- **Banking:** ~**137.87s**


## 👩‍💻 Authors
- **Omer Blau**  
- **Ofek Cohen**
