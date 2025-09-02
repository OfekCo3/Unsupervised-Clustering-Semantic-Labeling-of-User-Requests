#################################################################################
### Imports and libs
import contractions
import pandas as pd
import numpy as np
import logging
import torch
import time
import json
import nltk
import re

from compare_clustering_solutions import evaluate_clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from collections import defaultdict
from nltk.corpus import stopwords
from keybert import KeyBERT
from tqdm import tqdm

# Download required NLTK datasets quietly.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# Use GPU if available, otherwise fallback to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load the sentence transformer model and initialize KeyBERT.
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
kw_model = KeyBERT(model=model)


#################################################################################
### Data preperation
def clean_text(
    text, expand_contractions=True, remove_special=True, tokenize=True, remove_stopwords=True, lemmatize=True
):
    """
    Clean and preprocess the input text by optionally expanding contractions, removing special characters,
    tokenizing, removing stopwords, and lemmatizing.
    """
    # Expand contractions and remove special characters if requested.
    if expand_contractions:
        text = contractions.fix(text)
    if remove_special:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Tokenization is required for stopword removal or lemmatization.
    if not tokenize and (remove_stopwords or lemmatize):
        print("tokenize must be True for stopwords/lemmatization; defaulting to False.")
        return text

    tokens = nltk.word_tokenize(text)
    if remove_stopwords:
        tokens = [word for word in tokens if word not in STOP_WORDS]
    if lemmatize:
        lemmatizer = nltk.WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def embed_text(text, batch_size=128):
    """
    Encode the input text into an embedding using the pre-loaded model.
    """
    if not text:
        return None
    # Encode text into embeddings and average them.
    sentence_embeddings = model.encode([text], batch_size=batch_size, show_progress_bar=False)
    return np.mean(sentence_embeddings, axis=0)


def compute_embedding(text, normalize=False):
    """
    Compute the embedding for the given text and normalize it if specified.
    """
    emb = embed_text(text)
    if emb is None:
        return None
    if normalize:
        norm = np.linalg.norm(emb)
        emb = emb / norm if norm != 0 else emb
    return emb.tolist()


def read_requests(
    data_file, expand_contractions=True, remove_special=True, tokenize=True, remove_stopwords=True, lemmatize=True
):
    """
    Read a CSV file containing requests, clean the text data, and compute embeddings.
    """
    # Read the first two columns from CSV and rename them.
    df = pd.read_csv(data_file, usecols=[0, 1])
    df.columns = ["id", "text"]

    # Replace missing or empty texts and convert to lowercase.
    df["text"] = df["text"].apply(lambda x: x if isinstance(x, str) and x.strip() != "" else "empty").str.lower()

    tqdm.pandas(desc="text cleaning".ljust(25))
    df["cleaned_text"] = df["text"].progress_apply(
        lambda x: clean_text(
            x,
            expand_contractions=expand_contractions,
            remove_special=remove_special,
            tokenize=tokenize,
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize,
        )
    )

    tqdm.pandas(desc="embedding".ljust(25))
    df["embedding"] = df["cleaned_text"].progress_apply(lambda x: compute_embedding(x, normalize=True))

    return df


#################################################################################
### Clustering
def euclidean_distance(vec1, vec2):
    """
    Compute the Euclidean distance between two vectors.
    """
    return np.linalg.norm(vec1 - vec2)


def update_centroids(clusters, embeddings):
    """
    Update centroids by calculating the mean embedding for each non-empty cluster.
    """
    centroids = {}
    for cid, indices in clusters.items():
        if indices:  # Update only non-empty clusters.
            centroids[cid] = np.mean([embeddings[i] for i in indices], axis=0)
    return centroids


def cluster_requests(embeddings, threshold=1.0, max_iter=10, early_stop_ratio=0.01):
    """
    Cluster the input embeddings using iterative refinement.
    Initial assignments are made based on a distance threshold, and clusters are updated until
    convergence or a maximum number of iterations is reached.
    """
    n = len(embeddings)
    assignments = [-1] * n  # Cluster id for each embedding.
    clusters = {}  # Map: cluster_id -> list of embedding indices.
    centroids = {}  # Map: cluster_id -> centroid vector.
    next_cluster_id = 0

    # INITIAL ASSIGNMENT: assign to an existing cluster if within threshold, else create a new cluster.
    for i, emb in enumerate(embeddings):
        assigned = False
        for cid, centroid in centroids.items():
            if euclidean_distance(emb, centroid) <= threshold:
                assignments[i] = cid
                clusters[cid].append(i)
                assigned = True
                break
        if not assigned:
            cid = next_cluster_id
            next_cluster_id += 1
            assignments[i] = cid
            clusters[cid] = [i]
            centroids[cid] = emb

    centroids = update_centroids(clusters, embeddings)

    # ITERATIVE REFINEMENT of clusters.
    progress = tqdm(total=max_iter, desc="Refining clusters".ljust(25))
    for it in range(max_iter):
        changed = 0
        new_clusters = {cid: [] for cid in clusters.keys()}
        for i, emb in enumerate(embeddings):
            current_cid = assignments[i]
            best_cid = current_cid
            best_distance = euclidean_distance(emb, centroids[current_cid])
            # Check if another cluster is a better fit.
            for cid, centroid in centroids.items():
                d = euclidean_distance(emb, centroid)
                if d < best_distance and d <= threshold:
                    best_distance = d
                    best_cid = cid
            if best_cid != current_cid:
                changed += 1
            assignments[i] = best_cid
            new_clusters.setdefault(best_cid, []).append(i)
        clusters = new_clusters
        centroids = update_centroids(clusters, embeddings)

        progress.set_postfix({"Iteration": it + 1, "Changed": changed})
        progress.update(1)

        if changed < early_stop_ratio * n:
            break

    progress.close()
    return clusters


def format_clusters(df, clusters, min_size):
    """
    Format clusters by grouping texts from the DataFrame and separating clusters smaller than the minimum size.
    """
    cluster_list = []
    unclustered = []
    for cid, indices in clusters.items():
        texts = [df.iloc[i]["text"] for i in indices]
        if len(texts) >= min_size:
            cluster_list.append({"cluster_name": str(cid), "requests": texts})
        else:
            unclustered.extend(texts)
    return {"cluster_list": cluster_list, "unclustered": unclustered}


def write_results(results, output_file):
    """
    Write clustering results to a JSON file with pretty formatting.
    """
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Clustering results saved to {output_file}")


#################################################################################
### Naming the clusters
def get_top_tfidf_terms(texts, top_n=3):
    """
    Generate a label for a cluster by extracting the top TF-IDF terms from a list of texts.
    """
    if not texts:
        return "undefined"
    
    try:
        vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)  # No stopword removal
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        if len(feature_names) == 0:
            print(f"Warning: Empty vocabulary for texts: {texts}")
            return "empty"
    
    except ValueError:
        return "empty"
    
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-top_n:][::-1]
    top_terms = [feature_names[i] for i in top_indices]
    
    return " ".join(top_terms[:3]) if top_terms else "undefined"


def get_keybert_phrase(texts, min_chars=5, max_words=3, kw_model=None):
    """
    Extract a keyphrase from the provided texts using KeyBERT, ensuring it meets
    min_chars and max_words requirements.
    """
    if not texts:
        return "undefined"

    if kw_model is None:
        kw_model = KeyBERT()
    
    combined_text = " ".join(texts)
    keyphrases = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(1, 3), stop_words=None, top_n=10)  # No stopword removal
    
    # Filter phrases based on character and word length constraints
    valid_phrases = [
        phrase for phrase, score in keyphrases if len(phrase) >= min_chars and len(phrase.split()) <= max_words
    ]
    
    if valid_phrases:
        valid_phrases.sort(key=lambda x: len(x.split()))  # Prefer shorter phrases
        return valid_phrases[0]
    
    return "undefined"


def rename_clusters_keyphrase(clusters, df, min_size, min_chars=5, max_words=3, kw_model=None):
    """
    Rename clusters using keyphrases extracted from their texts.
    Clusters that do not meet the minimum size are treated as unclustered.
    """
    cluster_list = []
    unclustered = []
    
    if kw_model is None:
        kw_model = KeyBERT()

    for cid, indices in tqdm(clusters.items(), desc="Naming clusters".ljust(25)):
        texts = [df.iloc[i]["text"] for i in indices]

        if len(texts) >= int(min_size):
            keyphrase = get_keybert_phrase(texts, min_chars, max_words, kw_model=kw_model)

            if keyphrase == "undefined":
                keyphrase = get_top_tfidf_terms(texts)

            cluster_list.append({"cluster_name": keyphrase, "requests": texts, "indices": indices})
        else:
            unclustered.extend(texts)
    
    return {"cluster_list": cluster_list, "unclustered": unclustered}


#################################################################################
### Top K representives
def add_top_k_representative_sentences(results, df, precomputed_embeddings, num_representatives, similarity_threshold=0.93):
    """Add the top-K representative sentences to each cluster based on proximity to the centroid and semantic diversity."""
    for cluster in tqdm(results["cluster_list"], desc="Selecting representatives".ljust(25)):
        indices = cluster.get("indices", [])
        sentence_emb_pairs = [(df.iloc[i]["text"], precomputed_embeddings[i]) for i in indices if precomputed_embeddings[i] is not None]

        if sentence_emb_pairs:
            cluster_embeddings = [emb for _, emb in sentence_emb_pairs]
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = [(sent, np.linalg.norm(centroid - emb), emb) for sent, emb in sentence_emb_pairs]
            distances.sort(key=lambda x: x[1])

            selected_sentences = []
            selected_embeddings = []
            seen_sentences = set()

            # Step 1: Select diverse representatives using similarity threshold
            for sent, _, emb in distances:
                if sent.strip() and sent not in seen_sentences:
                    if not selected_embeddings or all(util.pytorch_cos_sim(emb, e)[0][0].item() < similarity_threshold for e in selected_embeddings):
                        selected_sentences.append(sent)
                        selected_embeddings.append(emb)
                        seen_sentences.add(sent)
                if len(selected_sentences) == num_representatives:
                    break

            # Step 2: Ensure exactly K representatives by backfilling if needed
            for sent, _, emb in distances:
                if len(selected_sentences) >= num_representatives:
                    break
                if sent not in seen_sentences:
                    selected_sentences.append(sent)
                    seen_sentences.add(sent)

            cluster["representatives"] = selected_sentences[:num_representatives]
        else:
            cluster["representatives"] = []
            logging.warning(f"Cluster '{cluster['cluster_name']}' has no embeddings and thus no representatives.")


def write_results_with_representatives(results, output_file):
    """
    Write the results with representatives to a JSON file, ensuring indices are removed from clusters.
    """
    for cluster in results["cluster_list"]:
        cluster.pop("indices", None)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def print_params(params, threshold, max_iter, early_stop_ratio):
    print(f"\n{threshold = }, {max_iter = }, {early_stop_ratio = }")
    print("Preprocessing Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")


#################################################################################
### Main
BATCH_SIZE = 128


def analyze_unrecognized_requests(data_file, output_file, num_representatives, min_size):
    threshold = 0.87
    max_iter = 30
    early_stop_ratio = 0.0001

    params = {
        "expand_contractions": True,
        "remove_special": True,
        "tokenize": False,
        "remove_stopwords": False,
        "lemmatize": False,
    }

    # Load and preprocess the data
    df = read_requests(
        data_file,
        expand_contractions=params["expand_contractions"],
        remove_special=params["remove_special"],
        tokenize=params["tokenize"],
        remove_stopwords=params["remove_stopwords"],
        lemmatize=params["lemmatize"],
    )

    embeddings = [np.array(e) for e in df["embedding"].tolist()]

    clusters = cluster_requests(embeddings, threshold=threshold, max_iter=max_iter, early_stop_ratio=early_stop_ratio)

    results = rename_clusters_keyphrase(clusters, df, min_size, min_chars=5, max_words=3)

    add_top_k_representative_sentences(results, df, embeddings, int(num_representatives))

    write_results_with_representatives(results, output_file)

    print_params(params, threshold, max_iter, early_stop_ratio)


if __name__ == "__main__":
    start_time = time.time()
    with open("config.json", "r") as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(
        config["data_file"], config["output_file"], config["num_of_representatives"], config["min_cluster_size"]
    )

    evaluate_clustering(config["example_solution_file"], config["output_file"])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExecution Time: {elapsed_time:.2f} seconds")

