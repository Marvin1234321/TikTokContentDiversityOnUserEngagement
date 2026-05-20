import pandas as pd
import numpy as np

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from umap import UMAP
from hdbscan import HDBSCAN

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import entropy


# =========================
# 1. DATA INLADEN
# =========================

file_path = "output-single/dataenrichment_transcribed.xlsx"
df = pd.read_excel(file_path)


# =========================
# 2. FILTERING
# =========================

df = df[
    ~(
        ((df['video_description'].isna()) | (df['video_description'].astype(str).str.strip() == "")) &
        ((df['video_duration'].isna()) | (df['video_duration'].astype(str).str.strip() == ""))
    )
].reset_index(drop=True)

print("Aantal rijen na filtering:", len(df))


# =========================
# 3. CLEAN TEXT
# =========================

for col in ['objects', 'video_description', 'transcriptie', 'subject']:
    df[col] = df[col].fillna("").astype(str)


# ==========================================================
# 🟢 PIPELINE 1: BERTopic (ZONDER TRANSCRIPTIES)
# ==========================================================

df['text_bertopic'] = (
    "OBJECTS: " + df['subject'] +
    " DESCRIPTION: " + df['video_description']
)

embedding_model_bert = SentenceTransformer("all-MiniLM-L6-v2")

embeddings_bert = embedding_model_bert.encode(
    df['text_bertopic'].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

umap_model = UMAP(
    n_neighbors=30,
    n_components=5,
    metric="cosine"
)

hdbscan_model = HDBSCAN(
    min_cluster_size=20,
    min_samples=5,
    prediction_data=False
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=False,
    verbose=True
)

topics, _ = topic_model.fit_transform(
    df['text_bertopic'].tolist(),
    embeddings_bert
)

df['topic_bertopic'] = topics


# =========================
# 📊 BERTopic ENTROPY PER SESSIE
# =========================

df['entropy_bertopic'] = 0.0

for sessie in df['sessie_nr'].unique():

    mask = df['sessie_nr'] == sessie
    topics = df.loc[mask, 'topic_bertopic'].values

    if len(topics) == 0:
        continue

    unique, counts = np.unique(topics, return_counts=True)
    probs = counts / counts.sum()

    df.loc[mask, 'entropy_bertopic'] = entropy(probs)


# ==========================================================
# 🔵 PIPELINE 2: DISTILBERT (MET TRANSCRIPTIES)
# ==========================================================

df['text_distilbert'] = (
    "OBJECTS: " + df['objects'] +
    "DESCRIPTION: " + df['video_description'] +
    " TRANSCRIPT: " + df['transcriptie']
)

embedding_model_distil = SentenceTransformer(
    "distilbert-base-nli-stsb-mean-tokens"
)

embeddings_distil = embedding_model_distil.encode(
    df['text_distilbert'].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)


# =========================
# 📊 SEMANTIC DIVERSITY PER SESSIE
# =========================

df['semantic_diversity'] = 0.0

for sessie in df['sessie_nr'].unique():

    mask = df['sessie_nr'] == sessie
    emb = embeddings_distil[mask]

    if len(emb) <= 1:
        continue

    dist_matrix = cosine_distances(emb)
    np.fill_diagonal(dist_matrix, np.nan)

    df.loc[mask, 'semantic_diversity'] = np.nanmean(dist_matrix, axis=1)


# =========================
# 5. SCORES
# =========================

# 🔴 RAW SCORE (niet genormaliseerd)
df['content_score_raw'] = (
    0.5 * df['entropy_bertopic'] +
    0.5 * df['semantic_diversity']
)


# =========================
# 6. OPSLAAN
# =========================

output_file = "total_data.xlsx"
df.to_excel(output_file, index=False)

output_file2 = "total_data.csv"
df.to_csv(output_file2, index=False)

print("Klaar! Bestand opgeslagen als:", output_file)
