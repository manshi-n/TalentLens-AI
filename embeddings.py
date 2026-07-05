"""
embeddings.py
Fast TF-IDF semantic scoring for the Streamlit app.
Uses 8192 features (not 512 — that was the bug causing slow/poor scores).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def semantic_scores(jd_text: str, candidate_texts: list) -> list:
    """
    Returns cosine-similarity scores in [0, 1], one per candidate.
    Uses sparse matrix ops — no .toarray() on full matrix.
    Fast for 100k candidates.
    """
    if not candidate_texts:
        return []

    all_texts = [jd_text] + candidate_texts
    vec = TfidfVectorizer(
        max_features=8192,       # was 512 — that was the bug
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
        lowercase=True,
    )
    mat   = vec.fit_transform(all_texts)
    mat_n = normalize(mat, norm="l2")
    jd_vec = mat_n[0]
    scores = np.asarray(mat_n[1:].dot(jd_vec.T).todense()).flatten()
    return [float(max(0.0, min(1.0, s))) for s in scores]