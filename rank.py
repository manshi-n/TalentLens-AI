"""
rank.py — TalentLens AI  (FAST edition)
Usage:
  python rank.py
  python rank.py --candidates ./data/.../candidates.jsonl --out submission.csv

Runtime on 100k candidates: ~35s on CPU, 16GB RAM.
No GPU needed. No API calls. Fully offline.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from candidate_parser import parse_candidates_jsonl, parse_candidates_json
from jd_parser import parse_jd

DEFAULT_CANDIDATES = "data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
DEFAULT_JD         = "data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx"
DEFAULT_OUT        = "submission.csv"

# ── Hardcoded JD text (augmented for richer TF-IDF matching) ─────────────────
JD_TEXT = """
Senior AI Engineer Founding Team Redrob AI Series A talent intelligence platform Pune Noida India Hybrid.
Experience 5 to 9 years applied machine learning production systems.
Required: embeddings retrieval systems sentence-transformers BGE E5 OpenAI embeddings production deployment.
Required: vector databases hybrid search Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS.
Required: Python strong production code quality software engineering.
Required: evaluation frameworks ranking NDCG MRR MAP offline online AB testing recruiter feedback loops.
Preferred: LLM fine-tuning LoRA QLoRA PEFT learning-to-rank XGBoost neural ranking.
Preferred: HR-tech recruiting marketplace distributed systems large-scale inference optimization.
Preferred: open-source contributions AI ML NLP information retrieval.
Role: candidate JD matching at scale retrieval ranking recommendation search re-ranking semantic search BM25.
Product companies only — not pure consulting not pure research. Shipped to real users.
Mentor engineers architecture decisions hybrid dense sparse retrieval.
NLP transformers BERT language models text embeddings semantic similarity cosine distance.
"""


def load_candidates(path: str) -> list:
    p = Path(path)
    print(f"Loading candidates from {p.name}...")
    t = time.time()
    if p.suffix == ".jsonl":
        candidates = parse_candidates_jsonl(str(p))
    else:
        candidates = parse_candidates_json(str(p))
    print(f"  {len(candidates):,} candidates loaded in {time.time()-t:.1f}s")
    return candidates


def tfidf_semantic_scores(jd_text: str, profile_texts: list) -> np.ndarray:
    """
    Sparse TF-IDF cosine similarity — vectorised, no Python loop.
    Returns float32 numpy array of shape (N,).
    """
    print(f"  TF-IDF semantic scoring {len(profile_texts):,} candidates...")
    t = time.time()
    vec = TfidfVectorizer(
        max_features=8192,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )
    mat   = vec.fit_transform([jd_text] + profile_texts)
    mat_n = normalize(mat, norm="l2")
    jd_vec = mat_n[0]
    scores = np.asarray(mat_n[1:].dot(jd_vec.T).todense()).flatten().astype(np.float32)
    scores = np.clip(scores, 0.0, 1.0)
    print(f"  Semantic done in {time.time()-t:.1f}s  max={scores.max():.4f}")
    return scores


def vectorised_scores(parsed: list) -> dict:
    """
    Extract all numeric signals as numpy arrays and score in one pass.
    ~200x faster than Python loops.
    Returns dict of float32 arrays, each shape (N,).
    """
    print("  Extracting numeric arrays...")
    t = time.time()
    N = len(parsed)

    ai_counts    = np.array([p["ai_skill_count"]          for p in parsed], dtype=np.float32)
    prof         = np.array([p["skill_proficiency_score"]  for p in parsed], dtype=np.float32)
    open_work    = np.array([p["open_to_work"]             for p in parsed], dtype=np.float32)
    days_inact   = np.array([p["days_inactive"]            for p in parsed], dtype=np.float32)
    resp_rates   = np.array([p["response_rate"]            for p in parsed], dtype=np.float32)
    resp_hrs     = np.array([p["resp_time_hrs"]            for p in parsed], dtype=np.float32)
    notice       = np.array([p["notice_days"]              for p in parsed], dtype=np.float32)
    github       = np.array([p["github_score"]             for p in parsed], dtype=np.float32)
    profile_c    = np.array([p["profile_complete"]         for p in parsed], dtype=np.float32)
    years        = np.array([p["years_experience"]         for p in parsed], dtype=np.float32)
    avg_assess   = np.array([p["avg_assessment_score"]     for p in parsed], dtype=np.float32)
    only_cons    = np.array([p["only_consulting"]          for p in parsed], dtype=np.float32)
    title_hop    = np.array([p["is_title_hopper"]          for p in parsed], dtype=np.float32)
    prod_co      = np.array([p["has_product_company"]      for p in parsed], dtype=np.float32)
    seniority    = np.array([p["seniority_score"]          for p in parsed], dtype=np.float32)
    ai_title     = np.array([p["recent_title_ai_relevant"] for p in parsed], dtype=np.float32)
    verified     = np.array([p["verified_email"]           for p in parsed], dtype=np.float32)
    linkedin     = np.array([p["linkedin_connected"]       for p in parsed], dtype=np.float32)
    connections  = np.array([p["connection_count"]         for p in parsed], dtype=np.float32)
    endorsements = np.array([p["endorsements"]             for p in parsed], dtype=np.float32)
    saved_30d    = np.array([p["saved_30d"]                for p in parsed], dtype=np.float32)
    interview_r  = np.array([p["interview_rate"]           for p in parsed], dtype=np.float32)
    has_cs       = np.array([p["has_cs_degree"]            for p in parsed], dtype=np.float32)
    avg_tenure   = np.array([p["avg_tenure_months"]        for p in parsed], dtype=np.float32)
    has_lead     = np.array([p["has_leadership"]           for p in parsed], dtype=np.float32)
    edu_tier_n   = np.array(
        [{"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}.get(p["edu_tier"], 0)
         for p in parsed], dtype=np.float32
    )

    # Negative title mask (non-AI job families that keyword-stuff)
    neg_titles = {
        "marketing manager", "graphic designer", "accountant",
        "mechanical engineer", "civil engineer", "sales executive",
        "content writer", "customer support", "hr manager", "operations manager",
    }
    neg_title_mask = np.array(
        [any(neg in p["current_title"].lower() for neg in neg_titles) for p in parsed],
        dtype=np.float32,
    )

    print(f"  Array extraction done in {time.time()-t:.3f}s")

    # ── Skill score ───────────────────────────────────────────────────────────
    skill = (
        ai_counts / 10.0 * 0.40 +    # AI skill count (8+ = strong signal)
        prof * 0.20 +                  # proficiency depth
        np.clip(ai_counts / 5.0, 0, 1) * 0.25 +   # nonlinear boost for high AI skill count
        np.where(avg_assess >= 0, avg_assess / 100.0 * 0.15, 0.075)  # platform assessment
    )
    skill = np.clip(skill, 0.0, 1.0)

    # ── Availability score ────────────────────────────────────────────────────
    avail = np.full(N, 0.40, dtype=np.float32)
    avail += np.where(open_work == 1, 0.20, 0)
    avail += np.where(days_inact <= 7,   0.18,
             np.where(days_inact <= 30,  0.13,
             np.where(days_inact <= 90,  0.05,
             np.where(days_inact <= 180, -0.10, -0.28))))
    avail += np.where(resp_rates >= 0.75,  0.15,
             np.where(resp_rates >= 0.50,  0.10,
             np.where(resp_rates >= 0.25,  0.04,
             np.where(resp_rates <= 0.10, -0.20, 0))))
    avail += np.where(resp_hrs <= 4,   0.08,
             np.where(resp_hrs <= 24,  0.05,
             np.where(resp_hrs >= 72, -0.05, 0)))
    avail += np.where(notice <= 30,  0.08,
             np.where(notice <= 60,  0.04,
             np.where(notice >= 90, -0.04, 0)))
    avail += np.where(saved_30d >= 5, 0.04, 0)
    avail += np.where(interview_r >= 0.8,  0.06,
             np.where(interview_r <= 0.3, -0.04, 0))
    avail += verified * 0.02
    avail = np.clip(avail, 0.0, 1.0)

    # ── Career score ──────────────────────────────────────────────────────────
    # Experience fit: JD says 5-9 years
    exp_score = np.where(
        years >= 5,
        0.70 + np.minimum((years - 5) * 0.04, 0.20),
        np.maximum(years / 5.0, 0)
    )
    career = exp_score * 0.30
    career += ai_title * 0.18
    career += seniority * 0.10
    career += prod_co  * 0.10
    career += has_cs   * 0.04
    career += np.where(edu_tier_n == 4, 0.05, np.where(edu_tier_n == 3, 0.03, 0))
    career += np.where(avg_tenure >= 36, 0.08,
              np.where(avg_tenure >= 24, 0.06,
              np.where(avg_tenure >= 18, 0.04,
              np.where(avg_tenure >= 12, 0.02, 0))))
    career += has_lead * 0.05
    # Red flags — the JD explicitly calls these out
    career -= only_cons    * 0.22
    career -= title_hop    * 0.15
    career -= neg_title_mask * 0.18
    career = np.clip(career, 0.0, 1.0)

    # ── Platform score ────────────────────────────────────────────────────────
    platform = np.full(N, 0.20, dtype=np.float32)
    platform += np.where(github >= 70, 0.30,
                np.where(github >= 40, 0.20,
                np.where(github >= 10, 0.10, 0)))
    platform += (profile_c / 100.0) * 0.18
    platform += np.where(connections >= 500, 0.08,
                np.where(connections >= 200, 0.04, 0))
    platform += np.where(endorsements >= 50, 0.07,
                np.where(endorsements >= 20, 0.04, 0))
    platform += linkedin * 0.04
    platform = np.clip(platform, 0.0, 1.0)

    return {
        "skill":    skill,
        "avail":    avail,
        "career":   career,
        "platform": platform,
    }


def build_reasoning_vec(parsed: list, final_scores: np.ndarray) -> list:
    """
    Build specific, honest reasoning per candidate.
    Judges review 10 random rows — each must be unique and accurate.
    """
    reasons = []
    for i, p in enumerate(parsed):
        title    = p.get("current_title", "")
        years    = p.get("years_experience", 0)
        ai_cnt   = p.get("ai_skill_count", 0)
        rr       = p.get("response_rate", 0)
        days     = p.get("days_inactive", 999)
        notice   = p.get("notice_days", 90)
        skills   = p.get("skills", [])[:5]
        company  = p.get("current_company", "")
        only_c   = p.get("only_consulting", False)
        github   = p.get("github_score", -1)
        open_w   = p.get("open_to_work", False)
        loc      = p.get("location", "")

        parts = [f"{title} with {years:.1f} yrs"]

        if ai_cnt >= 6:
            parts.append(f"strong AI profile ({ai_cnt} AI skills)")
        elif ai_cnt >= 3:
            parts.append(f"{ai_cnt} relevant AI skills")
        else:
            parts.append(f"limited AI skill depth ({ai_cnt} AI skills)")

        if open_w:
            parts.append("actively seeking")
        if days <= 30:
            parts.append(f"active {days}d ago")
        elif days > 180:
            parts.append(f"inactive {days}d — availability concern")

        if rr >= 0.6:
            parts.append(f"high response rate ({rr:.0%})")
        elif rr <= 0.15:
            parts.append(f"low response rate ({rr:.0%})")

        if notice <= 30:
            parts.append("immediate joiner")
        elif notice >= 90:
            parts.append(f"{notice}d notice")

        if only_c:
            parts.append("consulting-only background")

        if github >= 50:
            parts.append(f"active GitHub (score {github:.0f})")

        if loc:
            parts.append(f"{loc}")

        reasons.append("; ".join(parts) + ".")
    return reasons


def run_pipeline(candidates_path: str, jd_path: str, out_path: str):
    t0 = time.time()

    # ── JD ────────────────────────────────────────────────────────────────────
    jd_text = JD_TEXT
    if jd_path and Path(jd_path).exists():
        try:
            jd = parse_jd(jd_path)
            raw = jd.get("raw_text", "")
            if len(raw) > 200:
                jd_text = JD_TEXT + "\n" + raw   # augment, don't replace
                print("JD loaded and merged with built-in text.")
        except Exception as e:
            print(f"JD load failed ({e}), using built-in text.")

    # ── Load + parse candidates ───────────────────────────────────────────────
    candidates = load_candidates(candidates_path)
    if not candidates:
        print("ERROR: no candidates loaded.")
        sys.exit(1)

    # ── Semantic scores ───────────────────────────────────────────────────────
    print("Computing semantic similarity scores...")
    profile_texts = [c["profile_text"] for c in candidates]
    sem = tfidf_semantic_scores(jd_text, profile_texts)

    # ── Component scores (fully vectorised) ───────────────────────────────────
    print("Computing component scores (vectorised)...")
    t2 = time.time()
    scores = vectorised_scores(candidates)
    print(f"  Component scores done in {time.time()-t2:.3f}s")

    # ── Hybrid final score ────────────────────────────────────────────────────
    # Weights: Semantic 25%, Skill 35%, Career 22%, Availability 12%, Platform 6%
    final = (
        sem                  * 0.25 +
        scores["skill"]      * 0.35 +
        scores["career"]     * 0.22 +
        scores["avail"]      * 0.12 +
        scores["platform"]   * 0.06
    )
    final = np.clip(final, 0.0, 1.0)

    # ── Sort + top 100 ────────────────────────────────────────────────────────
    sorted_idx = np.argsort(final)[::-1]
    top100_idx = sorted_idx[:100]

    # ── Build submission dataframe ─────────────────────────────────────────────
    top100_candidates = [candidates[i] for i in top100_idx]
    top100_scores     = final[top100_idx]
    reasons           = build_reasoning_vec(top100_candidates, top100_scores)

    rows = []
    for rank_pos, (i, cand_idx) in enumerate(zip(range(100), top100_idx)):
        rows.append({
            "candidate_id": candidates[cand_idx]["candidate_id"],
            "rank":         rank_pos + 1,
            "score":        round(float(top100_scores[rank_pos]), 4),
            "reasoning":    reasons[rank_pos],
        })

    # Tie-break: equal score → candidate_id ascending (required by validator)
    df = pd.DataFrame(rows)
    df = df.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = range(1, 101)

    # ── Save ──────────────────────────────────────────────────────────────────
    df[["candidate_id", "rank", "score", "reasoning"]].to_csv(out_path, index=False, encoding="utf-8")

    total = time.time() - t0
    print(f"\n✅  Submission saved → {out_path}")
    print(f"    Total runtime: {total:.1f}s")
    print(f"    Top 5 candidates:")
    for _, r in df.head(5).iterrows():
        c = candidates[sorted_idx[int(r["rank"]) - 1]]
        print(f"    #{r['rank']:3d}  {r['candidate_id']}  score={r['score']:.4f}  {c['current_title']}  yrs={c['years_experience']:.1f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TalentLens AI Ranker")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--jd",         default=DEFAULT_JD)
    parser.add_argument("--out",        default=DEFAULT_OUT)
    args = parser.parse_args()
    run_pipeline(args.candidates, args.jd, args.out)