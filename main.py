"""
main.py
Orchestrates the full TalentLens AI pipeline:
  1. Parse JD
  2. Parse candidates
  3. Semantic embeddings + cosine scores
  4. LLM judgment scores
  5. Skill + behavioural signal scores
  6. Hybrid scoring + ranking
  7. Export ranked CSV
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# allow running from project root or src/
sys.path.insert(0, str(Path(__file__).parent))

from jd_parser import parse_jd
from candidate_parser import parse_candidates
from embeddings import semantic_scores
from scorer import skill_match_score, behavioural_score, hybrid_score, assign_tier
from llm_judge import llm_score_candidate


def run_pipeline(
    jd_source,
    candidates_source,
    output_path: str = "outputs/ranked_candidates.csv",
    use_llm: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline. Returns the ranked DataFrame and saves it to output_path.
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)

    # ── 1. Parse inputs ───────────────────────────────────────────────────────
    log("Parsing job description...", verbose)
    jd = parse_jd(jd_source)

    log("Parsing candidates...", verbose)
    candidates = parse_candidates(candidates_source)
    log(f"  → {len(candidates)} candidates loaded", verbose)

    # ── 2. Semantic scores ────────────────────────────────────────────────────
    log("Computing semantic similarity scores...", verbose)
    profile_texts = [c["profile_text"] for c in candidates]
    sem_scores = semantic_scores(jd["raw_text"], profile_texts)

    # ── 3. LLM + signal scores ────────────────────────────────────────────────
    log("Scoring candidates...", verbose)
    rows = []
    for i, candidate in enumerate(candidates):
        cid = candidate["candidate_id"]
        name = candidate["name"]

        # LLM score
        if use_llm:
            log(f"  [{i+1}/{len(candidates)}] LLM judging {name}...", verbose)
            llm_result = llm_score_candidate(candidate, jd)
        else:
            from llm_judge import _heuristic_fallback
            llm_result = _heuristic_fallback(candidate, jd)

        llm_s = llm_result.get("fit_score", 0.5)
        explanation = llm_result.get("explanation", "")
        strengths = "; ".join(llm_result.get("key_strengths", []))
        gaps = "; ".join(llm_result.get("gaps", []))

        # Skill + behavioural scores
        skill_s = skill_match_score(candidate, jd)
        behav_s = behavioural_score(candidate, jd)

        # Final hybrid
        final = hybrid_score(sem_scores[i], llm_s, skill_s, behav_s)
        tier = assign_tier(final)

        rows.append({
            "candidate_id": cid,
            "name": name,
            "current_title": candidate.get("current_title", ""),
            "years_experience": candidate.get("years_experience", 0),
            "final_score": round(final * 100, 1),
            "semantic_score": round(sem_scores[i] * 100, 1),
            "llm_score": round(llm_s * 100, 1),
            "skill_score": round(skill_s * 100, 1),
            "signal_score": round(behav_s * 100, 1),
            "tier": tier,
            "key_strengths": strengths,
            "gaps": gaps,
            "explanation": explanation,
        })

    # ── 4. Rank and export ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    df.to_csv(output_path, index=False)
    log(f"\nRanked output saved to: {output_path}", verbose)
    log(f"Top 3 candidates:", verbose)
    for _, row in df.head(3).iterrows():
        log(f"  #{row['rank']}  {row['name']:20s}  {row['final_score']:.1f}/100  {row['tier']}", verbose)

    return df


def log(msg, verbose):
    if verbose:
        print(msg)


if __name__ == "__main__":
    df = run_pipeline(
        jd_source="data/job_description.json",
        candidates_source="data/candidates.csv",
        output_path="outputs/ranked_candidates.csv",
        use_llm=bool(os.getenv("GROQ_API_KEY")),
        verbose=True,
    )
    print(df[["rank", "name", "final_score", "tier", "explanation"]].to_string(index=False))
