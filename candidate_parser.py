"""
candidate_parser.py  — REDROB SCHEMA EDITION
Parses the real hackathon JSONL (100k candidates) into structured profiles.
Handles the nested schema: profile / career_history / skills / education /
redrob_signals.
"""

import json
import re
from datetime import datetime, date
from pathlib import Path


CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "l&t infotech", "ltimindtree", "dunder mifflin", "acme corp",
}

AI_CORE_SKILLS = {
    "python", "machine learning", "deep learning", "nlp",
    "natural language processing", "embeddings", "transformers",
    "bert", "gpt", "llm", "rag", "langchain", "vector database",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "sentence-transformers", "openai", "huggingface",
    "pytorch", "tensorflow", "keras", "scikit-learn",
    "information retrieval", "ranking", "recommendation", "search",
    "fine-tuning", "lora", "qlora", "peft",
    "mlops", "kubeflow", "mlflow", "wandb", "weights & biases",
    "bm25", "elasticsearch", "opensearch", "sparse retrieval",
    "ndcg", "mrr", "learning to rank", "xgboost",
    "speech recognition", "computer vision", "image classification",
    "generative ai", "diffusion models", "gans",
}


def parse_candidates_jsonl(filepath: str, max_rows: int = None) -> list[dict]:
    """
    Fast streaming JSONL parser — never loads full file into RAM.
    max_rows=None means load all.
    """
    candidates = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                candidates.append(_parse_one(raw))
            except Exception:
                continue
    return candidates


def parse_candidates_json(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [_parse_one(r) for r in data]
    return []


def _parse_one(raw: dict) -> dict:
    """Convert one raw JSONL record into a flat scoring-ready dict."""
    profile    = raw.get("profile", {})
    career     = raw.get("career_history", [])
    skills_raw = raw.get("skills", [])
    edu        = raw.get("education", [])
    certs      = raw.get("certifications", [])
    signals    = raw.get("redrob_signals", {})

    # ── Basic fields ──────────────────────────────────────────────────────────
    candidate_id   = raw.get("candidate_id", "")
    name           = profile.get("anonymized_name", "Unknown")
    headline       = profile.get("headline", "")
    summary        = profile.get("summary", "")
    current_title  = profile.get("current_title", "")
    current_company= profile.get("current_company", "")
    years_exp      = float(profile.get("years_of_experience", 0) or 0)
    country        = profile.get("country", "")
    location       = profile.get("location", "")
    industry       = profile.get("current_industry", "")
    company_size   = profile.get("current_company_size", "")

    # ── Skills ────────────────────────────────────────────────────────────────
    skill_names = [s.get("name", "") for s in skills_raw if isinstance(s, dict)]
    skill_text  = " ".join(skill_names).lower()
    ai_skill_count = sum(1 for s in skill_names if s.lower() in AI_CORE_SKILLS)

    # Skill proficiency score (advanced/expert = high weight)
    prof_map = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
    skill_proficiency_score = 0.0
    if skills_raw:
        total_prof = sum(
            prof_map.get(s.get("proficiency", "beginner"), 0.25)
            for s in skills_raw if isinstance(s, dict)
        )
        skill_proficiency_score = total_prof / len(skills_raw)

    # Skill assessment scores from platform
    assessment_scores = signals.get("skill_assessment_scores", {})
    avg_assessment = (
        sum(assessment_scores.values()) / len(assessment_scores)
        if assessment_scores else -1
    )

    # ── Career signals ────────────────────────────────────────────────────────
    career_descriptions = " ".join(
        c.get("description", "") for c in career if isinstance(c, dict)
    )
    career_titles = [c.get("title", "") for c in career if isinstance(c, dict)]
    career_companies = [c.get("company", "") for c in career if isinstance(c, dict)]

    # Product company vs pure consulting
    only_consulting = _only_consulting(career_companies)
    has_product_company = _has_product_company(career_companies, industry)

    # Average tenure per role
    durations = [c.get("duration_months", 0) for c in career if isinstance(c, dict)]
    avg_tenure_months = sum(durations) / len(durations) if durations else 0

    # Title-hopper detection (avg < 18 months AND > 3 roles)
    is_title_hopper = (avg_tenure_months < 18 and len(career) > 3)

    # Seniority trajectory (are they growing upward?)
    seniority_score = _career_seniority_score(career_titles)

    # Has leadership
    has_leadership = any(
        kw in t.lower()
        for t in career_titles
        for kw in ["lead", "senior", "principal", "staff", "head", "architect", "director"]
    )

    # Recent title relevance
    recent_title_ai_relevant = _is_ai_relevant_title(current_title)

    # ── Education ────────────────────────────────────────────────────────────
    edu_tier = _best_edu_tier(edu)
    has_cs_degree = any(
        "computer" in (e.get("field_of_study", "") or "").lower() or
        "data" in (e.get("field_of_study", "") or "").lower() or
        "machine learning" in (e.get("field_of_study", "") or "").lower() or
        "ai" in (e.get("field_of_study", "") or "").lower()
        for e in edu if isinstance(e, dict)
    )

    # ── Redrob signals ────────────────────────────────────────────────────────
    open_to_work    = signals.get("open_to_work_flag", False)
    last_active_str = signals.get("last_active_date", "")
    days_inactive   = _days_inactive(last_active_str)
    response_rate   = float(signals.get("recruiter_response_rate", 0) or 0)
    resp_time_hrs   = float(signals.get("avg_response_time_hours", 999) or 999)
    github_score    = float(signals.get("github_activity_score", -1) or -1)
    profile_complete= float(signals.get("profile_completeness_score", 0) or 0)
    saved_30d       = int(signals.get("saved_by_recruiters_30d", 0) or 0)
    interview_rate  = float(signals.get("interview_completion_rate", 0) or 0)
    offer_acc_rate  = float(signals.get("offer_acceptance_rate", -1) or -1)
    notice_days     = int(signals.get("notice_period_days", 90) or 90)
    verified_email  = bool(signals.get("verified_email", False))
    linkedin_conn   = bool(signals.get("linkedin_connected", False))
    connection_count= int(signals.get("connection_count", 0) or 0)
    endorsements    = int(signals.get("endorsements_received", 0) or 0)
    search_appear   = int(signals.get("search_appearance_30d", 0) or 0)

    # ── Build profile_text for embedding ────────────────────────────────────
    profile_text = _build_profile_text(
        headline, summary, current_title, skill_names,
        career_descriptions, career_titles, edu, certs
    )

    return {
        # identity
        "candidate_id":      candidate_id,
        "name":              name,
        "current_title":     current_title,
        "current_company":   current_company,
        "location":          location,
        "country":           country,
        "years_experience":  years_exp,

        # skills
        "skills":                  skill_names,
        "skill_text":              skill_text,
        "ai_skill_count":          ai_skill_count,
        "skill_proficiency_score": skill_proficiency_score,
        "avg_assessment_score":    avg_assessment,

        # career
        "career_titles":           career_titles,
        "career_companies":        career_companies,
        "only_consulting":         only_consulting,
        "has_product_company":     has_product_company,
        "avg_tenure_months":       avg_tenure_months,
        "is_title_hopper":         is_title_hopper,
        "seniority_score":         seniority_score,
        "has_leadership":          has_leadership,
        "recent_title_ai_relevant":recent_title_ai_relevant,
        "num_roles":               len(career),

        # education
        "edu_tier":        edu_tier,
        "has_cs_degree":   has_cs_degree,

        # redrob signals
        "open_to_work":       open_to_work,
        "days_inactive":      days_inactive,
        "response_rate":      response_rate,
        "resp_time_hrs":      resp_time_hrs,
        "github_score":       github_score,
        "profile_complete":   profile_complete,
        "saved_30d":          saved_30d,
        "interview_rate":     interview_rate,
        "offer_acc_rate":     offer_acc_rate,
        "notice_days":        notice_days,
        "verified_email":     verified_email,
        "linkedin_connected": linkedin_conn,
        "connection_count":   connection_count,
        "endorsements":       endorsements,
        "search_appearance":  search_appear,

        # embedding
        "profile_text": profile_text,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _only_consulting(companies: list) -> bool:
    if not companies:
        return False
    clean = [c.lower().strip() for c in companies]
    return all(any(f in c for f in CONSULTING_FIRMS) for c in clean)


def _has_product_company(companies: list, industry: str) -> bool:
    if not companies:
        return False
    consulting_lower = CONSULTING_FIRMS
    clean = [c.lower().strip() for c in companies]
    return any(not any(f in c for f in consulting_lower) for c in clean)


def _career_seniority_score(titles: list) -> float:
    """Score 0-1 based on seniority trajectory across roles."""
    senior_kws = ["senior", "lead", "principal", "staff", "head", "director",
                  "architect", "manager", "vp", "chief"]
    junior_kws = ["junior", "intern", "trainee", "associate", "assistant"]
    score = 0.5
    if not titles:
        return score
    # Weight recent titles more
    for i, t in enumerate(titles):
        t_lower = t.lower()
        weight = (i + 1) / len(titles)
        if any(k in t_lower for k in senior_kws):
            score += 0.1 * weight
        if any(k in t_lower for k in junior_kws):
            score -= 0.05 * weight
    return round(max(0.0, min(1.0, score)), 4)


def _is_ai_relevant_title(title: str) -> bool:
    ai_titles = [
        "ml", "machine learning", "ai ", "data scientist", "nlp",
        "deep learning", "research scientist", "ai engineer",
        "data engineer", "backend engineer", "software engineer",
        "platform engineer", "search engineer", "ranking",
    ]
    t = title.lower()
    return any(kw in t for kw in ai_titles)


def _best_edu_tier(edu: list) -> str:
    tier_order = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    best = 0
    best_label = "unknown"
    for e in edu:
        if not isinstance(e, dict):
            continue
        t = e.get("tier", "unknown")
        val = tier_order.get(t, 0)
        if val > best:
            best = val
            best_label = t
    return best_label


def _days_inactive(last_active_str: str) -> int:
    if not last_active_str:
        return 999
    try:
        last = datetime.strptime(str(last_active_str)[:10], "%Y-%m-%d").date()
        return (date.today() - last).days
    except Exception:
        return 999


def _build_profile_text(headline, summary, current_title, skill_names,
                         career_desc, career_titles, edu, certs) -> str:
    edu_text = " ".join(
        f"{e.get('degree','')} {e.get('field_of_study','')} {e.get('institution','')}"
        for e in edu if isinstance(e, dict)
    )
    cert_text = " ".join(
        c.get("name", "") for c in certs if isinstance(c, dict)
    )
    titles_text = " ".join(career_titles)
    return " | ".join(filter(None, [
        headline, summary, current_title, titles_text,
        " ".join(skill_names), career_desc[:1000], edu_text, cert_text
    ]))
def parse_candidates(filepath: str, max_rows: int = None) -> list[dict]:
    ext = str(filepath).lower().split(".")[-1]

    if ext == "jsonl":
        return parse_candidates_jsonl(filepath, max_rows=max_rows)

    if ext == "json":
        data = parse_candidates_json(filepath)
        return data[:max_rows] if max_rows else data

    if ext == "csv":
        import pandas as pd
        df = pd.read_csv(filepath)
        if max_rows:
            df = df.head(max_rows)
        return df.to_dict("records")

    if ext in ["xlsx", "xls"]:
        import pandas as pd
        df = pd.read_excel(filepath)
        if max_rows:
            df = df.head(max_rows)
        return df.to_dict("records")

    return []