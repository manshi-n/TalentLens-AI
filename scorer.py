"""
scorer.py — TalentLens AI
Explainable multi-signal scoring for INDIA RUNS Data & AI Challenge.
"""

JD_REQUIRED_SKILLS = {
    "embeddings", "sentence-transformers", "vector database", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "hybrid search", "information retrieval",
    "python", "ranking", "ndcg", "mrr", "evaluation", "a/b testing",
    "retrieval", "re-ranking", "semantic search", "machine learning",
    "ml", "ai", "nlp", "llm", "data science",
}

JD_PREFERRED_SKILLS = {
    "fine-tuning", "lora", "qlora", "peft", "learning to rank",
    "xgboost", "distributed systems", "large-scale inference",
    "hr-tech", "recruiting", "open-source", "bge", "e5",
    "rag", "transformers", "bert", "gpt", "hugging face",
}

JD_NEGATIVE_SIGNALS = {
    "marketing manager", "graphic designer", "accountant",
    "mechanical engineer", "civil engineer", "sales executive",
    "content writer", "customer support", "hr manager",
    "operations manager",
}

JD_POSITIVE_TITLES = {
    "machine learning engineer", "ml engineer", "ai engineer",
    "data scientist", "nlp engineer", "research scientist",
    "search engineer", "ranking engineer", "platform engineer",
    "backend engineer", "software engineer", "senior machine learning",
    "senior ml", "senior ai", "applied scientist", "data engineer",
    "recommendation systems engineer",
}


def _safe_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _as_skill_set(candidate: dict) -> set:
    skills = candidate.get("skills", [])

    if isinstance(skills, list):
        return {str(s).lower().strip() for s in skills if str(s).strip()}

    if isinstance(skills, str):
        return {s.strip().lower() for s in skills.split(",") if s.strip()}

    return set()


def _profile_text(candidate: dict) -> str:
    parts = [
        candidate.get("name", ""),
        candidate.get("current_title", ""),
        candidate.get("profile_text", ""),
        candidate.get("summary", ""),
        candidate.get("headline", ""),
        candidate.get("experience_summary", ""),
        candidate.get("projects", ""),
    ]

    skills = candidate.get("skills", [])
    if isinstance(skills, list):
        parts.append(" ".join(str(s) for s in skills))
    else:
        parts.append(str(skills))

    return " ".join(str(p) for p in parts if p).lower()


def matched_and_missing_skills(candidate: dict, jd: dict = None) -> tuple[list, list]:
    skill_names = _as_skill_set(candidate)
    text = _profile_text(candidate)

    required = set(JD_REQUIRED_SKILLS)

    if jd:
        jd_required = jd.get("required_skills", [])
        if isinstance(jd_required, list) and jd_required:
            required = {str(s).lower().strip() for s in jd_required if str(s).strip()}

    matched = []
    missing = []

    for skill in sorted(required):
        if skill in skill_names or skill in text:
            matched.append(skill)
        else:
            missing.append(skill)

    return matched, missing


def skill_match_score(candidate: dict, jd: dict = None) -> float:
    skill_names = _as_skill_set(candidate)
    profile_text = _profile_text(candidate)

    required = set(JD_REQUIRED_SKILLS)
    preferred = set(JD_PREFERRED_SKILLS)

    if jd:
        jd_required = jd.get("required_skills", [])
        jd_preferred = jd.get("preferred_skills", [])

        if isinstance(jd_required, list) and jd_required:
            required = {str(s).lower().strip() for s in jd_required if str(s).strip()}

        if isinstance(jd_preferred, list) and jd_preferred:
            preferred = {str(s).lower().strip() for s in jd_preferred if str(s).strip()}

    req_hits = sum(1 for s in required if s in skill_names or s in profile_text)
    pref_hits = sum(1 for s in preferred if s in skill_names or s in profile_text)

    req_score = req_hits / max(len(required), 1)
    pref_score = pref_hits / max(len(preferred), 1)

    ai_count = _safe_float(candidate.get("ai_skill_count", 0))
    ai_count_score = min(ai_count / 10.0, 1.0)

    proficiency = _safe_float(candidate.get("skill_proficiency_score", 0.5), 0.5)

    matched_skills, missing_skills = matched_and_missing_skills(candidate, jd)

    base = (
        req_score * 0.65 +
        pref_score * 0.10 +
        ai_count_score * 0.15 +
        proficiency * 0.10
    )

    final_score = max(0.0, min(1.0, base))

    required_count = len(required)
    missing_count = len(missing_skills)

    # Hard caps for realism
    if missing_count > 0:
        final_score = min(final_score, 0.92)

    if required_count > 0:
        missing_ratio = missing_count / required_count

        if missing_ratio >= 0.60:
            final_score = min(final_score, 0.78)
        elif missing_ratio >= 0.45:
            final_score = min(final_score, 0.82)
        elif missing_ratio >= 0.30:
            final_score = min(final_score, 0.88)

    return round(final_score, 4)


def career_score(candidate: dict, jd: dict = None) -> float:
    score = 0.0
    years = _safe_float(candidate.get("years_experience", 0))

    required_exp = 5.0
    if jd:
        required_exp = _safe_float(jd.get("experience_required", 5), 5)

    if years >= required_exp:
        exp_score = 0.70 + min((years - required_exp) * 0.06, 0.30)
    else:
        exp_score = max(0.0, years / max(required_exp, 1))

    score += exp_score * 0.35

    title = str(candidate.get("current_title", "")).lower()

    if any(t in title for t in JD_POSITIVE_TITLES):
        score += 0.20
    elif candidate.get("recent_title_ai_relevant", False):
        score += 0.15
    elif any(neg in title for neg in JD_NEGATIVE_SIGNALS):
        score -= 0.20
    else:
        score += 0.06

    seniority = _safe_float(candidate.get("seniority_score", 0.5), 0.5)
    score += seniority * 0.12

    if candidate.get("has_product_company", False):
        score += 0.10

    if candidate.get("has_cs_degree", False):
        score += 0.04

    edu_tier = str(candidate.get("edu_tier", "unknown")).lower()
    if edu_tier == "tier_1":
        score += 0.04
    elif edu_tier == "tier_2":
        score += 0.02

    avg_tenure = _safe_float(candidate.get("avg_tenure_months", 0))
    if avg_tenure >= 36:
        score += 0.08
    elif avg_tenure >= 24:
        score += 0.06
    elif avg_tenure >= 18:
        score += 0.04
    elif avg_tenure >= 12:
        score += 0.02

    if candidate.get("has_leadership", False):
        score += 0.05

    india_locations = [
        "india", "pune", "noida", "hyderabad", "mumbai",
        "delhi", "bengaluru", "bangalore", "chennai",
    ]

    loc_text = (
        str(candidate.get("location", "")) +
        " " +
        str(candidate.get("country", ""))
    ).lower()

    if any(loc in loc_text for loc in india_locations):
        score += 0.02

    if candidate.get("only_consulting", False):
        score -= 0.20

    if candidate.get("is_title_hopper", False):
        score -= 0.15

    if any(neg in title for neg in JD_NEGATIVE_SIGNALS):
        score -= 0.15

    return round(max(0.0, min(1.0, score)), 4)


def availability_score(candidate: dict, jd: dict = None) -> float:
    score = 0.45

    if candidate.get("open_to_work", False):
        score += 0.18

    days = _safe_float(candidate.get("days_inactive", 999), 999)

    if days <= 7:
        score += 0.18
    elif days <= 30:
        score += 0.13
    elif days <= 90:
        score += 0.05
    elif days <= 180:
        score -= 0.10
    else:
        score -= 0.25

    rr = _safe_float(candidate.get("response_rate", 0), 0)

    if rr >= 0.75:
        score += 0.15
    elif rr >= 0.50:
        score += 0.10
    elif rr >= 0.25:
        score += 0.04
    elif rr <= 0.10:
        score -= 0.18

    resp_hrs = _safe_float(candidate.get("resp_time_hrs", 999), 999)

    if resp_hrs <= 4:
        score += 0.08
    elif resp_hrs <= 24:
        score += 0.05
    elif resp_hrs >= 72:
        score -= 0.05

    notice = _safe_float(candidate.get("notice_days", 90), 90)

    if notice <= 30:
        score += 0.08
    elif notice <= 60:
        score += 0.04
    elif notice >= 90:
        score -= 0.04

    saved = _safe_float(candidate.get("saved_30d", 0), 0)
    if saved >= 5:
        score += 0.04

    interview_rate = _safe_float(candidate.get("interview_rate", 0), 0)
    if interview_rate >= 0.8:
        score += 0.06
    elif interview_rate <= 0.3:
        score -= 0.04

    if candidate.get("verified_email", False):
        score += 0.02

    return round(max(0.0, min(1.0, score)), 4)


def platform_signal_score(candidate: dict, jd: dict = None) -> float:
    score = 0.25

    github = _safe_float(candidate.get("github_score", -1), -1)

    if github >= 70:
        score += 0.28
    elif github >= 40:
        score += 0.18
    elif github >= 10:
        score += 0.08

    completeness = _safe_float(candidate.get("profile_complete", 0), 0)
    score += (completeness / 100.0) * 0.18

    connections = _safe_float(candidate.get("connection_count", 0), 0)
    if connections >= 500:
        score += 0.08
    elif connections >= 200:
        score += 0.04

    endorsements = _safe_float(candidate.get("endorsements", 0), 0)
    if endorsements >= 50:
        score += 0.07
    elif endorsements >= 20:
        score += 0.035

    if candidate.get("linkedin_connected", False):
        score += 0.04

    avg_assessment = _safe_float(candidate.get("avg_assessment_score", -1), -1)
    if avg_assessment >= 70:
        score += 0.12
    elif avg_assessment >= 50:
        score += 0.07
    elif avg_assessment >= 0:
        score += 0.02

    return round(max(0.0, min(1.0, score)), 4)


def hybrid_score(
    semantic: float,
    skill: float,
    career: float,
    availability: float,
    platform: float,
    weights: dict = None,
) -> float:
    w = weights or {
        "semantic": 0.15,
        "skill": 0.40,
        "career": 0.25,
        "availability": 0.15,
        "platform": 0.05,
    }

    total_w = sum(w.values()) or 1.0

    score = (
        semantic * (w.get("semantic", 0) / total_w) +
        skill * (w.get("skill", 0) / total_w) +
        career * (w.get("career", 0) / total_w) +
        availability * (w.get("availability", 0) / total_w) +
        platform * (w.get("platform", 0) / total_w)
    )

    return round(max(0.0, min(1.0, score)), 4)


def assign_tier(score: float) -> str:
    if score >= 0.80:
        return "A+ — Exceptional Fit"
    elif score >= 0.70:
        return "A — Strong Fit"
    elif score >= 0.55:
        return "B — Good Fit"
    elif score >= 0.40:
        return "C — Moderate Fit"
    elif score >= 0.25:
        return "D — Weak Fit"
    else:
        return "E — Reject"


def build_reasoning(candidate: dict, scores: dict) -> str:
    title = candidate.get("current_title", "Candidate")
    years = _safe_float(candidate.get("years_experience", 0))
    ai_count = int(_safe_float(candidate.get("ai_skill_count", 0)))
    response_rate = _safe_float(candidate.get("response_rate", 0))

    matched = scores.get("matched_skills", [])
    missing = scores.get("missing_skills", [])

    matched_text = ", ".join(matched[:3]) if matched else "strong AI/ML profile alignment"
    missing_text = ", ".join(missing[:3]) if missing else "no major missing skill"

    return (
        f"{title} with {years:.1f} yrs experience; "
        f"{ai_count} AI/core skills; "
        f"matches {matched_text}; "
        f"gap: {missing_text}; "
        f"response rate {response_rate:.2f}."
    )


def skill_match_percent(candidate: dict, jd: dict = None) -> float:
    matched, missing = matched_and_missing_skills(candidate, jd)
    total = len(matched) + len(missing)
    return round((len(matched) / max(total, 1)) * 100, 1)


def risk_flags(candidate: dict) -> str:
    risks = []

    if _safe_float(candidate.get("response_rate", 0)) < 0.30:
        risks.append("Low Response Rate")

    if _safe_float(candidate.get("days_inactive", 999)) > 120:
        risks.append("Inactive Profile")

    if _safe_float(candidate.get("notice_days", 90)) > 90:
        risks.append("Long Notice Period")

    return ", ".join(risks) if risks else "No major risk"


def top_strengths(candidate: dict, scores: dict) -> str:
    strengths = []

    if scores.get("semantic", 0) >= 0.40:
        strengths.append("Semantic Fit")

    if scores.get("skill", 0) >= 0.70:
        strengths.append("Technical Match")

    if scores.get("career", 0) >= 0.70:
        strengths.append("Career Fit")

    if scores.get("availability", 0) >= 0.70:
        strengths.append("High Availability")

    if scores.get("platform", 0) >= 0.70:
        strengths.append("Platform Trust")

    return ", ".join(strengths) if strengths else "Partial Fit"
