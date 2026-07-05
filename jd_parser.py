"""
jd_parser.py — improved JD parser for TalentLens AI.
Handles JSON, TXT, DOCX and extracts title, seniority, experience, required skills.
"""

import json
import re
from pathlib import Path


def parse_jd(source) -> dict:
    if isinstance(source, dict):
        return _normalise(source)

    path = Path(str(source))

    if path.exists():
        suffix = path.suffix.lower()

        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return _normalise(json.load(f))

        if suffix == ".docx":
            return _normalise({"description": _read_docx(path)})

        return _normalise({
            "description": path.read_text(encoding="utf-8", errors="ignore")
        })

    return _normalise({"description": str(source)})


def _read_docx(path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    except Exception:
        return ""


def _normalise(raw: dict) -> dict:
    description = str(raw.get("description", ""))

    responsibilities = raw.get("responsibilities", [])
    if isinstance(responsibilities, str):
        responsibilities = [responsibilities]

    preferred_skills = raw.get("preferred_skills", [])
    if isinstance(preferred_skills, str):
        preferred_skills = [preferred_skills]

    culture_signals = raw.get("culture_signals", [])
    if isinstance(culture_signals, str):
        culture_signals = [culture_signals]

    title = raw.get("title", "") or _infer_title(description)

    all_text = " ".join([
        str(title),
        description,
        " ".join(responsibilities),
        " ".join(preferred_skills),
        " ".join(culture_signals),
    ]).strip()

    required_skills = raw.get("required_skills", [])
    if isinstance(required_skills, str):
        required_skills = [required_skills]

    extracted_skills = _extract_skills(all_text)

    if not required_skills:
        required_skills = extracted_skills

    return {
        "title": title,
        "seniority": raw.get("seniority", "") or _infer_seniority(all_text),
        "experience_required": raw.get("experience_required", "") or _infer_experience(all_text),
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "responsibilities": responsibilities,
        "culture_signals": culture_signals,
        "raw_text": all_text,
    }


def _infer_title(text: str) -> str:
    text = text.lower()

    patterns = [
        r"senior\s+ai\s+engineer",
        r"ai\s+engineer",
        r"machine\s+learning\s+engineer",
        r"ml\s+engineer",
        r"data\s+scientist",
        r"nlp\s+engineer",
        r"ranking\s+engineer",
        r"search\s+engineer",
        r"backend\s+engineer",
        r"software\s+engineer",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).title()

    return "AI Candidate Ranking Role"


def _infer_seniority(text: str) -> str:
    text = text.lower()

    if any(x in text for x in ["lead", "principal", "staff"]):
        return "Lead"

    if any(x in text for x in ["senior", "5+", "6+", "7+", "5-9", "5 to 9"]):
        return "Senior"

    if any(x in text for x in ["junior", "fresher", "entry", "0-2"]):
        return "Junior"

    return "Mid"


def _infer_experience(text: str) -> int:
    text = text.lower()

    patterns = [
        r"(\d+)\s*[–\-]\s*(\d+)\s*years?",
        r"(\d+)\s*to\s*(\d+)\s*years?",
        r"(\d+)\+?\s*years?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    return 5


def _extract_skills(text: str) -> list:
    text = text.lower()

    skill_aliases = {
        "python": ["python"],
        "machine learning": ["machine learning", "ml"],
        "artificial intelligence": ["artificial intelligence", "ai"],
        "data science": ["data science", "data scientist"],
        "deep learning": ["deep learning"],
        "nlp": ["nlp", "natural language processing"],
        "llm": ["llm", "large language model", "large language models"],
        "generative ai": ["generative ai", "genai"],
        "rag": ["rag", "retrieval augmented generation"],
        "embeddings": ["embedding", "embeddings", "vector embedding", "vector embeddings"],
        "semantic search": ["semantic search", "semantic similarity"],
        "vector database": ["vector database", "vector db", "vector store"],
        "faiss": ["faiss"],
        "pinecone": ["pinecone"],
        "weaviate": ["weaviate"],
        "qdrant": ["qdrant"],
        "milvus": ["milvus"],
        "elasticsearch": ["elasticsearch", "elastic search"],
        "opensearch": ["opensearch", "open search"],
        "hybrid search": ["hybrid search"],
        "information retrieval": ["information retrieval", "retrieval"],
        "ranking": ["ranking", "candidate ranking", "rank candidates"],
        "candidate ranking": ["candidate ranking", "rank candidates", "ranks candidates"],
        "re-ranking": ["re-ranking", "reranking", "re rank"],
        "recommendation system": ["recommendation system", "recommender system"],
        "learning to rank": ["learning to rank"],
        "evaluation": ["evaluation", "evaluate", "metrics"],
        "ndcg": ["ndcg"],
        "mrr": ["mrr"],
        "precision": ["precision"],
        "recall": ["recall"],
        "a/b testing": ["a/b testing", "ab testing"],
        "sentence transformers": ["sentence-transformers", "sentence transformers"],
        "transformers": ["transformers", "bert", "gpt"],
        "pytorch": ["pytorch"],
        "tensorflow": ["tensorflow"],
        "scikit-learn": ["scikit-learn", "sklearn"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "sql": ["sql"],
        "fastapi": ["fastapi"],
        "flask": ["flask"],
        "api": ["api", "rest api"],
        "docker": ["docker"],
        "aws": ["aws"],
        "github": ["github", "git"],
        "resume screening": ["resume screening"],
        "candidate discovery": ["candidate discovery"],
        "hiring workflow": ["hiring workflow", "hiring workflows"],
        "behavioral signals": ["behavioral signals", "platform activity", "availability signals"],
        "career history": ["career history"],
        "skills": ["skills", "skill depth"],
        "platform activity": ["platform activity"],
        "shortlist": ["shortlist", "shortlisted"],
    }

    found = []

    for skill, aliases in skill_aliases.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", text):
                found.append(skill)
                break

    # Fallback for INDIA RUNS Data & AI Challenge
    # This activates whenever the JD is about candidate ranking / hiring,
    # even if the uploaded DOCX does not explicitly list many technical skills.
    if any(x in text for x in ["candidate", "rank", "recruiter", "hiring", "profile", "shortlist"]):
        fallback = [
            "python",
            "machine learning",
            "artificial intelligence",
            "data science",
            "nlp",
            "llm",
            "embeddings",
            "semantic search",
            "vector database",
            "information retrieval",
            "ranking",
            "candidate ranking",
            "resume screening",
            "candidate discovery",
            "hiring workflow",
            "evaluation",
            "behavioral signals",
        ]

        found = list(dict.fromkeys(found + fallback))

    return list(dict.fromkeys(found))