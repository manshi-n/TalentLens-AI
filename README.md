# TalentLens AI

**Explainable Multi-Signal Candidate Intelligence Engine**

TalentLens AI ranks candidates the way an experienced recruiter would — combining semantic relevance, skill coverage, career alignment, availability signals, and platform trust indicators instead of relying solely on keyword matching.

---

## Overview

TalentLens AI helps recruiters quickly identify the strongest candidates from large talent pools.

The system:

* Reads a Job Description (JD)
* Extracts required skills and role requirements
* Analyzes thousands of candidate profiles
* Computes multi-signal fit scores
* Explains strengths and gaps
* Generates recruiter-ready ranked shortlists

Designed for large-scale hiring workflows, TalentLens AI can process and rank **10,000+ candidates within seconds**.

---

## Key Features

### Semantic Matching

Uses TF-IDF semantic similarity to measure how closely a candidate profile aligns with the job description.

### Skill Coverage Analysis

Measures coverage of required JD skills and identifies missing competencies.

### Career Fit Scoring

Evaluates:

* Years of experience
* Title alignment
* Career progression
* Seniority suitability

### Availability Signals

Considers:

* Open-to-work status
* Notice period
* Recruiter responsiveness
* Recent activity

### Platform Trust Signals

Analyzes platform-level indicators such as:

* Response rate
* Profile completeness
* Engagement signals

### Explainable Recommendations

Each candidate receives:

* Final score
* Hiring recommendation
* Skill match percentage
* Risk flags
* Top strengths
* Missing skills

---

## Screenshots

### Dashboard & Data Upload

![Dashboard](screenshots/1-dashboard-upload.png)

### Ranking Results Overview

![Results Overview](screenshots/2-results-overview.png)

### Explainable Candidate Analysis

![Candidate Analysis](screenshots/3-candidate-details.png)

### Analytics & Score Visualization

![Analytics](screenshots/4-score-analysis.png)

### Ranked Candidate Table

![Ranked Table](screenshots/5-ranked-table.png)

---

## Scoring Framework

TalentLens AI combines five independent signals.

| Signal           | Purpose                                     |
| ---------------- | ------------------------------------------- |
| Semantic Match   | Similarity between JD and candidate profile |
| Skill Match      | Coverage of required skills                 |
| Career Fit       | Experience and title alignment              |
| Availability     | Hiring readiness and responsiveness         |
| Platform Signals | Trust and engagement indicators             |

The final ranking score is generated using a weighted hybrid scoring model.

---

## Hiring Recommendations

Candidates are automatically classified into recruiter-friendly categories:

| Recommendation        | Meaning                    |
| --------------------- | -------------------------- |
| Interview Immediately | Strong overall fit         |
| Keep in Pipeline      | Good fit for future rounds |
| Review Manually       | Requires recruiter review  |
| Reject                | Low relevance to role      |

---

## Technology Stack

### Backend

* Python
* NumPy
* Pandas
* Scikit-learn

### Semantic Matching

* TF-IDF Vectorization
* Cosine Similarity

### Dashboard

* Streamlit
* Plotly

### Data Processing

* JSON
* JSONL
* CSV
* XLSX

---

## Supported Inputs

### Job Description

* DOCX
* TXT
* JSON
* Direct text input

### Candidate Dataset

* JSONL
* JSON
* CSV
* XLSX

> Note: Large candidate datasets are not included in this repository. Users can upload their own JSONL, CSV, JSON, or XLSX files directly through the Streamlit interface.

---

## Output

### Ranked Candidates CSV

Includes:

* Rank
* Candidate ID
* Name
* Current Title
* Experience
* Final Score
* Semantic Score
* Skill Score
* Career Score
* Availability Score
* Platform Score
* Skill Match Percentage
* Hiring Recommendation
* Risk Flags
* Tier Classification

### Submission CSV

Top 100 candidates with:

* Candidate ID
* Rank
* Score
* Reasoning

Suitable for recruiter review and hackathon evaluation.

---

## Project Structure

```text
TalentLens-AI/
│
├── app.py
├── scorer.py
├── rank.py
├── jd_parser.py
├── candidate_parser.py
├── embeddings.py
│
├── screenshots/
│   ├── 1-dashboard-upload.png
│   ├── 2-results-overview.png
│   ├── 3-candidate-details.png
│   ├── 4-score-analysis.png
│   └── 5-ranked-table.png
│
├── outputs/                 # Generated after running the app
│
├── requirements.txt
└── README.md
```

---

## Performance

Typical performance:

| Candidate Count | Runtime    |
| --------------- | ---------- |
| 1,000           | ~1–2 sec   |
| 10,000          | ~5–10 sec  |
| 50,000          | ~20–30 sec |
| 100,000         | ~40–60 sec |

Performance depends on hardware and dataset size.

---

## Use Cases

* Resume Screening
* Candidate Ranking
* Talent Discovery
* Recruiter Intelligence
* Hiring Analytics
* Talent Pipeline Prioritization

---

## Impact

TalentLens AI reduces recruiter screening effort from thousands of profiles to an explainable ranked shortlist within seconds.

Instead of manually reviewing every profile, recruiters can focus on the highest-potential candidates first.

---

## Run Locally

```bash
git clone https://github.com/manshi-n/TalentLens-AI.git

cd TalentLens-AI

pip install -r requirements.txt

streamlit run app.py
```

---

## Future Enhancements

* Resume Parsing
* Candidate Skill Gap Learning Paths
* Recruiter Copilot
* Diversity Analytics
* Interview Question Recommendations
* Advanced Semantic Embeddings
* Real-Time Talent Search

---

## Author

**Manshi**
B.Tech CSE (AI & Data Science)
Graphic Era Hill University

Built for **Data & AI Hackathon 2026**.
