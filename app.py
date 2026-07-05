"""
app.py — TalentLens AI Streamlit UI
Run: streamlit run app.py
"""

import os
import time
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from jd_parser import parse_jd
from candidate_parser import parse_candidates_jsonl, parse_candidates_json
from rank import vectorised_scores, tfidf_semantic_scores, JD_TEXT, build_reasoning_vec
from scorer import (
    assign_tier,
    matched_and_missing_skills,
    skill_match_percent,
    risk_flags,
    top_strengths,
)

st.set_page_config(page_title="TalentLens AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main-title{font-size:42px;font-weight:800;color:#4f46e5}
.subtitle{font-size:17px;color:#6b7280;margin-top:-8px}
.score-badge{font-size:26px;font-weight:800;color:#16a34a}
.speed-note{background:#ecfdf5;padding:10px 14px;border-radius:8px;
            color:#065f46;font-size:13px;border:1px solid #6ee7b7}
</style>
""", unsafe_allow_html=True)


def hiring_decision(score: float) -> str:
    if score >= 0.70:
        return "🟢 Interview Immediately"
    elif score >= 0.55:
        return "🟡 Keep in Pipeline"
    elif score >= 0.40:
        return "🟠 Review Later"
    return "🔴 Reject"


with st.sidebar:
    st.markdown("## 🧠 CandidateIQ")
    st.caption("Explainable multi-signal candidate ranking")
    st.divider()

    max_candidates = st.select_slider(
        "Max candidates to score",
        options=[1000, 2000, 5000, 10000, 25000, 50000, 100000],
        value=10000,
    )

    st.markdown("### ⚖️ Scoring Weights")
    w_sem = st.slider("Semantic AI Match", 0, 100, 25, 5)
    w_sk = st.slider("Skill Match", 0, 100, 35, 5)
    w_car = st.slider("Career Fit", 0, 100, 22, 5)
    w_av = st.slider("Availability", 0, 100, 12, 5)
    w_pl = st.slider("Platform Signals", 0, 100, 6, 5)

    total_w = w_sem + w_sk + w_car + w_av + w_pl
    if total_w == 0:
        total_w = 100

    weights = {
        "semantic": w_sem / total_w,
        "skill": w_sk / total_w,
        "career": w_car / total_w,
        "availability": w_av / total_w,
        "platform": w_pl / total_w,
    }

    if total_w != 100:
        st.warning(f"Weights sum to {total_w}%. Auto-normalised.")

    st.divider()
    st.caption("Runtime: ~10s for 10k | ~40s for 100k")


st.markdown('<div class="main-title">🧠 TalentLens AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Explainable multi-signal candidate intelligence — not keyword matching</div>',
    unsafe_allow_html=True,
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Job Description")
    jd_mode = st.radio(
        "Input",
        ["Upload file (.docx/.txt/.json)", "Paste text"],
        horizontal=True,
    )
    jd_source = None

    if jd_mode == "Upload file (.docx/.txt/.json)":
        jd_file = st.file_uploader(
            "Upload JD",
            type=["docx", "txt", "json"],
            label_visibility="collapsed",
        )
        if jd_file:
            suffix = "." + jd_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(jd_file.read())
                jd_source = tmp.name
            st.success(f"Loaded: {jd_file.name}")
    else:
        jd_text_input = st.text_area(
            "Paste JD text here",
            height=200,
            placeholder="Paste the full job description...",
        )
        if jd_text_input.strip():
            jd_source = jd_text_input

with col2:
    st.subheader("👥 Candidate Dataset")
    cand_file = st.file_uploader(
        "Upload candidates (.jsonl / .json / .csv)",
        type=["jsonl", "json", "csv", "xlsx", "xls"],
    )

    cand_source = None

    if cand_file:
        sz_mb = cand_file.size / (1024 * 1024)
        suffix = "." + cand_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(cand_file.read())
            cand_source = tmp.name

        if sz_mb > 50:
            st.markdown(
                f'<div class="speed-note">⚡ Large file ({sz_mb:.0f} MB) — '
                f'will stream first {max_candidates:,} candidates. '
                f'Estimated time: ~{max(8, max_candidates // 500)}s</div>',
                unsafe_allow_html=True,
            )
        else:
            st.success(f"Loaded: {cand_file.name} ({sz_mb:.1f} MB)")

st.divider()

run = st.button(
    "🚀 Rank Candidates",
    type="primary",
    use_container_width=True,
    disabled=(not cand_source),
)

if run:
    t_start = time.time()
    status = st.status("Running TalentLens AI...", expanded=True)

    with status:
        st.write("📄 Parsing job description...")
        jd_text_for_scoring = JD_TEXT
        jd = {}

        if jd_source:
            try:
                jd = parse_jd(jd_source)
                raw = jd.get("raw_text", "")
                if len(raw) > 100:
                    jd_text_for_scoring = JD_TEXT + "\n" + raw
                st.write(f"   JD parsed: **{jd.get('title', 'Senior AI Engineer')}**")
            except Exception as e:
                st.write(f"   JD parse failed, using built-in: {e}")
        else:
            st.write("   Using built-in JD")

        st.write(f"👥 Loading up to {max_candidates:,} candidates...")
        t2 = time.time()
        ext = cand_source.split(".")[-1].lower()

        if ext == "jsonl":
            candidates = parse_candidates_jsonl(cand_source, max_rows=max_candidates)
        elif ext == "json":
            data = parse_candidates_json(cand_source)
            candidates = data[:max_candidates]
        elif ext == "csv":
            df_raw = pd.read_csv(cand_source, nrows=max_candidates)
            candidates = df_raw.to_dict("records")
        else:
            df_raw = pd.read_excel(cand_source, nrows=max_candidates)
            candidates = df_raw.to_dict("records")

        n = len(candidates)
        st.write(f"   Loaded **{n:,}** candidates in {time.time() - t2:.1f}s")

        if n == 0:
            st.error("No candidates loaded — check your file.")
            st.stop()

        st.write(f"🧠 Computing semantic AI match ({n:,} candidates)...")
        t3 = time.time()

        profile_texts = [c.get("profile_text", "") for c in candidates]
        sem_arr = np.array(
            tfidf_semantic_scores(jd_text_for_scoring, profile_texts),
            dtype=np.float32,
        )

        st.write(f"   Semantic matching done in {time.time() - t3:.1f}s")

        st.write("⚡ Computing 5-signal scores...")
        t4 = time.time()

        comp = vectorised_scores(candidates)
        skill_adj = np.array(comp["skill"], dtype=np.float32).copy()

        for idx, cand in enumerate(candidates):
            matched_tmp, missing_tmp = matched_and_missing_skills(cand, jd)
            missing_count = len(missing_tmp)
            required_count = len(matched_tmp) + missing_count

            if missing_count > 0:
                skill_adj[idx] = min(skill_adj[idx], 0.92)

            if required_count > 0:
                missing_ratio = missing_count / required_count

                if missing_ratio >= 0.60:
                    skill_adj[idx] = min(skill_adj[idx], 0.78)
                elif missing_ratio >= 0.45:
                    skill_adj[idx] = min(skill_adj[idx], 0.82)
                elif missing_ratio >= 0.30:
                    skill_adj[idx] = min(skill_adj[idx], 0.88)

        st.write(f"   Component scores done in {time.time() - t4:.3f}s")

        final = (
            sem_arr * weights["semantic"] +
            skill_adj * weights["skill"] +
            comp["career"] * weights["career"] +
            comp["avail"] * weights["availability"] +
            comp["platform"] * weights["platform"]
        )

        final = np.clip(final, 0.0, 1.0)

        sorted_idx = np.argsort(final)[::-1]
        top_n_show = n

        rows = []

        for rank_pos, cand_idx in enumerate(sorted_idx[:top_n_show]):
            c = candidates[cand_idx]
            sc = float(final[cand_idx])
            matched, missing = matched_and_missing_skills(c, jd)

            score_parts = {
                "semantic": float(sem_arr[cand_idx]),
                "skill": float(skill_adj[cand_idx]),
                "career": float(comp["career"][cand_idx]),
                "availability": float(comp["avail"][cand_idx]),
                "platform": float(comp["platform"][cand_idx]),
            }

            rows.append({
                "rank": rank_pos + 1,
                "candidate_id": c.get("candidate_id", ""),
                "name": c.get("name", f"Candidate {cand_idx}"),
                "current_title": c.get("current_title", ""),
                "years_experience": c.get("years_experience", 0),
                "final_score": round(sc * 100, 2),
                "semantic_score": round(float(sem_arr[cand_idx]) * 100, 1),
                "skill_score": round(float(skill_adj[cand_idx]) * 100, 1),
                "career_score": round(float(comp["career"][cand_idx]) * 100, 1),
                "availability_score": round(float(comp["avail"][cand_idx]) * 100, 1),
                "platform_score": round(float(comp["platform"][cand_idx]) * 100, 1),
                "tier": assign_tier(sc),
                "hiring_decision": hiring_decision(sc),
                "skill_match_percent": skill_match_percent(c, jd),
                "risk_flags": risk_flags(c),
                "top_strengths": top_strengths(c, score_parts),
                "matched_skills": ", ".join(matched[:6]),
                "missing_skills": ", ".join(missing[:6]),
                "open_to_work": c.get("open_to_work", False),
                "response_rate": c.get("response_rate", 0),
                "days_inactive": c.get("days_inactive", 999),
                "notice_days": c.get("notice_days", 90),
                "location": c.get("location", ""),
            })

        df = pd.DataFrame(rows)

        top100_cands = [candidates[i] for i in sorted_idx[:100]]
        top100_scores = final[sorted_idx[:100]]
        reasons = build_reasoning_vec(top100_cands, top100_scores)

        submission_rows = []

        for rank_i, cand in enumerate(top100_cands, start=1):
            score = float(top100_scores[rank_i - 1])

            submission_rows.append({
                "candidate_id": cand.get("candidate_id", ""),
                "rank": rank_i,
                "score": round(score, 4),
                "skill_match_percent": skill_match_percent(cand, jd),
                "risk_flags": risk_flags(cand),
                "hiring_recommendation": hiring_decision(score),
                "reasoning": reasons[rank_i - 1],
            })

        submission_df = pd.DataFrame(submission_rows)

        submission_df = submission_df.sort_values(
            ["score", "candidate_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        submission_df["rank"] = range(1, len(submission_df) + 1)

        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/ranked_candidates.csv", index=False)
        submission_df.to_csv("outputs/submission.csv", index=False)

        total_t = time.time() - t_start

        status.update(
            label=f"✅ Done — {n:,} candidates ranked in {total_t:.1f}s",
            state="complete",
            expanded=False,
        )

        st.session_state["df"] = df
        st.session_state["submission_df"] = submission_df
        st.session_state["jd"] = jd
        st.session_state["n_total"] = n
        st.session_state["total_time"] = total_t

if "df" in st.session_state:
    df = st.session_state["df"]
    sub = st.session_state["submission_df"]
    n = st.session_state.get("n_total", len(df))
    tt = st.session_state.get("total_time", 0)

    st.subheader("📊 Results Overview")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric("Pool Screened", f"{n:,}")
    k2.metric("Runtime", f"{tt:.1f}s")
    k3.metric("Top Score", f"{df['final_score'].max():.1f}/100")
    k4.metric("Avg Score", f"{df['final_score'].mean():.1f}/100")
    k5.metric("Interview Ready", len(df[df["tier"].isin(["A+ — Exceptional Fit", "A — Strong Fit"])]))
    k6.metric("Open to Work", len(df[df["open_to_work"] == True]))

    st.divider()

    st.subheader("🏆 Top Shortlist")

    n_show = st.slider("Show top N candidates", 3, min(30, len(df)), 5)

    for _, row in df.head(n_show).iterrows():
        with st.expander(
            f"**#{row['rank']}** — {row['name']} · {row['current_title']} · "
            f"**{row['final_score']:.1f}/100** · {row['tier']}",
            expanded=(row["rank"] <= 3),
        ):
            left, right = st.columns([3, 1])

            with left:
                st.markdown(
                    f"**Experience:** {row['years_experience']} yrs | "
                    f"**Location:** {row['location']} | "
                    f"**Notice:** {row['notice_days']}d"
                )

                st.markdown(
                    f"**Open to work:** {'✅ Yes' if row['open_to_work'] else '❌ No'} | "
                    f"**Response rate:** {row['response_rate']:.0%} | "
                    f"**Inactive:** {row['days_inactive']}d"
                )

                st.success(f"🎯 Recommendation: {row['hiring_decision']}")
                st.info(f"💪 Strengths: {row['top_strengths']}")
                st.warning(f"⚠️ Risk Flags: {row['risk_flags']}")
                st.markdown(f"**Skill Match:** {row['skill_match_percent']}%")

            with right:
                st.markdown(
                    f'<div class="score-badge">{row["final_score"]:.1f}/100</div>',
                    unsafe_allow_html=True,
                )

            s1, s2, s3, s4, s5 = st.columns(5)

            s1.metric("Semantic", row["semantic_score"])
            s2.metric("Skill", row["skill_score"])
            s3.metric("Career", row["career_score"])
            s4.metric("Availability", row["availability_score"])
            s5.metric("Platform", row["platform_score"])

            g1, g2 = st.columns(2)

            with g1:
                st.markdown("**✅ Matched Skills**")
                st.success(row["matched_skills"] or "Strong AI/ML profile alignment")

            with g2:
                st.markdown("**⚠️ Missing Skills**")
                st.warning(row["missing_skills"] or "No major gaps")

    st.divider()

    st.subheader("📈 Score Analysis")

    color_map = {
        "A+ — Exceptional Fit": "#10b981",
        "A — Strong Fit": "#3b82f6",
        "B — Good Fit": "#f59e0b",
        "C — Moderate Fit": "#f97316",
        "D — Weak Fit": "#ef4444",
        "E — Reject": "#7f1d1d",
    }

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            df.head(20),
            x="name",
            y="final_score",
            color="tier",
            color_discrete_map=color_map,
            title="Top 20 Candidate Scores",
            labels={"final_score": "Score (0–100)", "name": ""},
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        radar = go.Figure()

        for _, row in df.head(5).iterrows():
            radar.add_trace(go.Scatterpolar(
                r=[
                    row["semantic_score"],
                    row["skill_score"],
                    row["career_score"],
                    row["availability_score"],
                    row["platform_score"],
                ],
                theta=["Semantic", "Skill", "Career", "Availability", "Platform"],
                fill="toself",
                name=row["name"],
            ))

        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Score Breakdown — Top 5",
        )

        st.plotly_chart(radar, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        tier_df = df["tier"].value_counts().reset_index()
        tier_df.columns = ["tier", "count"]

        fig3 = px.pie(
            tier_df,
            names="tier",
            values="count",
            color="tier",
            color_discrete_map=color_map,
            title="Tier Distribution",
        )

        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = px.scatter(
            df.head(100),
            x="semantic_score",
            y="skill_score",
            size="final_score",
            color="tier",
            hover_name="name",
            color_discrete_map=color_map,
            title="Semantic vs Skill (top 100)",
        )

        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    st.subheader("📋 Full Ranked Table")

    all_tiers = df["tier"].unique().tolist()

    tier_filter = st.multiselect(
        "Filter by tier",
        all_tiers,
        default=all_tiers,
    )

    if tier_filter:
        df_show = df[df["tier"].isin(tier_filter)]
    else:
        df_show = df

    show_cols = [
        "rank",
        "candidate_id",
        "name",
        "current_title",
        "years_experience",
        "final_score",
        "skill_match_percent",
        "skill_score",
        "career_score",
        "availability_score",
        "hiring_decision",
        "risk_flags",
        "tier",
    ]

    st.dataframe(
        df_show[show_cols].style.background_gradient(
            subset=["final_score"],
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
        ),
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    st.divider()

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "⬇️ Download Full Ranked CSV",
            df.to_csv(index=False).encode("utf-8"),
            "ranked_candidates.csv",
            "text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "⬇️ Download Submission CSV (top 100, validator-ready)",
            sub.to_csv(index=False).encode("utf-8"),
            "submission.csv",
            "text/csv",
            use_container_width=True,
        )