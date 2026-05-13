# =============================================================================
# STREAMLIT UI - WhatsApp-style chat + Evaluation Dashboard
# -----------------------------------------------------------------------------
# Two tabs:
#   1. Chat                 - WhatsApp-style conversational UI with the
#                             5-agent CrewAI workflow
#   2. Evaluation Dashboard - reads the 4 CSV files produced by
#                             run_evaluation.py and visualises them
# =============================================================================

import os
import requests
import pandas as pd
import streamlit as st

API_URL = "http://localhost:8000/query"

# Paths to the four CSVs produced by run_evaluation.py
EVAL_BASELINE_CSV   = "evaluation_baseline.csv"
EVAL_AGENTIC_CSV    = "evaluation_agentic.csv"
EVAL_COMPARISON_CSV = "evaluation_comparison.csv"
EVAL_SUMMARY_CSV    = "evaluation_summary.csv"


st.set_page_config(
    page_title="Agentic RAG Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS - WhatsApp-style bubbles + responsive wrapping
# =============================================================================
st.markdown("""
<style>
    /* User bubble - green like WhatsApp, dark text */
    .user-bubble {
        background-color: #DCF8C6 !important;
        color: #000000 !important;
        padding: 10px 14px;
        border-radius: 12px 12px 2px 12px;
        margin: 8px 0 8px auto;
        max-width: 70%;
        width: fit-content;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
        box-shadow: 0 1px 1px rgba(0,0,0,0.1);
        margin-left: auto;
    }

    /* Container for assistant response */
    .agent-response {
        margin: 8px 0;
        max-width: 85%;
    }

    /* Final answer bubble - white background, FORCED dark text */
    .final-bubble {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 2px;
        margin: 4px 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
        box-shadow: 0 1px 1px rgba(0,0,0,0.1);
        border-left: 3px solid #25D366;
    }

    /* Force dark text on ALL elements INSIDE the final bubble */
    .final-bubble * {
        color: #000000 !important;
    }

    /* Agent card - light grey background, FORCED dark text */
    .agent-card {
        background-color: #F0F2F5 !important;
        color: #000000 !important;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
        font-size: 0.9em;
        border-left: 3px solid #888;
    }

    /* Force dark text on ALL elements INSIDE agent cards */
    .agent-card * {
        color: #000000 !important;
    }

    /* Status pills */
    .pass-pill {
        background-color: #D4EDDA !important;
        color: #155724 !important;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .review-pill {
        background-color: #FFF3CD !important;
        color: #856404 !important;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =============================================================================
# SIDEBAR - conversation history
# =============================================================================
with st.sidebar:
    st.markdown("### 💬 Conversation History")

    if not st.session_state.messages:
        st.info("No questions yet. Ask one to get started!")
    else:
        user_questions = [
            (i, msg["content"])
            for i, msg in enumerate(st.session_state.messages)
            if msg["role"] == "user"
        ]
        question_labels = [
            f"Q{idx+1}: {q[:50]}{'...' if len(q) > 50 else ''}"
            for idx, (_, q) in enumerate(user_questions)
        ]
        question_labels = ["-- Select a previous question --"] + question_labels

        selected = st.selectbox(
            "Jump to a previous question:",
            options=range(len(question_labels)),
            format_func=lambda x: question_labels[x],
        )

        if selected > 0:
            idx, full_q = user_questions[selected - 1]
            st.markdown(f"**Full question:**")
            st.markdown(f"> {full_q}")

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# =============================================================================
# MAIN AREA - two tabs
# =============================================================================
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Dashboard"])


# =============================================================================
# TAB 1 - Chat
# =============================================================================
# =============================================================================
# TAB 1 - Chat
# =============================================================================
with tab_chat:
    st.markdown("## Agentic RAG Chat")
    st.caption("Ask financial questions about SEC 10-K filings. "
               "Five specialist agents will analyze your question.")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="agent-response">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="final-bubble">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

                if msg.get("agent_steps"):
                    # Filter out the Retriever -- its raw chunk output is
                    # noisy and duplicates what's already shown in the
                    # "Retrieved context" expander below.
                    visible_steps = [
                        s for s in msg["agent_steps"]
                        if s["agent"] != "Financial Filings Retriever"
                    ]

                    with st.expander(f"🤖 View {len(visible_steps)} agent contributions"):
                        agent_emoji = {
                            "Planning Coordinator": "🎯",
                            "Financial Analyst": "📈",
                            "Portfolio Strategist": "💼",
                            "Risk Assessment Specialist": "⚠️",
                        }
                        for step in visible_steps:
                            emoji = agent_emoji.get(step["agent"], "🤖")
                            st.markdown(f"**{emoji} {step['agent']}**")
                            st.markdown(
                                f'<div class="agent-card">{step["output"]}</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown("")

                if msg.get("context"):
                    with st.expander("📄 Retrieved context"):
                        st.markdown(
                            f'<div class="agent-card">{msg["context"]}</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Type your question here...")
    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })
        with st.spinner("🤔 Agents are thinking..."):
            try:
                r = requests.post(API_URL, json={"query": user_input}, timeout=300)
                data = r.json()
            except Exception as e:
                st.error(f"API request failed: {e}")
                st.stop()
        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("response", ""),
            "agent_steps": data.get("agent_steps", []),
            "context": data.get("retrieved_context", ""),
        })
        st.rerun()


# =============================================================================
# TAB 2 - Evaluation Dashboard
# -----------------------------------------------------------------------------
# Mirrors the structure of run_evaluation.py output:
#   - Summary table (PASS/REVIEW status per metric)
#   - Bar chart comparing Baseline vs Agentic
#   - Side-by-side per-question comparison
#   - Individual baseline and agentic tables
#   - Verdict
# =============================================================================
# =============================================================================
# TAB 2 - Evaluation Dashboard
# -----------------------------------------------------------------------------
# Reads evaluation_results.csv and evaluation_summary.csv produced by
# run_evaluation.py. Shows BLEU, ROUGE-1/2/L, Relevance comparisons.
# =============================================================================
with tab_eval:
    st.markdown("## 📊 Evaluation Dashboard")
    st.markdown(
    "Comparison of **Baseline RAG** vs **Agentic RAG** using BLEU, "
    "ROUGE-L, and relevance scores on ground-truth questions."
    )
    st.markdown("Generate the data by running:")
    st.code("python -m app.evaluation.run_evaluation", language="bash")

    if not (os.path.exists("evaluation_results.csv") and
            os.path.exists("evaluation_summary.csv")):
        st.warning("Run the evaluation script first to generate results.")
    else:
        results_df = pd.read_csv("evaluation_results.csv")
        summary_df = pd.read_csv("evaluation_summary.csv")

        st.success(f"Loaded results for {len(results_df)} questions.")

        # -----------------------------------------------------------------
        # SECTION 1: Aggregate summary as KPI cards
        # -----------------------------------------------------------------
        st.subheader("📋 Aggregate Scores")

        cols = st.columns(len(summary_df))
        for i, row in summary_df.iterrows():
            with cols[i]:
                st.metric(
                    label=row["Metric"],
                    value=f"{row['Agentic (avg)']:.4f}",
                    delta=f"{row['Improvement']:+.4f}",
                )
                winner = row["Winner"]
                color = "#D4EDDA" if winner == "Agentic" else "#F8D7DA" if winner == "Baseline" else "#FFF3CD"
                text_color = "#155724" if winner == "Agentic" else "#721C24" if winner == "Baseline" else "#856404"
                st.markdown(
                    f'<span style="background-color:{color}; color:{text_color}; '
                    f'padding:4px 10px; border-radius:12px; font-weight:bold; '
                    f'font-size:0.85em;">{winner} wins</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # -----------------------------------------------------------------
        # SECTION 2: Summary table
        # -----------------------------------------------------------------
        st.subheader("📑 Summary Table")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # -----------------------------------------------------------------
        # SECTION 3: Bar chart - Baseline vs Agentic per metric
        # -----------------------------------------------------------------
        st.subheader("📊 Visual Comparison")
        chart_df = summary_df.set_index("Metric")[["Baseline (avg)", "Agentic (avg)"]]
        st.bar_chart(chart_df)

        # -----------------------------------------------------------------
        # SECTION 4: Win count per metric
        # -----------------------------------------------------------------
        st.subheader("🏆 Per-Question Wins")
        st.caption("How many questions did each pipeline win on, by metric?")
        
        metrics = ["BLEU", "ROUGE-L", "Relev"]
        
        win_rows = []
        for m in metrics:
                    agentic_wins = int((results_df[f"Agent {m}"] > results_df[f"Base {m}"]).sum())
                    baseline_wins = int((results_df[f"Base {m}"] > results_df[f"Agent {m}"]).sum())
                    ties = len(results_df) - agentic_wins - baseline_wins
        
                    win_rows.append({
                      "Metric": m,
                      "Agentic Wins": agentic_wins,
                      "Baseline Wins": baseline_wins,
                      "Ties": ties,
            })
        
        win_df = pd.DataFrame(win_rows)
        st.dataframe(win_df, use_container_width=True, hide_index=True)

        # -----------------------------------------------------------------
        # SECTION 5: Per-question scores
        # -----------------------------------------------------------------
        st.subheader("🔍 Per-Question Scores")
        score_cols = [
            "ID", "Question",
            "Base BLEU", "Agent BLEU",
            "Base ROUGE-L", "Agent ROUGE-L",
            "Base Relev", "Agent Relev",
        ]
        st.dataframe(results_df[score_cols], use_container_width=True, hide_index=True)

        # -----------------------------------------------------------------
        # SECTION 6: Read the actual answers (collapsible)
        # -----------------------------------------------------------------
        with st.expander("📝 View all generated answers"):
            for _, row in results_df.iterrows():
                st.markdown(f"### {row['ID']}: {row['Question']}")
                st.markdown(f"**Reference:** {row['Reference']}")
                st.markdown(f"**Baseline:** {row['Baseline Answer']}")
                st.markdown(f"**Agentic:** {row['Agentic Answer']}")
                st.markdown("---")

        # -----------------------------------------------------------------
        # SECTION 7: Download buttons
        # -----------------------------------------------------------------
        st.subheader("💾 Download Results")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download per-question results (CSV)",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Download summary (CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )