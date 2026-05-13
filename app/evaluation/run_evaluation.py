# =============================================================================
# RAG EVALUATION - Baseline vs Agentic
# -----------------------------------------------------------------------------
# Implements the evaluation requirements from the capstone spec:
#   - BLEU score
#   - ROUGE score (ROUGE-L)
#   - Relevance score
#   - Comparison: baseline RAG vs agentic workflow-enhanced RAG
#
# Run from project root:
#     python -m app.evaluation.run_evaluation
#
# Produces:
#     evaluation_results.csv   (per-question scores, both pipelines)
#     evaluation_summary.csv   (aggregated averages)
# =============================================================================

import os
import sys
import contextlib
import numpy as np
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from app.rag.rag_pipeline import create_rag_pipeline, generate_response
from app.agents.agents import run_agentic_workflow


# =============================================================================
# HELPER: silence CrewAI's verbose output during evaluation
# =============================================================================
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# =============================================================================
# 1. GROUND TRUTH QUESTIONS
# -----------------------------------------------------------------------------
# 10 hand-crafted Q&A pairs based on companies in the
# virattt/financial-qa-10K dataset.
# Format: (id, question, reference_answer)
# =============================================================================
GROUND_TRUTH = [
    ("Q1",
     "What are Apple's main revenue segments?",
     "Apple's main revenue segments are iPhone, Mac, iPad, Wearables, and Services."),

    ("Q2",
     "What products does Apple sell?",
     "Apple sells iPhone, Mac computers, iPad, Apple Watch, AirPods, and various services."),

    ("Q3",
     "What supply chain risks does Apple face?",
     "Apple faces risks from supply chain disruptions, concentrated manufacturing in Asia, and reliance on key component suppliers."),

    ("Q4",
     "What is Microsoft's cloud business called?",
     "Microsoft's cloud business is called Azure, part of the Intelligent Cloud segment."),

    ("Q5",
     "What does Amazon's AWS segment do?",
     "AWS provides cloud computing services including storage, compute, databases, and machine learning to enterprises and developers."),

    ("Q6",
     "What are Tesla's main business risks?",
     "Tesla faces risks related to production scaling, supply chain constraints, battery materials, regulatory changes, and competition in electric vehicles."),

    ("Q7",
     "How does Meta generate revenue?",
     "Meta generates the majority of its revenue from advertising on Facebook, Instagram, and WhatsApp."),

    ("Q8",
     "What competition does Microsoft face?",
     "Microsoft faces competition in cloud services from Amazon AWS and Google Cloud, in productivity from Google Workspace, and in gaming from Sony and Nintendo."),

    ("Q9",
     "What is Alphabet's main product?",
     "Alphabet's main product is Google Search, supported by advertising revenue, along with YouTube, Android, and Google Cloud."),

    ("Q10",
     "What regulatory risks does Meta face?",
     "Meta faces regulatory risks related to data privacy, antitrust scrutiny, content moderation requirements, and changes in advertising policies."),
]


# =============================================================================
# 2. SCORING FUNCTIONS - BLEU, ROUGE, Relevance
# -----------------------------------------------------------------------------
# All three scoring functions defined upfront so the main loop is clean.
# =============================================================================

# --- BLEU ---------------------------------------------------------------------
# BLEU measures n-gram overlap between generated and reference answers.
# We use smoothing so a missing 4-gram doesn't crash the score to zero.
def compute_bleu(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    reference_tokens = [reference.lower().split()]
    prediction_tokens = prediction.lower().split()
    smoother = SmoothingFunction().method1
    return round(
        sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoother),
        4,
    )


# --- ROUGE-L ------------------------------------------------------------------
# ROUGE-L measures longest common subsequence overlap.
# Better for semantic similarity and paraphrased responses.
_rouge_scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True,
)

def compute_rouge(prediction: str, reference: str) -> dict:
    if not prediction or not reference:
        return {"rougeL": 0.0}

    scores = _rouge_scorer.score(reference, prediction)

    return {
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }

# --- Relevance ----------------------------------------------------------------
# Cosine similarity between question embedding and answer embedding.
# High score = answer addresses the question semantically.
def compute_relevance(question: str, answer: str, embeddings) -> float:
    if not question or not answer:
        return 0.0
    try:
        q_vec = np.array(embeddings.embed_query(question))
        a_vec = np.array(embeddings.embed_query(answer))
        cosine = float(
            np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec) + 1e-9)
        )
        return round(cosine, 4)
    except Exception:
        return 0.0


# =============================================================================
# 3. INITIALISE THE RAG PIPELINE
# -----------------------------------------------------------------------------
# Loads the vectorstore + LLM. Needed before we can call generate_response()
# or run_agentic_workflow().
# =============================================================================
print("Initialising RAG pipeline...")
create_rag_pipeline()

# Grab the embedding model from the vectorstore for relevance scoring
from app.rag.rag_pipeline import _vectorstore
hf_embeddings = _vectorstore.embeddings


# =============================================================================
# 4. RUN BOTH PIPELINES
# -----------------------------------------------------------------------------
# Same questions through both pipelines so the comparison is apples-to-apples.
# We collect everything into a single `records` list with both versions.
# =============================================================================
records = []

print("\n" + "=" * 60)
print("RUNNING EVALUATION ON 10 QUESTIONS")
print("=" * 60)

for qid, question, reference in GROUND_TRUTH:
    print(f"\n{qid}: {question}")

    # --- Baseline RAG ---
    try:
        with suppress_output():
            base_result = generate_response(question)
        base_answer = base_result.get("answer", "").strip()
    except Exception as e:
        base_answer = f"ERROR: {e}"

    # --- Agentic RAG ---
    try:
        with suppress_output():
            agent_result = run_agentic_workflow(question)
        agent_answer = agent_result.get("response", "").strip()
    except Exception as e:
        agent_answer = f"ERROR: {e}"

    # --- Score both ---
    base_bleu = compute_bleu(base_answer, reference)
    base_rouge = compute_rouge(base_answer, reference)
    base_relevance = compute_relevance(question, base_answer, hf_embeddings)

    agent_bleu = compute_bleu(agent_answer, reference)
    agent_rouge = compute_rouge(agent_answer, reference)
    agent_relevance = compute_relevance(question, agent_answer, hf_embeddings)

    records.append({
    "ID": qid,
    "Question": question,
    "Reference": reference,
    "Baseline Answer": base_answer,
    "Agentic Answer": agent_answer,

    # Baseline scores
    "Base BLEU":    base_bleu,
    "Base ROUGE-L": base_rouge["rougeL"],
    "Base Relev":   base_relevance,

    # Agentic scores
    "Agent BLEU":    agent_bleu,
    "Agent ROUGE-L": agent_rouge["rougeL"],
    "Agent Relev":   agent_relevance,
})

    print(f"  Baseline -> BLEU: {base_bleu:.3f}  ROUGE-L: {base_rouge['rougeL']:.3f}  "
          f"Relevance: {base_relevance:.3f}")
    print(f"  Agentic  -> BLEU: {agent_bleu:.3f}  ROUGE-L: {agent_rouge['rougeL']:.3f}  "
          f"Relevance: {agent_relevance:.3f}")


# =============================================================================
# 5. BUILD THE RESULTS DATAFRAME
# =============================================================================
results_df = pd.DataFrame(records)

print("\n" + "=" * 80)
print("PER-QUESTION RESULTS")
print("=" * 80)
display_cols = [
    "ID",
    "Base BLEU", "Agent BLEU",
    "Base ROUGE-L", "Agent ROUGE-L",
    "Base Relev", "Agent Relev",
]
print(results_df[display_cols].to_string(index=False))


# =============================================================================
# 6. AGGREGATE SUMMARY
# -----------------------------------------------------------------------------
# Average each metric across all questions, then compute the improvement
# of agentic over baseline.
# =============================================================================
metrics = ["BLEU","ROUGE-L", "Relev"]

summary_rows = []
for metric in metrics:
    base_avg = results_df[f"Base {metric}"].mean()
    agent_avg = results_df[f"Agent {metric}"].mean()
    improvement = agent_avg - base_avg
    winner = "Agentic" if improvement > 0 else "Baseline" if improvement < 0 else "Tie"

    summary_rows.append({
        "Metric": metric,
        "Baseline (avg)": round(base_avg, 4),
        "Agentic (avg)": round(agent_avg, 4),
        "Improvement": round(improvement, 4),
        "Winner": winner,
    })

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 80)
print("AGGREGATE SUMMARY - BASELINE vs AGENTIC")
print("=" * 80)
print(summary_df.to_string(index=False))


# =============================================================================
# 7. WIN COUNT (per-question)
# -----------------------------------------------------------------------------
# How often did agentic beat baseline on each metric?
# =============================================================================
print("\n" + "=" * 80)
print("WIN COUNT - Agentic vs Baseline (per question)")
print("=" * 80)
for metric in metrics:
    agentic_wins = (results_df[f"Agent {metric}"] > results_df[f"Base {metric}"]).sum()
    baseline_wins = (results_df[f"Base {metric}"] > results_df[f"Agent {metric}"]).sum()
    ties = len(results_df) - agentic_wins - baseline_wins
    print(f"  {metric:<10} | Agentic: {agentic_wins}  Baseline: {baseline_wins}  Ties: {ties}")


# =============================================================================
# 8. SAVE CSVs
# =============================================================================
results_df.to_csv("evaluation_results.csv", index=False)
summary_df.to_csv("evaluation_summary.csv", index=False)

print("\n" + "=" * 80)
print("Saved:")
print("  - evaluation_results.csv  (per-question scores + answers)")
print("  - evaluation_summary.csv  (aggregated comparison)")
print("=" * 80)