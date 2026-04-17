"""
inference.py — ESG Authenticity Scorer Inference & Pillar Scoring
=================================================================
Capstone Project 2025-26  |  Apoorva Singh  |  PRN 202201070030

Architecture Context
--------------------
At inference time we load the fine-tuned ``climatebert/distilroberta-base-
climate-f`` checkpoint saved by train.py.  For each sentence the model
outputs a probability distribution over {SPECIFIC, VAGUE}.

P(VAGUE) is the raw vagueness signal.  We invert it to get a
"specificity" sub-score:

    specificity_score = 1 − P(VAGUE)

This is combined with four heuristic sub-scores (quantitative density,
citation evidence, claim verification proxy, topic coverage) using the
weights from config.SCORE_COMPONENT_WEIGHTS to produce a per-pillar score.

Final Authenticity Score (Slide 12)
------------------------------------
    Final Score = 0.40 × E-Score + 0.35 × S-Score + 0.25 × G-Score

Score Interpretation
--------------------
    80–100  Highly Specific — strong quantitative evidence, verifiable claims
    60–79   Mostly Specific — some vague language, generally well-supported
    40–59   Mixed           — notable greenwashing signals present
    20–39   Mostly Vague    — significant hedge language, few concrete facts
     0–19   Highly Vague    — pervasive greenwashing language detected

Usage
-----
    # Interactive mode (prompts for text input):
    python inference.py

    # Score a pre-built text file:
    python inference.py --file path/to/report_chunk.txt

    # Score a single sentence directly:
    python inference.py --text "We aim to reduce emissions over time."

    # JSON output (useful for downstream pipeline):
    python inference.py --text "..." --json
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure logger for inference module."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FMT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8"),
        ],
    )
    return logging.getLogger("inference")


logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class SentenceResult(NamedTuple):
    """Stores the per-sentence scoring outputs."""
    text:        str
    pillar:      str
    label:       str    # "SPECIFIC" or "VAGUE"
    p_vague:     float  # raw P(VAGUE) from model
    p_specific:  float  # raw P(SPECIFIC) from model
    specificity: float  # 1 - p_vague, used in scoring (0–1)
    quant_score: float  # quantitative density heuristic (0–1)
    cite_score:  float  # citation / evidence heuristic (0–1)
    coverage:    float  # topic coverage for this pillar (0–1)


class PillarScore(NamedTuple):
    """Aggregated score for one ESG pillar."""
    pillar:        str
    score:         float   # 0–100
    n_sentences:   int
    vague_count:   int
    specific_count: int
    top_vague_sentences: list[str]


class AuthenticityReport(NamedTuple):
    """Full authenticity report for a text chunk."""
    final_score:      float         # 0–100 weighted average across pillars
    interpretation:   str           # human-readable label
    e_score:          PillarScore
    s_score:          PillarScore
    g_score:          PillarScore
    sentence_results: list[SentenceResult]


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

# def load_model_and_tokenizer(model_path: Path | None = None):
#     """
#     Load the fine-tuned ClimateBERT model and its tokenizer.

#     Falls back to the base pre-trained model if the fine-tuned checkpoint
#     does not exist — useful for testing the pipeline before training.

#     Parameters
#     ----------
#     model_path : Path | None
#         Directory containing the fine-tuned checkpoint.  Defaults to
#         ``config.FINE_TUNED_MODEL_PATH``.

#     Returns
#     -------
#     tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]
#     """
#     if model_path is None:
#         model_path = config.FINE_TUNED_MODEL_PATH

#     if model_path.exists():
#         load_from = str(model_path)
#         logger.info("Loading fine-tuned model from %s …", load_from)
#     else:
#         logger.warning(
#             "Fine-tuned model not found at %s.  "
#             "Loading base model '%s' — run train.py first for best results.",
#             model_path, config.BASE_MODEL_NAME,
#         )
#         load_from = config.BASE_MODEL_NAME

#     tokenizer = AutoTokenizer.from_pretrained(load_from, use_fast=False)
#     model = AutoModelForSequenceClassification.from_pretrained(load_from)

#     # Device selection
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         logger.info("Inference on CUDA GPU: %s", torch.cuda.get_device_name(0))
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#         logger.info("Inference on Apple Silicon MPS.")
#     else:
#         device = torch.device("cpu")
#         logger.info("Inference on CPU.")

#     model.to(device)
#     model.eval()
#     return tokenizer, model, device

from transformers import RobertaConfig, RobertaForSequenceClassification, AutoTokenizer

def load_model_and_tokenizer(model_path=None):
    # 1. Hardcode the path to your fine-tuned weights
    load_from = r"C:\Users\aditi\OneDrive\Desktop\esg_scorer\models\climatebert-esg-scorer"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading tokenizer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-f")
    
    print(f"Loading model weights from {load_from}...")
    
    # 2. MANUALLY DEFINE CONFIG TO OVERRIDE THE BROKEN JSON
    # This ensures "model_type": "roberta" is forced into the system
    config = RobertaConfig.from_pretrained(
        "climatebert/distilroberta-base-climate-f", 
        num_labels=2,
        id2label={0: "SPECIFIC", 1: "VAGUE"},
        label2id={"SPECIFIC": 0, "VAGUE": 1}
    )

    # 3. Load the model using the explicit RoBERTa class instead of 'Auto'
    model = RobertaForSequenceClassification.from_pretrained(
        load_from, 
        config=config,
        ignore_mismatched_sizes=True
    )
    
    model.to(device)
    model.eval()
    return tokenizer, model, device


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """
    Split a multi-sentence ESG text chunk into individual sentences.

    Uses a simple regex approach robust to abbreviations (e.g., "Scope 3.2").
    Sentences shorter than 15 characters are dropped (usually artefacts).

    Parameters
    ----------
    text : str
        Raw ESG report text.

    Returns
    -------
    list[str]
        List of individual sentence strings.
    """
    # Split on sentence-ending punctuation followed by whitespace + capital
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 15]
    logger.debug("Split text into %d sentences.", len(sentences))
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic sub-score functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_quant_density(sentence: str) -> float:
    """
    Compute the quantitative density score for a single sentence.

    A sentence is considered "quantitative" if it contains a number,
    percentage, or measurable unit — signals of concrete ESG evidence.

    Returns
    -------
    float
        1.0 if quantitative, 0.0 otherwise.
    """
    pattern = r"\b\d+[\.,]?\d*\s*(%|percent|MW|GWh|tCO2|tonnes?|metric tons?|USD|million|billion|km²?)\b"
    return 1.0 if re.search(pattern, sentence, re.IGNORECASE) else 0.0


def compute_citation_score(sentence: str) -> float:
    """
    Estimate citation / evidence score for a sentence.

    Looks for references to audits, certifications, third-party standards,
    or formal frameworks — all indicators of verifiability (Slide 12).

    Returns
    -------
    float
        Score between 0.0 (no evidence markers) and 1.0 (multiple markers).
    """
    evidence_patterns = [
        r"\bGRI\b", r"\bSASB\b", r"\bTCFD\b", r"\bISO\s*\d+",
        r"\baudited?\b", r"\bcertified?\b", r"\bverified?\b",
        r"\bthird.?party\b", r"\bindependent\b",
        r"\bassurance\b", r"\bUN SDG\b", r"\bParis Agreement\b",
        r"\bSBTi\b", r"\bEU SFDR\b",
    ]
    hits = sum(bool(re.search(p, sentence, re.IGNORECASE)) for p in evidence_patterns)
    return min(hits / 2.0, 1.0)  # normalise: ≥2 patterns → full score


def compute_hedge_score(sentence: str) -> float:
    """
    Count hedge / vague language patterns in a sentence (Slide 3).

    Returns a vagueness signal (0=no hedging, 1=heavy hedging).
    This is complementary to the model's P(VAGUE) — a linguistic
    cross-check rather than a replacement.

    Returns
    -------
    float
        Hedge density between 0.0 and 1.0.
    """
    hits = sum(
        bool(re.search(p, sentence, re.IGNORECASE))
        for p in config.HEDGE_PATTERNS
    )
    return min(hits / 3.0, 1.0)  # normalise: ≥3 patterns → max hedge


def compute_topic_coverage(sentence: str, pillar: str) -> float:
    """
    Measure how well a sentence covers its assigned ESG pillar's key topics.

    Parameters
    ----------
    sentence : str
        Sentence text.
    pillar : str
        One of "E", "S", "G".

    Returns
    -------
    float
        Proportion of pillar keywords present (0.0–1.0), capped at 1.0.
    """
    kws = config.PILLAR_KEYWORDS.get(pillar, [])
    if not kws:
        return 0.0
    hits = sum(kw in sentence.lower() for kw in kws)
    return min(hits / max(len(kws) * 0.1, 1.0), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring logic
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_sentences(
    sentences: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> list[SentenceResult]:
    """
    Run ClimateBERT inference on a list of sentences and compute all
    sub-scores for each.

    The model produces logits over {SPECIFIC, VAGUE}.  Softmax converts
    these to probabilities.  P(SPECIFIC) = 1 − P(VAGUE) is the core
    specificity signal, which feeds into the weighted scoring formula.

    Parameters
    ----------
    sentences : list[str]
        Pre-split ESG sentences.
    tokenizer : PreTrainedTokenizer
        ClimateBERT tokenizer.
    model : PreTrainedModel
        Fine-tuned classification model.
    device : torch.device
        Target device.

    Returns
    -------
    list[SentenceResult]
        One result per sentence with model probabilities + heuristic scores.
    """
    results = []

    for sent in sentences:
        # Infer pillar for this sentence
        pillar = _infer_pillar_local(sent)

        # Tokenise
        enc = tokenizer(
            sent,
            truncation=True,
            padding="max_length",
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Forward pass
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

        p_specific = float(probs[0])
        p_vague    = float(probs[1])
        label      = "VAGUE" if p_vague >= 0.5 else "SPECIFIC"

        results.append(SentenceResult(
            text        = sent,
            pillar      = pillar,
            label       = label,
            p_vague     = p_vague,
            p_specific  = p_specific,
            specificity = 1.0 - p_vague,
            quant_score = compute_quant_density(sent),
            cite_score  = compute_citation_score(sent),
            coverage    = compute_topic_coverage(sent, pillar),
        ))

        logger.debug(
            "[%s | %s] P(VAGUE)=%.3f  Q=%.2f  C=%.2f  — %s",
            pillar, label, p_vague,
            results[-1].quant_score, results[-1].cite_score,
            sent[:80],
        )

    return results


def _infer_pillar_local(text: str) -> str:
    """Local copy of pillar inference (avoids circular import from config)."""
    text_lower = text.lower()
    hits = {
        p: sum(kw in text_lower for kw in kws)
        for p, kws in config.PILLAR_KEYWORDS.items()
    }
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "E"


def aggregate_pillar_score(
    results: list[SentenceResult],
    pillar: str,
) -> PillarScore:
    """
    Aggregate sentence-level results into a 0–100 pillar score.

    Scoring formula for each sentence (Slide 12):
        pillar_component = (
            0.25 × specificity       +
            0.20 × quant_density     +
            0.20 × citation_score    +
            0.20 × claim_verif_proxy +  ← hedge_score inverted
            0.15 × topic_coverage
        )

    The pillar score is the mean component score across all sentences
    assigned to that pillar, scaled to 0–100.

    Parameters
    ----------
    results : list[SentenceResult]
        All sentence results for the full document.
    pillar : str
        "E", "S", or "G".

    Returns
    -------
    PillarScore
        Aggregated pillar score and diagnostics.
    """
    pillar_sents = [r for r in results if r.pillar == pillar]
    w = config.SCORE_COMPONENT_WEIGHTS

    if not pillar_sents:
        logger.warning(
            "No sentences found for pillar '%s'.  Assigning score 0.", pillar
        )
        return PillarScore(
            pillar=pillar, score=0.0, n_sentences=0,
            vague_count=0, specific_count=0, top_vague_sentences=[],
        )

    component_scores = []
    for r in pillar_sents:
        claim_verif_proxy = 1.0 - compute_hedge_score(r.text)   # invert hedge
        component = (
            w["vagueness"]      * r.specificity   +
            w["quant_density"]  * r.quant_score   +
            w["citation"]       * r.cite_score     +
            w["claim_verif"]    * claim_verif_proxy +
            w["topic_coverage"] * r.coverage
        )
        component_scores.append(component)

    raw_score = float(np.mean(component_scores))
    scaled_score = round(raw_score * 100, 2)

    vague_sents = [r for r in pillar_sents if r.label == "VAGUE"]
    top_vague = sorted(vague_sents, key=lambda r: r.p_vague, reverse=True)[:3]

    return PillarScore(
        pillar          = pillar,
        score           = scaled_score,
        n_sentences     = len(pillar_sents),
        vague_count     = len(vague_sents),
        specific_count  = len(pillar_sents) - len(vague_sents),
        top_vague_sentences = [r.text for r in top_vague],
    )


def compute_final_score(e: PillarScore, s: PillarScore, g: PillarScore) -> float:
    """
    Compute the weighted final authenticity score.

    Formula (Slide 12):
        Final Score = 0.40 × E-Score + 0.35 × S-Score + 0.25 × G-Score

    Returns
    -------
    float
        Authenticity score 0–100.
    """
    w = config.PILLAR_WEIGHTS
    return round(w["E"] * e.score + w["S"] * s.score + w["G"] * g.score, 2)


def interpret_score(score: float) -> str:
    """
    Map a numeric authenticity score to a human-readable interpretation.

    Interpretation bands are derived from the evaluation targets
    in Slide 14 (>80 F1 target implies high-quality classification).

    Parameters
    ----------
    score : float
        Authenticity score 0–100.

    Returns
    -------
    str
        Interpretation label with emoji indicator.
    """
    if score >= 80:
        return "✅ Highly Specific — strong quantitative evidence, verifiable claims"
    elif score >= 60:
        return "🟡 Mostly Specific — some vague language, generally well-supported"
    elif score >= 40:
        return "🟠 Mixed           — notable greenwashing signals present"
    elif score >= 20:
        return "🔴 Mostly Vague    — significant hedge language, few concrete facts"
    else:
        return "❌ Highly Vague    — pervasive greenwashing language detected"


# ─────────────────────────────────────────────────────────────────────────────
# Top-level scoring function
# ─────────────────────────────────────────────────────────────────────────────

def score_text(
    text: str,
    tokenizer,
    model,
    device: torch.device,
) -> AuthenticityReport:
    """
    Score an ESG report chunk and return a full AuthenticityReport.

    This is the primary public API of this module.  It orchestrates
    sentence splitting → ClimateBERT inference → pillar aggregation →
    final score computation.

    Parameters
    ----------
    text : str
        Raw ESG report text (one or more sentences).
    tokenizer : PreTrainedTokenizer
        ClimateBERT tokenizer (from load_model_and_tokenizer).
    model : PreTrainedModel
        Fine-tuned classification model.
    device : torch.device
        Target compute device.

    Returns
    -------
    AuthenticityReport
        Complete scoring report including per-pillar breakdown and
        flagged vague sentences.
    """
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("Input text could not be split into any valid sentences.")

    logger.info("Scoring %d sentences …", len(sentences))
    results = score_sentences(sentences, tokenizer, model, device)

    e_score = aggregate_pillar_score(results, "E")
    s_score = aggregate_pillar_score(results, "S")
    g_score = aggregate_pillar_score(results, "G")

    final = compute_final_score(e_score, s_score, g_score)

    return AuthenticityReport(
        final_score      = final,
        interpretation   = interpret_score(final),
        e_score          = e_score,
        s_score          = s_score,
        g_score          = g_score,
        sentence_results = results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: AuthenticityReport) -> None:
    """Print a formatted authenticity report to stdout."""
    sep = "─" * 70
    print(f"\n{sep}")
    print("  ESG REPORT AUTHENTICITY SCORE")
    print(sep)
    print(f"  Final Score   :  {report.final_score:>6.2f} / 100")
    print(f"  Interpretation:  {report.interpretation}")
    print(sep)
    print("  PILLAR BREAKDOWN")
    print(sep)
    for ps in (report.e_score, report.s_score, report.g_score):
        weight = config.PILLAR_WEIGHTS[ps.pillar]
        pillar_full = {"E": "Environmental", "S": "Social", "G": "Governance"}[ps.pillar]
        bar_len = int(ps.score / 5)     # max 20 chars
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f"  [{ps.pillar}] {pillar_full:<16} │ {bar} │ "
            f"{ps.score:>5.1f}/100  "
            f"(weight {weight:.0%})  "
            f"[{ps.n_sentences} sents | "
            f"{ps.specific_count} specific / {ps.vague_count} vague]"
        )
    print(sep)
    print("  TOP FLAGGED SENTENCES (by vagueness probability)")
    print(sep)
    flagged = sorted(
        [r for r in report.sentence_results if r.label == "VAGUE"],
        key=lambda r: r.p_vague,
        reverse=True,
    )[:5]
    if flagged:
        for i, r in enumerate(flagged, 1):
            print(f"  {i}. [{r.pillar}] P(VAGUE)={r.p_vague:.3f}")
            print(f"      '{r.text[:120]}{'...' if len(r.text) > 120 else ''}'")
            
    else:
        print("  No VAGUE sentences detected.")
    print(sep + "\n")


def report_to_dict(report: AuthenticityReport) -> dict:
    """Serialise an AuthenticityReport to a JSON-compatible dict."""
    return {
        "final_score": report.final_score,
        "interpretation": report.interpretation,
        "pillars": {
            ps.pillar: {
                "score": ps.score,
                "n_sentences": ps.n_sentences,
                "specific_count": ps.specific_count,
                "vague_count": ps.vague_count,
                "top_vague_sentences": ps.top_vague_sentences,
            }
            for ps in (report.e_score, report.s_score, report.g_score)
        },
        "sentences": [
            {
                "text": r.text,
                "pillar": r.pillar,
                "label": r.label,
                "p_vague": round(r.p_vague, 4),
                "p_specific": round(r.p_specific, 4),
            }
            for r in report.sentence_results
        ],
    }


from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    print(f"📄 Extracting text from: {pdf_path}")
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        # Extract text and add a newline to keep sentences separate
        full_text += page.extract_text() + "\n"
    return full_text

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load model and tokenizer once at the start
    # Passing None uses the default path in your config
    tokenizer, model, device = load_model_and_tokenizer()

    while True:
        print("\n" + "="*50)
        print("🌿 ESG AUTHENTICITY SCORER - MAIN MENU")
        print("="*50)
        print("1. Analyze a PDF or Text File")
        print("2. Paste a Paragraph (Interactive)")
        print("3. Quit")
        
        choice = input("\nSelect an option (1-3): ").strip()

        if choice == '1':
            file_path = input("📂 Enter the full path to your file (PDF or TXT): ").strip()
            # Remove quotes if user copied path as "C:\path\file.pdf"
            file_path = file_path.replace('"', '').replace("'", "")
            fp = Path(file_path)
            
            if not fp.exists():
                print(f"❌ Error: File not found at {file_path}")
                continue
            
            print(f"⌛ Processing {fp.name}...")
            if fp.suffix.lower() == ".pdf":
                raw_text = extract_text_from_pdf(fp)
            else:
                raw_text = fp.read_text(encoding="utf-8")
            
            report = score_text(raw_text, tokenizer, model, device)
            print_report(report)

        elif choice == '2':
            print("\n📝 Paste your text below.")
            print("   (Press Enter TWICE to analyze, or type 'back' to return to menu)")
            lines = []
            while True:
                line = input()
                if line.strip().lower() == 'back':
                    break
                if line == "": # Second enter
                    break
                lines.append(line)
            
            if lines:
                raw_text = " ".join(lines)
                print("⌛ Analyzing text...")
                report = score_text(raw_text, tokenizer, model, device)
                print_report(report)

        elif choice == '3' or choice.lower() == 'quit':
            print("Goodbye!")
            break
        
        else:
            print("⚠️ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
