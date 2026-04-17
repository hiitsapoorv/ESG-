"""
config.py — Central Configuration for ESG Report Authenticity Scorer
=====================================================================
All file paths, model identifiers, and hyperparameters live here.
Import this module everywhere else; never hard-code values in scripts.

Architecture Context
--------------------
We fine-tune ``climatebert/distilroberta-base-climate-f`` — a DistilRoBERTa
model pre-trained on 2M+ climate/ESG paragraphs (Webersinke et al., 2022).
Its domain-adapted vocabulary dramatically reduces the out-of-vocabulary
rate for ESG jargon compared with standard RoBERTa, giving us a meaningful
head-start before fine-tuning on our vague/specific sentence labels.

Scoring Formula (from Slide 12)
--------------------------------
    Final Score = 0.40 × E-Score + 0.35 × S-Score + 0.25 × G-Score

Each pillar score is a weighted average of five sub-components:
    • Vagueness Score          (25%) — model output probability
    • Quantitative Density     (20%) — % sentences with numeric data
    • Citation / Evidence      (20%) — reference / audit ratio
    • Claim Verification       (20%) — NLI entailment score
    • Topic Coverage           (15%) — ESG pillar representation
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Root directories  (all paths are relative to this file's parent)
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data"
MODEL_DIR  = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR    = ROOT_DIR / "logs"

# Ensure directories exist at import time
for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset paths
# ─────────────────────────────────────────────────────────────────────────────
# Drop your CSV / Excel files into data/ and update these paths as needed.
GOLD_LABEL_PATH = DATA_DIR / "gold_labels.csv"       # 300 manually labelled rows
AUTO_LABEL_PATH = DATA_DIR / "auto_labels.csv"        # ~4,900 auto-labelled rows
MERGED_DATA_PATH = DATA_DIR / "merged_dataset.csv"   # written by train.py

# Expected column names in both CSVs
TEXT_COL   = "text_chunk"        # sentence / paragraph text
LABEL_COL  = "label"       # 0 = SPECIFIC, 1 = VAGUE
PILLAR_COL = "pillar"      # "E" | "S" | "G"  (optional; inferred if absent)
SOURCE_COL = "source"      # "gold" | "auto"  (added during merge)

# ─────────────────────────────────────────────────────────────────────────────
# Model identifiers
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "climatebert/distilroberta-base-climate-f"
# Fine-tuned checkpoint saved here after training
FINE_TUNED_MODEL_PATH = MODEL_DIR / "climatebert-esg-scorer"
# MLflow experiment name
MLFLOW_EXPERIMENT = "esg-authenticity-scorer"

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
# Adjust BATCH_SIZE / GRAD_ACCUM to match available VRAM / RAM:
#   GPU  16 GB → BATCH_SIZE=16, GRAD_ACCUM=1
#   CPU  only  → BATCH_SIZE=4,  GRAD_ACCUM=4  (effective batch = 16)
BATCH_SIZE           = 8
GRAD_ACCUM_STEPS     = 2       # effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS
NUM_EPOCHS           = 5
LEARNING_RATE        = 2e-5
WEIGHT_DECAY         = 0.01
WARMUP_RATIO         = 0.1    # fraction of total steps used for LR warmup
MAX_SEQ_LENGTH       = 256    # tokens; ESG sentences rarely exceed 200 tokens
VAL_SPLIT_RATIO      = 0.15   # 15% of merged data → validation
RANDOM_SEED          = 42

# Class weights for WeightedTrainer  (set to None → computed from label freq)
# Override with explicit floats if you prefer manual tuning, e.g. [1.0, 2.5]
CLASS_WEIGHTS        = None   # None → auto-computed in train.py

# ─────────────────────────────────────────────────────────────────────────────
# Scoring formula weights  (Slide 12)
# ─────────────────────────────────────────────────────────────────────────────
PILLAR_WEIGHTS = {
    "E": 0.40,   # Environmental
    "S": 0.35,   # Social
    "G": 0.25,   # Governance
}

# Sub-component weights within each pillar score
SCORE_COMPONENT_WEIGHTS = {
    "vagueness":      0.25,   # model P(VAGUE) inverted to specificity
    "quant_density":  0.20,   # % sentences with numeric data
    "citation":       0.20,   # reference / audit mention ratio
    "claim_verif":    0.20,   # NLI entailment confidence
    "topic_coverage": 0.15,   # pillar keyword density
}

# ─────────────────────────────────────────────────────────────────────────────
# Pillar keyword dictionaries  (used for pillar inference & topic coverage)
# ─────────────────────────────────────────────────────────────────────────────
PILLAR_KEYWORDS = {
    "E": [
        "carbon", "emission", "greenhouse", "ghg", "climate", "renewable",
        "energy", "solar", "wind", "water", "waste", "biodiversity",
        "deforestation", "net zero", "scope 1", "scope 2", "scope 3",
        "carbon neutral", "paris agreement", "tcfd",
    ],
    "S": [
        "employee", "worker", "diversity", "inclusion", "gender", "pay gap",
        "health", "safety", "community", "human rights", "supply chain",
        "labour", "training", "well-being", "social impact", "dei",
    ],
    "G": [
        "board", "governance", "audit", "transparency", "executive",
        "remuneration", "shareholder", "compliance", "regulation",
        "anti-corruption", "bribery", "ethics", "disclosure", "policy",
    ],
}

# Hedge / vague language patterns  (Slide 3 — language loopholes)
HEDGE_PATTERNS = [
    r"\baim(s)? to\b", r"\bstriv(e|es|ing) (to|toward)\b",
    r"\bcommitt?ed? to\b", r"\bworking toward(s)?\b", r"\baspir(e|es|ing)\b",
    r"\bseek(s|ing)? to\b", r"\bintend(s)? to\b", r"\bplan(s|ning)? to\b",
    r"\bhop(e|es|ing) to\b", r"\bwhere possible\b", r"\bwhenever feasible\b",
    r"\bto the extent\b", r"\bover time\b", r"\bin due course\b",
    r"\bsoon\b", r"\bnear.?term\b",
]

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation targets  (Slide 14)
# ─────────────────────────────────────────────────────────────────────────────
EVAL_TARGETS = {
    "f1_macro":        0.80,
    "precision":       0.85,
    "auc_roc":         0.85,
    "spearman_corr":   0.65,
    "cohens_kappa":    0.70,
}

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
LOG_FILE      = LOG_DIR / "esg_scorer.log"
LOG_LEVEL     = "INFO"    # "DEBUG" for verbose token-level output
LOG_FORMAT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FMT  = "%Y-%m-%d %H:%M:%S"
