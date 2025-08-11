# config.py

# -------- File Paths --------
KEYWORDS_FILE = "keywords.json"
FEEDBACK_FILE = "feedback.jsonl"

# -------- Summarization Models --------
EXTRACTIVE_MODEL_NAME = "facebook/bart-large-cnn"  # Example for extractive
ABSTRACTIVE_MODEL_NAME = "google/pegasus-xsum"     # Example for abstractive

# -------- Domain Detection --------
DOMAIN_MATCH_THRESHOLD = 0.15  # Minimum keyword match ratio to assign a domain

# -------- UI Settings --------
MAX_INPUT_CHARS = 2000         # Limit for text input
MAX_SUMMARY_SENTENCES = 5      # For extractive summarizer

# -------- Training --------
RLHF_TRAINING_EPOCHS = 3
RLHF_LEARNING_RATE = 1e-5

# Paths
KEYWORDS_FILE = "domain/keywords.json"
FEEDBACK_FILE = "data/feedback_log.jsonl"

# Model names
EXTRACTIVE_MODEL_NAME = "facebook/bart-large-cnn"
ABSTRACTIVE_MODEL_NAME = "google/pegasus-xsum"

# Domain detection threshold
DOMAIN_MATCH_THRESHOLD = 0.15

