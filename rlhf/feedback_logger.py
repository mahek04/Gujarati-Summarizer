import json
from config import FEEDBACK_FILE

def log_feedback(domain, text, extractive_summary, abstractive_summary, chosen):
    feedback_entry = {
        "domain": domain,
        "text": text,
        "extractive_summary": extractive_summary,
        "abstractive_summary": abstractive_summary,
        "chosen": chosen
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_entry) + "\n")
