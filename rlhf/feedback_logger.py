# feedback_logger.py
"""
Feedback Logger
---------------
Logs user feedback for RLHF-style learning.
Stores:
- domain detected
- original input text
- extractive summary
- abstractive summary
- which one user preferred
- timestamp
"""

import json
import os
from datetime import datetime
from typing import Literal

class FeedbackLogger:
    def __init__(self, log_file: str = "feedback.jsonl"):
        self.log_file = log_file
        # Ensure file exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                pass

    def log_feedback(
        self,
        domain: str,
        original_text: str,
        extractive_summary: str,
        abstractive_summary: str,
        chosen: Literal["extractive", "abstractive"]
    ):
        """
        Append feedback entry to JSONL file.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "domain": domain,
            "original_text": original_text,
            "extractive_summary": extractive_summary,
            "abstractive_summary": abstractive_summary,
            "chosen": chosen
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"✅ Feedback logged: {chosen} summarizer preferred for domain '{domain}'")
