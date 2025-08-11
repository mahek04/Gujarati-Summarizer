# summarizers/abstractive.py
"""
Abstractive Summarizer
- Placeholder for API call or local model.
- Default: Uses Hugging Face Transformers (mT5 or BART) for multilingual summarization.
- You can later replace `summarize()` with Sutra API or OpenAI API calls.
"""

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class AbstractiveSummarizer:
    def __init__(self, model_name: str = "google/mt5-base", max_input_length: int = 512, max_output_length: int = 150):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def summarize(self, text: str, target_lang: Optional[str] = None) -> str:
        """
        Generate abstractive summary.
        - target_lang: Optional language code (e.g., 'gu' for Gujarati, 'en' for English)
        """
        if not text.strip():
            return ""

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True
        ).to(self.device)

        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=self.max_output_length,
            early_stopping=True
        )

        # Decode output
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
