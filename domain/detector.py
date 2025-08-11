# domain/detector.py
"""
DomainDetector for Gujarati text.
- Loads a keywords.json file (UTF-8) mapping domain -> list of Gujarati keywords.
- Counts keyword occurrences using word-boundary aware regex (works with Gujarati script).
- Returns the top domain (or "જનરલ" if no keyword matches).
- Also provides a method to return the full score dictionary.

Usage:
    detector = DomainDetector("keywords.json")
    top = detector.detect_domain(text)
    scores = detector.score_text(text)
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Tuple, Optional


class DomainDetector:
    def __init__(self, keywords_file: str = "keywords.json"):
        if not os.path.exists(keywords_file):
            raise FileNotFoundError(f"keywords file not found: {keywords_file}")

        with open(keywords_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Normalize domains and keywords; store compiled regex for each keyword
        self.domain_keywords: Dict[str, List[str]] = {}
        self.domain_patterns: Dict[str, List[re.Pattern]] = {}

        for domain, kws in raw.items():
            # keep domain label as-is (likely Gujarati); normalize keywords
            normalized_kws = [self._normalize_text(str(k)) for k in kws if k]
            self.domain_keywords[domain] = normalized_kws
            patterns = []
            for kw in normalized_kws:
                # word-boundary regex: \b doesn't work with all Unicode scripts consistently,
                # so use lookaround to approximate boundaries (start/end or whitespace/punct)
                # This pattern matches the keyword as a whole token if possible.
                pat = re.compile(rf"(?<!\S){re.escape(kw)}(?!\S)", flags=re.IGNORECASE)
                patterns.append(pat)
            self.domain_patterns[domain] = patterns

    def _normalize_text(self, text: str) -> str:
        """
        Normalize unicode, strip extra whitespace, and lower-case.
        Keeps Gujarati script intact.
        """
        if text is None:
            return ""
        # NFC normalization helps with composed/decomposed forms
        t = unicodedata.normalize("NFC", text)
        t = t.strip()
        # Lowercasing for Gujarati won't change script but keeps logic consistent if mixed with english
        try:
            t = t.lower()
        except Exception:
            pass
        return t

    def score_text(self, text: str) -> Dict[str, int]:
        """
        Returns a dict: domain -> score (sum of keyword match counts).
        """
        if not text:
            return {domain: 0 for domain in self.domain_keywords}

        norm_text = self._normalize_text(text)

        scores: Dict[str, int] = {}
        for domain, patterns in self.domain_patterns.items():
            count = 0
            for pat in patterns:
                # findall returns list of non-overlapping matches
                try:
                    matches = pat.findall(norm_text)
                    count += len(matches)
                except re.error:
                    # fallback: simple substring count
                    kw = pat.pattern
                    count += norm_text.count(kw)
            scores[domain] = count
        return scores

    def detect_domain(self, text: str, return_scores: bool = False) -> Tuple[str, Optional[Dict[str, int]]]:
        """
        Detect top domain for the given text.
        - If return_scores is False: returns (top_domain, None)
        - If return_scores is True: returns (top_domain, scores_dict)
        If no keywords match, returns ("જનરલ", scores_dict) where "જનરલ" means general.
        """
        scores = self.score_text(text)
        # pick domain with highest score
        top_domain = max(scores, key=scores.get)
        if scores[top_domain] == 0:
            top_domain_label = "જનરલ"  # Gujarati for "general"
        else:
            top_domain_label = top_domain

        if return_scores:
            return top_domain_label, scores
        else:
            return top_domain_label, None


# Utility convenience function (module-level)
_default_detector: Optional[DomainDetector] = None


def get_default_detector(keywords_file: str = "keywords.json") -> DomainDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = DomainDetector(keywords_file)
    return _default_detector


if __name__ == "__main__":
    # quick local test example
    DET_FILE = "keywords.json"
    if not os.path.exists(DET_FILE):
        print(f"Place your Gujarati keywords in {DET_FILE} and re-run.")
    else:
        d = DomainDetector(DET_FILE)
        sample = "ભારત ક્રિકેટ મેચમાં કેપ્ટનની સફળતાથી ટીમને વિજય મળ્યો."
        top, scores = d.detect_domain(sample, return_scores=True)
        print("Top domain:", top)
        print("Scores:", scores)
