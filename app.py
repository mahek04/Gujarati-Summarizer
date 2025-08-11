import gradio as gr
import logging
from config import (
    KEYWORDS_FILE, FEEDBACK_FILE,
    MAX_INPUT_CHARS
)
from domain.detector import detect_domain
from summarisers.extractive import extractive_summarize
from summarisers.abstractive import abstractive_summarize
from rlhf.feedback_logger import FeedbackLogger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feedback logger instance
feedback_logger = FeedbackLogger(FEEDBACK_FILE)


def summarize_text(text):
    """Detect domain, summarize text with both methods."""
    if not text.strip():
        return "No input provided.", "", ""

    if len(text) > MAX_INPUT_CHARS:
        return f"⚠️ Input too long. Limit is {MAX_INPUT_CHARS} characters.", "", ""

    # Domain detection
    domain = detect_domain(text)
    logging.info(f"Detected domain: {domain}")

    # Summaries
    extractive_summary = extractive_summarize(text)
    abstractive_summary = abstractive_summarize(text)

    return domain, extractive_summary, abstractive_summary


def record_feedback(domain, text, extractive_summary, abstractive_summary, choice):
    """Save feedback for RLHF training."""
    if choice not in ["extractive", "abstractive"]:
        return "⚠️ Please select which summary is better."

    feedback_logger.log_feedback(
        domain=domain,
        text=text,
        extractive_summary=extractive_summary,
        abstractive_summary=abstractive_summary,
        chosen=choice
    )

    logging.info(f"Feedback recorded: {choice} chosen for domain {domain}")
    return "✅ Feedback recorded. Thank you!"


# ---------------- UI ---------------- #
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Text Summarizer with Domain Detection + RLHF")
    gr.Markdown(
        "Enter your text, get summaries from **Extractive** and **Abstractive** models, "
        "pick the better one, and help the system improve."
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter your text",
                lines=10,
                placeholder="Paste your text here...",
            )
            summarize_btn = gr.Button("Summarize")

        with gr.Column():
            domain_output = gr.Label(label="Detected Domain")
            extractive_output = gr.Textbox(label="Extractive Summary", lines=5)
            abstractive_output = gr.Textbox(label="Abstractive Summary", lines=5)

    with gr.Row():
        feedback_choice = gr.Radio(
            ["extractive", "abstractive"],
            label="Which summary is better?",
        )
        feedback_btn = gr.Button("Submit Feedback")
        feedback_status = gr.Label()

    # Events
    summarize_btn.click(
        summarize_text,
        inputs=[text_input],
        outputs=[domain_output, extractive_output, abstractive_output]
    )

    feedback_btn.click(
        record_feedback,
        inputs=[domain_output, text_input, extractive_output, abstractive_output, feedback_choice],
        outputs=[feedback_status]
    )

if __name__ == "__main__":
    demo.launch()
