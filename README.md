# Text Summarizer with Domain Detection and Feedback

This project summarizes text using two methods: **extractive** and **abstractive**.
It detects the domain of the input text (15 categories) and allows users to choose which summary is better.
User feedback is stored and later used to improve the system using RLHF (Reinforcement Learning from Human Feedback).

---

## Project Structure

```
data/
    feedback_log.jsonl        # Stores user feedback
domain/
    detector.py               # Detects the domain of the text
    keywords.json             # Keywords for domain detection
rlhf/
    feedback_logger.py        # Handles saving user feedback
    policy.py                 # RLHF policy model
    train.py                  # RLHF training script
summarisers/
    extractive.py              # Extractive summarizer
    abstractive.py             # Abstractive summarizer
app.py                        # Main Gradio app
config.py                     # Configuration settings
requirements.txt              # Python dependencies
```

---

## How to Run

1. Install the required packages:

```
pip install -r requirements.txt
```

2. Start the application:

```
python app.py
```

3. Open the link shown in the terminal (e.g., [http://127.0.0.1:7860](http://127.0.0.1:7860)).

4. Enter text, view both summaries, and select which one is better.
   The feedback will be saved automatically.

---


