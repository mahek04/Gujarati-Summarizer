# rlhf/train.py
"""
Training script for RLHF-style policy gradients on the ExtractivePolicy.

Assumptions:
- feedback.jsonl is located at project root (or pass path).
- Each line is a JSON object with keys:
    - "timestamp", "domain", "original_text", "extractive_summary", "abstractive_summary", "chosen"
- Reward: 1.0 if chosen == "extractive", else 0.0
- We train by sampling actions from the policy for each example's sentence embeddings
  and applying REINFORCE: loss = - (reward - baseline) * log_prob.
- A running EMA baseline is used to reduce variance.

Usage (CLI-ish):
    python rlhf/train.py --feedback_path data/feedback.jsonl --save_path models/extractive_policy.pth \
                         --epochs 3 --batch_size 8 --lr 1e-4
"""

import os
import json
import argparse
from tqdm import tqdm
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import nltk

from rlhf.policy import ExtractivePolicy

nltk.download("punkt", quiet=True)


def split_sentences(text: str) -> List[str]:
    # Use nltk punkt; for Gujarati, punctuation might differ, but this is a start.
    # You may customize a Gujarati-aware splitter later.
    sents = nltk.sent_tokenize(text)
    # Filter out empty or whitespace-only sentences
    sents = [s.strip() for s in sents if s.strip()]
    return sents


def load_feedbacks(path: str) -> List[dict]:
    feedbacks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                feedbacks.append(obj)
            except json.JSONDecodeError:
                # Maybe the file is a JSON array; try loading everything
                f.seek(0)
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    raise
    return feedbacks


def prepare_examples(feedbacks: List[dict]) -> List[Tuple[List[str], float]]:
    """
    Convert feedback entries to (sentences, reward) pairs.
    reward = 1.0 if chosen == 'extractive', else 0.0
    """
    examples = []
    for fb in feedbacks:
        orig = fb.get("original_text") or fb.get("text") or ""
        chosen = fb.get("chosen", "").lower()
        reward = 1.0 if chosen.startswith("extract") else 0.0
        sents = split_sentences(orig)
        if len(sents) == 0:
            continue
        examples.append((sents, reward))
    return examples


def collate_batch(batch: List[Tuple[List[str], float]], embedder: SentenceTransformer, device: torch.device):
    """
    Given a batch of examples (sents_list, reward),
    compute embeddings and return:
      - embeddings tensor: (batch, max_n, emb_dim)
      - mask tensor: (batch, max_n) -> 1 for valid sentence positions, 0 for padding
      - rewards tensor: (batch,)
    """
    batch_sents = [ex[0] for ex in batch]
    rewards = torch.tensor([ex[1] for ex in batch], dtype=torch.float32, device=device)

    # flatten all sentences to compute embeddings in one go
    all_sents = [s for sents in batch_sents for s in sents]
    if len(all_sents) == 0:
        return None

    # Use embedder to get embeddings (returns numpy or torch depending on settings)
    embeddings_all = embedder.encode(all_sents, convert_to_tensor=True, show_progress_bar=False)
    emb_dim = embeddings_all.size(-1)

    # rebuild batch tensor with padding
    max_n = max(len(s) for s in batch_sents)
    batch_embeddings = torch.zeros((len(batch), max_n, emb_dim), device=device)
    mask = torch.zeros((len(batch), max_n), device=device)

    idx = 0
    for i, sents in enumerate(batch_sents):
        for j, _ in enumerate(sents):
            batch_embeddings[i, j] = embeddings_all[idx].to(device)
            mask[i, j] = 1.0
            idx += 1

    return batch_embeddings, mask, rewards


def train(
    feedback_path: str = "data/feedback.jsonl",
    model_save_path: str = "models/extractive_policy.pth",
    model_emb_model: str = "paraphrase-MiniLM-L6-v2",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-4,
    top_k: int = 3,
    device_str: str = None,
    save_every: int = 1,
):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)

    feedbacks = load_feedbacks(feedback_path)
    if len(feedbacks) == 0:
        raise ValueError(f"No feedback found in {feedback_path}")

    examples = prepare_examples(feedbacks)
    if len(examples) == 0:
        raise ValueError("No usable examples after preprocessing.")

    # SentenceTransformer embedder
    embedder = SentenceTransformer(model_emb_model, device=str(device))
    emb_dim = embedder.get_sentence_embedding_dimension()

    # Create policy
    policy = ExtractivePolicy(emb_dim=emb_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Baseline for REINFORCE: running exponential moving average of rewards
    baseline = None
    ema_alpha = 0.9

    # Training loop
    print(f"Starting training on device={device} | examples={len(examples)} | batch_size={batch_size}")
    for epoch in range(1, epochs + 1):
        # Shuffle
        import random
        random.shuffle(examples)

        # Mini-batches
        batches = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]
        epoch_loss = 0.0
        pbar = tqdm(batches, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            collated = collate_batch(batch, embedder, device)
            if collated is None:
                continue
            batch_embeddings, mask, rewards = collated  # shapes: (B, N, D), (B, N), (B,)
            B, N, D = batch_embeddings.size()

            # Compute logits with policy: shape (B, N)
            logits = policy(batch_embeddings)  # (B, N)

            # For each batch element sample actions and obtain log_prob
            actions, log_probs = policy.sample_actions_and_logprob(logits, deterministic=False)
            # Ensure shape: actions (B, N), log_probs (B,)
            if log_probs.dim() == 0:
                log_probs = log_probs.unsqueeze(0)

            # Compute baseline
            batch_reward_mean = rewards.mean().item()
            if baseline is None:
                baseline = batch_reward_mean
            else:
                baseline = ema_alpha * baseline + (1 - ema_alpha) * batch_reward_mean

            # Advantage
            advantage = rewards - baseline  # shape (B,)

            # Policy gradient loss (negative because we ascend reward)
            # We want to maximize expected reward: L = - E[(r - b) * log_prob]
            loss = - (advantage.detach() * log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{epoch_loss:.4f}", "baseline": f"{baseline:.4f}"})

        # End epoch
        if epoch % save_every == 0:
            torch.save(policy.state_dict(), model_save_path)
            print(f"Saved policy to {model_save_path} after epoch {epoch}")

    # Final save
    torch.save(policy.state_dict(), model_save_path)
    print("Training complete. Model saved to:", model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback_path", type=str, default="data/feedback.jsonl")
    parser.add_argument("--model_save_path", type=str, default="models/extractive_policy.pth")
    parser.add_argument("--model_emb_model", type=str, default="paraphrase-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        feedback_path=args.feedback_path,
        model_save_path=args.model_save_path,
        model_emb_model=args.model_emb_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        top_k=args.top_k,
        device_str=args.device,
    )
