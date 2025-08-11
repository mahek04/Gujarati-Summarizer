# rlhf/policy.py
"""
ExtractivePolicy module.

Design:
- Given a matrix of sentence embeddings (num_sentences x emb_dim),
  the policy produces a logit per sentence.
- We treat selection as independent Bernoulli actions per sentence
  (sampled from sigmoid(logit)).
- Provides helper methods:
    - forward(embeddings) -> logits
    - sample_actions(logits) -> actions (0/1), log_prob (scalar)
    - greedy_select(logits, top_k) -> deterministic top-k selection

Note: The policy is intentionally small (single or two-layer MLP),
because sentence embeddings are already high-quality semantic vectors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtractivePolicy(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Simple two-layer MLP mapping embedding -> logit
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)  # produce scalar logit per sentence

    def forward(self, sent_embeddings: torch.Tensor) -> torch.Tensor:
        """
        sent_embeddings: (num_sentences, emb_dim) or (batch, num_sentences, emb_dim)
        returns logits of shape (num_sentences,) or (batch, num_sentences)
        """
        single = False
        x = sent_embeddings
        if x.dim() == 2:
            # (num_sentences, emb_dim) -> (num_sentences, 1)
            out = self.fc1(x)
            out = self.act(out)
            out = self.dropout(out)
            out = self.fc2(out)  # (num_sentences, 1)
            logits = out.squeeze(-1)  # (num_sentences,)
            return logits
        elif x.dim() == 3:
            # batch case: (batch, num_sentences, emb_dim)
            b, n, d = x.size()
            x = x.view(b * n, d)
            out = self.fc1(x)
            out = self.act(out)
            out = self.dropout(out)
            out = self.fc2(out)
            logits = out.view(b, n)
            return logits
        else:
            raise ValueError("sent_embeddings must be 2D or 3D tensor")

    @staticmethod
    def sample_actions_and_logprob(logits: torch.Tensor, deterministic: bool = False):
        """
        logits: (num_sentences,) or (batch, num_sentences)
        Returns:
            actions: same shape (0/1)
            log_prob: summed log probability of the sampled action vector (scalar per batch or scalar)
        If deterministic=True -> actions = (sigmoid(logits) > 0.5).float() and log_prob is computed under that action.
        """
        probs = torch.sigmoid(logits)
        if deterministic:
            actions = (probs > 0.5).float()
            # compute log_prob under Bernoulli
            eps = 1e-8
            log_prob = actions * torch.log(probs + eps) + (1 - actions) * torch.log(1 - probs + eps)
            # sum across sentences, and across batch if present
            return actions, log_prob.sum(dim=-1)
        else:
            bern = torch.distributions.Bernoulli(probs=probs)
            actions = bern.sample()
            log_prob = bern.log_prob(actions).sum(dim=-1)  # sum over sentences; shape=(batch,) or scalar
            return actions, log_prob

    @staticmethod
    def greedy_topk_from_logits(logits: torch.Tensor, top_k: int = 3):
        """
        Deterministic top-k selection based on logits.
        logits: (num_sentences,) or (batch, num_sentences)
        Returns a binary mask (0/1) indicating selected sentences.
        """
        if logits.dim() == 1:
            n = logits.size(0)
            top_k = max(1, min(top_k, n))
            vals, idx = torch.topk(logits, k=top_k)
            mask = torch.zeros_like(logits)
            mask[idx] = 1.0
            return mask
        elif logits.dim() == 2:
            b, n = logits.size()
            top_k = max(1, min(top_k, n))
            vals, idx = torch.topk(logits, k=top_k, dim=-1)
            mask = torch.zeros_like(logits)
            # fill masks per-batch
            arange = torch.arange(b, device=logits.device).unsqueeze(-1).repeat(1, top_k)
            mask[arange, idx] = 1.0
            return mask
        else:
            raise ValueError("logits dimension not supported")

