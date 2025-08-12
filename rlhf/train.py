import json
from config import FEEDBACK_FILE
from rlhf.policy import policy

def train_policy():
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            policy.update(entry["chosen"])

    print("Training complete. Stats:", policy.stats)

if __name__ == "__main__":
    train_policy()

