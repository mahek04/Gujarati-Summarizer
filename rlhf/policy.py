# Simple placeholder policy — learns later from feedback
class Policy:
    def __init__(self):
        # Track performance stats
        self.stats = {"extractive": 0, "abstractive": 0}

    def update(self, chosen):
        if chosen in self.stats:
            self.stats[chosen] += 1

    def choose(self, domain=None):
        # Currently defaults to extractive; can be improved
        return "extractive" if self.stats["extractive"] >= self.stats["abstractive"] else "abstractive"

policy = Policy()


