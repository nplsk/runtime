# decay_tracker.py
import time

class TagDecayTracker:
    def __init__(self, decay_rate=0.001):
        self.tags = {}  # tag: (score, last_updated)
        self.decay_rate = decay_rate

    def update(self, tag):
        now = time.time()
        if tag in self.tags:
            score, last = self.tags[tag]
            elapsed = now - last
            decayed = score * (1 - self.decay_rate * elapsed)
            new_score = min(1.0, decayed + 0.2)
        else:
            new_score = 1.0
        self.tags[tag] = (new_score, now)

    def get_scores(self):
        now = time.time()
        scores = {}
        for tag, (score, last) in list(self.tags.items()):
            elapsed = now - last
            decayed = score * (1 - self.decay_rate * elapsed)
            if decayed <= 0.01:
                del self.tags[tag]
            else:
                scores[tag] = round(decayed, 3)
                self.tags[tag] = (decayed, now)
        return scores