import csv
import random
from pathlib import Path

MAX_DURATION = 300  # seconds
CSV_PATH = Path("movement_csvs/elemental.csv")

def jaccard(a, b):
    a_set, b_set = set(a), set(b)
    return len(a_set & b_set) / len(a_set | b_set) if (a_set | b_set) else 0

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = list(csv.DictReader(f))

unused = reader.copy()
sequence = []
total_duration = 0

current = random.choice(unused)
unused.remove(current)
current['playback_duration'] = min(float(current['duration']), random.uniform(8, 20))
sequence.append(current)
total_duration += current["playback_duration"]

while total_duration < MAX_DURATION and unused:
    current_tags = (
        current["semantic_tags"] + "," + current["mood_tag"] + "," + current["motion_tag"]
    ).split(", ")
    
    scored = []
    for candidate in unused:
        candidate_tags = (
            candidate["semantic_tags"] + "," + candidate["mood_tag"] + "," + candidate["motion_tag"]
        ).split(", ")
        score = jaccard(current_tags, candidate_tags)
        scored.append((score, candidate))
    
    scored = sorted(scored, key=lambda x: -x[0])
    top = [c for s, c in scored[:5] if s > 0.1]

    if not top:
        print(f"⚠️ No strong match found for {current['video_id']}. Selecting random fallback.")
        next_clip = random.choice(unused)
    else:
        next_clip = random.choice(top)
        score = jaccard(
            (current["semantic_tags"] + "," + current["mood_tag"] + "," + current["motion_tag"]).split(", "),
            (next_clip["semantic_tags"] + "," + next_clip["mood_tag"] + "," + next_clip["motion_tag"]).split(", "),
        )
        print(f"🔗 Matched '{current['video_id']}' → '{next_clip['video_id']}' (Jaccard: {score:.2f})")

    unused.remove(next_clip)
    next_clip['playback_duration'] = min(float(next_clip['duration']), random.uniform(8, 20))
    sequence.append(next_clip)
    total_duration += next_clip['playback_duration']
    current = next_clip

# Output the sequence
print("Generated Sequence:")
for i, row in enumerate(sequence):
    print(f"{i+1}. {row['video_id']} ({row['duration']}s → playing {row['playback_duration']:.1f}s)")

print(f"\n🕒 Total Duration: {total_duration:.1f} seconds")