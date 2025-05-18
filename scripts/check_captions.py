"""
This script validates and filters video captions for quality control.
It checks captions against a set of criteria to identify potentially problematic or low-quality descriptions.

The script processes a CSV file containing video captions and flags entries that:
1. Are too short (less than 5 words)
2. Contain banned phrases (indicating potential hallucination or incorrect descriptions)
3. Use generic/stock phrases

Input:
- all_captions.csv: CSV file with columns [video_id, caption]

Output:
- flagged_weird_captions.csv: CSV file with columns [video_id, caption, reason]
  containing entries that failed validation checks
"""

import csv

# Phrases that indicate potential hallucination or incorrect descriptions
# These are typically impossible or unlikely to appear in the actual video content
BANNED_PHRASES = ["airplane", "elephant", "unicorn", "flying in the sky", "close-up of an object"]

# Process the input CSV and write flagged entries to output CSV
with open("all_captions.csv") as f, open("flagged_weird_captions.csv", "w") as out:
    reader = csv.reader(f)
    writer = csv.writer(out)
    
    # Write header for output CSV
    writer.writerow(["video_id", "caption", "reason"])
    next(reader)  # skip header row from input file

    # Process each caption
    for row in reader:
        vid, caption = row
        
        # Check if caption is too short (less than 5 words)
        if len(caption.split()) < 5:
            writer.writerow([vid, caption, "too short"])
            
        # Check if caption contains any banned phrases
        elif any(b in caption.lower() for b in BANNED_PHRASES):
            writer.writerow([vid, caption, "banned phrase"])
            
        # Check for generic/stock phrases
        elif caption.lower().startswith("a black and white image"):
            writer.writerow([vid, caption, "generic/stock phrase"])