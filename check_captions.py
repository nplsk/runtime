import csv

BANNED_PHRASES = ["airplane", "elephant", "unicorn", "flying in the sky", "close-up of an object"]

with open("all_captions.csv") as f, open("flagged_weird_captions.csv", "w") as out:
    reader = csv.reader(f)
    writer = csv.writer(out)
    writer.writerow(["video_id", "caption", "reason"])
    next(reader)  # skip header

    for row in reader:
        vid, caption = row
        if len(caption.split()) < 5:
            writer.writerow([vid, caption, "too short"])
        elif any(b in caption.lower() for b in BANNED_PHRASES):
            writer.writerow([vid, caption, "banned phrase"])
        elif caption.lower().startswith("a black and white image"):
            writer.writerow([vid, caption, "generic/stock phrase"])