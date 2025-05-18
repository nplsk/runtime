# assign_phases.py
import os
import json

INPUT_DIR = "/Volumes/CORSAIR/DESCRIPTIONS"
OUTPUT_FILE = "movement_archive.json"

phase_assignments = {
    "orientation": [],
    "elemental": [],
    "built": [],
    "people": [],
    "blur": []
}

elemental_tags = {
    "nature", "forest", "water", "snow", "serene", "calm", "still", "ambient_light",
    "diffused_light", "natural_light", "grassy_field", "rocks", "earth_materials",
    "vegetation", "fog", "trees", "stream", "wildlife", "sky", "ice", "clouds", "sun",
    "cold", "oak_tree", "rain", "dusk", "lake", "moss", "neutral_mood", "tranquil",
    "peaceful", "relaxed", "quiet", "natural_elements", "soft_lighting", "low_light",
    "waterfall", "river", "spider_webs", "lava", "volcano", "insect", "swarming"
}

built_tags = {
    "interior", "architecture", "urban", "structure", "object", "enclosed_space", "room",
    "bookshelves", "chairs", "sofas", "living_room", "apartment", "furniture", "kitchen",
    "bathroom", "books", "domestic", "indoor", "interior_design", "curtains",
    "light_fixture", "mirror", "ceiling", "walls", "floor", "television", "lamp", "window",
    "warm_lighting", "artificial_light", "ambient", "helicopter", "airplane", "airport",
    "abandoned_building", "asphalt", "concrete", "metal", "toilet", "grate", "projection",
    # --- New tags for improved built/people separation ---
    "locker_room", "hotel_room", "waiting_room", "hallway", "office", "bedroom", "bathroom_sink",
    "closet", "garage", "workshop", "elevator", "escalator", "school", "hall", "bench", "urban_alley",
    "hallway_corner", "lecture_hall", "booth", "gym", "bar", "restaurant", "studio_apartment", "laundromat"
}
BUILT_DISQUALIFY_IF_PRESENT = set()
PEOPLE_DISQUALIFY_IF_PRESENT = set()
BLUR_DISQUALIFY_IF_PRESENT = set()

ELEMENTAL_DISQUALIFY_IDS = {"IMG_7769_prores_scene_002"}
# Shared disqualifier tags for elemental phase
ELEMENTAL_DISQUALIFY_IF_PRESENT = {

# Specific video IDs to disqualify from elemental

    # copy all unique tags formerly listed in disqualify_if_present blocks
    "people", "human", "crowd", "group", "event", "elevated_stage", "gathering", "audience", "portrait",
    "musicians", "hug", "kiss", "smile", "companionship", "interaction", "relationship", "guests", "bride",
    "groom", "individual", "dog", "buttons", "woman_walking", "jesus", "dj", "car_interior", "human_body",
    "road", "holiday", "performance", "robot", "studio", "fabric", "gown", "christmas", "celebration",
    "stairs", "hallway", "balcony", "closet", "furniture", "tools", "clothing", "instrument", "glasses",
    "wheel", "drumset", "carpet", "table", "walking", "talking", "looking", "laughing", "dancing",
    "singing", "posing", "car", "bus", "train", "traffic", "crosswalk", "bridge", "technology", "electronic",
    "screen", "camera", "monitor", "equipment", "airplane", "stage", "figure", "illumination",
    "helicopter", "airport", "rotors", "machine", "cage","bearded_man","kitchen","cooking","cooking_show",
    "paws_face", "stage", "suitcase", "wall", "flooring", "clothes", "setup_lighting", "chair",
    "microphone", "bed", "light_fixture", "flight", "photograph", "cityscape", "room", "performer",
    "indoor", "floor", "turntable", "auditorium", "interior", "urban", "metallic", "brick", "curtains",
    "cables", "keyboard", "window", "ceiling", "lamp", "electronics", "pose", "standing", "seated",
    "auditory_performance", "drumming", "speech", "waving", "interior_architecture", "studio_setup",
    "drum_kit", "audience_members", "flash_photography", "performer_clothing", "greeting", "conference",
    "indoor_space", "theatrical", "stage_setup", "technical_equipment", "visual_focus", "stage_lighting",
    "backdrop", "urban_background", "chair_rows", "art_installation", "drums", "drummers", "music",
    "groups", "band", "holiday", "singalong", "karaoke", "lyrics", "stage_lights", "stage_performance",
    "christmas_tree", "songbook", "silhouette", "television_watching", "alien_figure", "runway",
    "aircraft_aluminum", "metal_aircraft", "dj", "blue_lights", "stage", "stadium", "football"
}

people_tags = {
    "group", "portrait", "audience", "hands", "family", "figures", "individuals", "people",
    "musicians", "birthday_cake", "celebration", "dance", "performance", "duo", "crowd",
    "children", "friends", "baby", "interaction", "gathering", "relationship", "intimacy",
    "laughing", "social_gathering", "affection", "companionship", "smile", "hug", "hugging",
    "kiss", "embrace", "joy", "emotion", "play", "human",
    "couple", "together", "social", "togetherness", "companionship", "gesture",
    "waving", "wedding", "portrait", "observation", "warm_lighting", "playful",
    "relaxed", "poise", "expression", "dancing", "event",
    "human_presence", "posing",
    # --- New tags for improved built/people separation ---
    "face", "eyes", "expression", "talking", "crying", "laughing", "screaming", "clapping",
    "cheering", "walking_together", "handshake", "farewell", "welcome", "speaking", "headshot",
    "interview", "interaction_emotional", "body_language", "hugging_family", "looking_into_camera"
}

blur_tags = {
    "chaotic", "dynamic", "blur", "fast", "motion_blur", "shaky", "rapid_motion", "blurry",
    "high-energy", "vibrant", "movement", "dynamic_movement", "chaos", "perceptual_motion",
    "visual_noise", "overexposed", "abstract", "flicker", "motion", "dynamic",
    "abstract_pattern", "geometric_shapes", "time_lapse", "rays", "monochrome"
}

def score_phase(tags, mood, motion, motion_score):
    scores = {
        "orientation": 1,  # baseline
        "elemental": 0,
        "built": 0,
        "people": 0,
        "blur": 0,
    }

    # Tag matches
    scores["elemental"] += len(elemental_tags & tags)
    if any(t in tags for t in ELEMENTAL_DISQUALIFY_IF_PRESENT):
        scores["elemental"] = 0
    if any(t in tags for t in BUILT_DISQUALIFY_IF_PRESENT):
        scores["built"] = 0
    if any(t in tags for t in PEOPLE_DISQUALIFY_IF_PRESENT):
        scores["people"] = 0
    if any(t in tags for t in BLUR_DISQUALIFY_IF_PRESENT):
        scores["blur"] = 0
    scores["built"] += len(built_tags & tags)
    scores["people"] += len(people_tags & tags)
    scores["blur"] += len(blur_tags & tags)

    # --- Refinement: additional rules to distinguish built vs people ---
    refined_people_tags = {"human", "people", "hug", "kiss", "family", "portrait", "smiling", "laughing",
                           "emotion", "face", "eye_contact", "group", "interaction", "gesture", "talking", "touch"}
    refined_built_tags = {"interior", "hallway", "kitchen", "apartment", "bedroom", "architecture", "wall",
                          "ceiling", "door", "window", "urban", "corridor", "enclosed_space", "structure", "bathroom"}
    # Refined tag multipliers (ensure present)
    scores["people"] += len(refined_people_tags & tags)
    scores["built"] += len(refined_built_tags & tags)

    # Co-occurrence boosting
    if "people" in tags and "interaction" in emotional_tags:
        scores["people"] += 2
    if "interior" in tags and "structure" in material_tags:
        scores["built"] += 2

    # Motion/mood influence
    if motion == "moderate" and "human" in tags:
        scores["people"] += 1
    if motion == "still" and "room" in tags:
        scores["built"] += 1

    # Mood and motion modifiers for built, people, blur only
    if mood in {"warm", "joyful", "intimate", "relaxed"}:
        scores["people"] += 1
    if motion_score > 8:
        scores["built"] += 1
    if motion == "dynamic" or motion_score > 20:
        scores["blur"] += 1

    natural_signifiers = {"forest", "moss", "stream", "sunlight", "rocks", "spider_webs", "earth_materials"}
    if len(natural_signifiers & tags) >= 2:
        scores["elemental"] += 2

    soft_nature_tags = {"wind", "grass", "leaves", "branches", "breeze", "sunlight", "natural_color", "flora", "backlit", "morning", "twilight", "rainfall", "mist", "dirt", "clay"}
    if len(soft_nature_tags & tags) >= 2:
        scores["elemental"] += 1

    if any(tag in tags for tag in {"photograph", "screen", "tv", "static", "projection"}) and scores["elemental"] <= 2:
        scores["elemental"] = 0
    # Require at least two raw tag matches for elemental
    if scores["elemental"] < 2:
        scores["elemental"] = 0

    # Blur fallback logic
    if any(tag in {"blur", "motion", "abstract", "monochrome", "static", "nostalgia", "empty", "photograph", "silhouette", "image", "screen", "tv", "projection"} for tag in tags):
        scores["blur"] += 1

    return max(scores, key=scores.get), scores

unmatched_orientation_files = []
all_filenames = []
filename_to_path = {}
filename_to_data = {}

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    filepath = os.path.join(INPUT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        semantic_tags = set(data.get("semantic_tags", []))
        formal_tags = set(data.get("formal_tags", []))
        emotional_tags = set(data.get("emotional_tags", []))
        material_tags = set(data.get("material_tags", []))
        mood_tag = data.get("mood_tag", "")
        motion_tag = data.get("motion_tag", "")
        motion_score = data.get("motion_score", 0)
        all_tags = semantic_tags | formal_tags | emotional_tags | material_tags
        if "people" not in all_tags and "human" in all_tags:
            all_tags.add("people")

        filename = os.path.basename(filepath)
        # --- Insert filename substring disqualify check here ---
        disqualify_in_filename = ["jay", "sweet_home", "birthday", "christmas", "wedding", "dj"]
        if any(sub in filename.lower() for sub in disqualify_in_filename):
            # We will set scores["elemental"]=0 after score_phase, so store this info for later
            filename_disqualify_elemental = True
        else:
            filename_disqualify_elemental = False
        all_filenames.append(filename)
        filename_to_path[filename] = filepath
        filename_to_data[filename] = data

        top_phase, score_dict = score_phase(all_tags, mood_tag, motion_tag, motion_score)
        # Debug print statements for scoring
        print(f"🧮 Score breakdown for {filename}: {score_dict}")
        print(f"🏷️ Assigned top phase: {top_phase}")
        # Disqualify specific video IDs from elemental phase
        video_id = data.get("video_id", "")
        if video_id in ELEMENTAL_DISQUALIFY_IDS:
            score_dict["elemental"] = 0
        # Apply filename substring disqualify logic
        if filename_disqualify_elemental:
            score_dict["elemental"] = 0
        # --- Insert location_tag check here ---
        location_hint = data.get("location_tag", "")
        if location_hint in {"studio", "interior", "performance_space"}:
            score_dict["elemental"] = 0

        # Now, assign phase based on possibly adjusted scores
        top_phase = max(score_dict, key=score_dict.get)
        phase_assignments[top_phase].append(filename)
        # Add suggested_phase and update JSON file
        data["suggested_phase"] = top_phase
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if top_phase == "orientation" and all(v == 0 for k, v in score_dict.items() if k != "orientation"):
            print(f"⚠️ Unmatched but included in orientation: {filename}")
            unmatched_orientation_files.append(filename)
        else:
            print(f"→ Assigned to {top_phase} | Score: {score_dict[top_phase]} | {filename}")

    except Exception as e:
        print(f"Error reading {filename}: {e}")

# --- Reallocation Step for unmatched orientation files ---
import random

# Load tag_history if available for synonyms/expanded tags
tag_synonyms = {}
try:
    with open("tag_history.json", "r", encoding="utf-8") as f:
        tag_history = json.load(f)
        # Assume tag_history is a dict: {tag: [synonyms]}
        for tag, syns in tag_history.items():
            tag_synonyms[tag] = set(syns)
except Exception:
    tag_history = {}

def expand_tags(tags):
    expanded = set(tags)
    queue = list(tags)
    while queue:
        tag = queue.pop()
        if tag in tag_synonyms:
            for syn in tag_synonyms[tag]:
                if syn not in expanded:
                    expanded.add(syn)
                    queue.append(syn)
    return expanded

def relaxed_score_phase(tags, mood, motion, motion_score):
    # Lower threshold: allow a single tag match or strong mood/motion
    scores = {
        "orientation": 1,  # baseline
        "elemental": 0,
        "built": 0,
        "people": 0,
        "blur": 0,
    }
    # Use expanded tags
    expanded = expand_tags(tags)
    scores["elemental"] += len(elemental_tags & expanded)
    if any(t in expanded for t in ELEMENTAL_DISQUALIFY_IF_PRESENT):
        scores["elemental"] = 0  # disqualify if humans or social/built tags are present
    scores["built"] += len(built_tags & expanded)
    scores["people"] += len(people_tags & expanded)
    scores["blur"] += len(blur_tags & expanded)

    # Mood and motion modifiers (relaxed: weight more strongly)
    if mood in {"cool", "neutral"}:
        scores["elemental"] += 2
    if mood in {"warm", "joyful", "intimate", "relaxed"}:
        scores["people"] += 2
    if motion in {"still", "moderate"}:
        scores["elemental"] += 2
    if motion == "dynamic" or motion_score > 20:
        scores["blur"] += 2
    if motion == "moderate" and "people" in expanded:
        scores["people"] += 2
    if motion_score > 5:
        scores["built"] += 1
    # Blur fallback logic
    if any(tag in {"blur", "motion", "abstract", "monochrome", "static", "nostalgia", "empty", "photograph", "silhouette", "image", "screen", "tv", "projection"} for tag in expanded):
        scores["blur"] += 1
    return max(scores, key=scores.get), scores

# Remove unmatched files from orientation for possible reassignment
for fname in unmatched_orientation_files:
    if fname in phase_assignments["orientation"]:
        phase_assignments["orientation"].remove(fname)

reassigned = []
for fname in unmatched_orientation_files:
    data = filename_to_data[fname]
    semantic_tags = set(data.get("semantic_tags", []))
    formal_tags = set(data.get("formal_tags", []))
    emotional_tags = set(data.get("emotional_tags", []))
    material_tags = set(data.get("material_tags", []))
    mood_tag = data.get("mood_tag", "")
    motion_tag = data.get("motion_tag", "")
    motion_score = data.get("motion_score", 0)
    all_tags = semantic_tags | formal_tags | emotional_tags | material_tags
    if "people" not in all_tags and "human" in all_tags:
        all_tags.add("people")
    # Try relaxed scoring
    new_phase, scores = relaxed_score_phase(all_tags, mood_tag, motion_tag, motion_score)
    # Disqualify specific video IDs from elemental in relaxed reassignment
    video_id = data.get("video_id", "")
    if new_phase == "elemental" and video_id in ELEMENTAL_DISQUALIFY_IDS:
        continue
    # Only reassign if not orientation and score > 1
    if new_phase != "orientation" and scores[new_phase] > 0:
        phase_assignments[new_phase].append(fname)
        data["suggested_phase"] = new_phase
        # Save to file
        with open(filename_to_path[fname], "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"♻️ Reassigned {fname} to {new_phase} (relaxed match, score {scores[new_phase]})")
        reassigned.append(fname)
    else:
        # If still not matched, keep in orientation for now
        phase_assignments["orientation"].append(fname)

# Final attempt to sort lingering unmatched orientation files
for fname in phase_assignments["orientation"][:]:
    data = filename_to_data[fname]
    if data.get("suggested_phase") != "orientation":
        continue  # Already updated
    semantic_tags = set(data.get("semantic_tags", []))
    formal_tags = set(data.get("formal_tags", []))
    emotional_tags = set(data.get("emotional_tags", []))
    material_tags = set(data.get("material_tags", []))
    mood_tag = data.get("mood_tag", "")
    motion_tag = data.get("motion_tag", "")
    motion_score = data.get("motion_score", 0)
    all_tags = semantic_tags | formal_tags | emotional_tags | material_tags
    if "people" not in all_tags and "human" in all_tags:
        all_tags.add("people")
    expanded_tags = expand_tags(all_tags)
    
    if any(t in expanded_tags for t in ELEMENTAL_DISQUALIFY_IF_PRESENT):
         elemental_score = 0
    else:
        elemental_score = len(elemental_tags & expanded_tags)

    natural_signifiers = {"forest", "moss", "stream", "sunlight", "rocks", "spider_webs", "earth_materials"}
    if len(natural_signifiers & expanded_tags) >= 2:
        elemental_score += 2

    soft_nature_tags = {"wind", "grass", "leaves", "branches", "breeze", "sunlight", "natural_color", "flora", "backlit", "morning", "twilight", "rainfall", "mist", "dirt", "clay"}
    if len(soft_nature_tags & expanded_tags) >= 2:
        elemental_score += 1

    if any(tag in expanded_tags for tag in {"photograph", "screen", "tv", "static", "projection"}) and elemental_score <= 2:
        elemental_score = 0

    scores = {
        "elemental": elemental_score,
        "built": len(built_tags & expanded_tags),
        "people": len(people_tags & expanded_tags),
        "blur": len(blur_tags & expanded_tags)
    }
    # Fallback using mood/motion as soft influence
    if mood_tag in {"cool", "neutral"}:
        scores["elemental"] += 1
    if mood_tag in {"warm", "joyful", "intimate", "relaxed"}:
        scores["people"] += 1
    if motion_tag == "dynamic" or motion_score > 20:
        scores["blur"] += 1
    if motion_score > 5:
        scores["built"] += 1

    # People fallback improvement
    if any(tag in expanded_tags for tag in {"group", "party", "celebration", "audience", "gathering", "crowd"}) and scores["people"] >= scores["elemental"]:
        scores["people"] += 2

    best_phase = max(scores, key=scores.get)
    if best_phase:
        if best_phase == "elemental":
            continue  # Never forcibly assign lingering files to elemental
        phase_assignments["orientation"].remove(fname)
        phase_assignments[best_phase].append(fname)
        data["suggested_phase"] = best_phase
        with open(filename_to_path[fname], "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"📦 Forcibly reassigned lingering file to {best_phase}: {fname}")

# --- Rebuild orientation as a random sample of all processed files ---
# Exclude files already strongly assigned to another movement
strongly_assigned = set()
for k in phase_assignments:
    if k != "orientation":
        strongly_assigned.update(phase_assignments[k])
eligible = [fname for fname in all_filenames if fname not in strongly_assigned]
# Choose 5-10% of all processed files (at least 1)
sample_size = max(1, int(0.07 * len(all_filenames)))
random.seed(42)
orientation_sample = random.sample(eligible, min(sample_size, len(eligible)))
phase_assignments["orientation"] = list(orientation_sample)
for fname in orientation_sample:
    data = filename_to_data[fname]
    data["suggested_phase"] = "orientation"
    with open(filename_to_path[fname], "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# Save assignment
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(phase_assignments, f, indent=2)

print(f"✅ Saved movement archive to {OUTPUT_FILE}")