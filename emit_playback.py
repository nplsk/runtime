import argparse
import csv, random, yaml, uuid
from pathlib import Path
import pandas as pd
import math

# --- Debug mode toggle ---
DEBUG_MODE = True  # Set to False to suppress debug logging

# --- helpers for semantic/color reordering ---
from itertools import permutations

def jaccard(tags1, tags2):
    s1, s2 = set(tags1), set(tags2)
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0

def reorder_group(entries, strategy, tags_key):
    """
    Greedy reorder by Jaccard: high_similarity keeps similar clips together,
    low_similarity spaces them apart.
    """
    if not entries: return entries
    entries = entries.copy()
    ordered = [entries.pop(0)]
    while entries:
        current = ordered[-1]
        if strategy == 'high_similarity':
            best = max(entries, key=lambda e: jaccard(current[tags_key], e[tags_key]))
        else:
            best = min(entries, key=lambda e: jaccard(current[tags_key], e[tags_key]))
        entries.remove(best)
        ordered.append(best)
    return ordered

def reorder_by_color(entries):
    """
    Greedy reorder by Euclidean distance on first dominant color.
    """
    def parse(col): return tuple(int(col.lstrip('#')[i:i+2], 16) for i in (0,2,4))
    def dist(a, b): return sum((a[i]-b[i])**2 for i in range(3))**0.5
    if not entries: return entries
    entries = entries.copy()
    ordered = [entries.pop(0)]
    while entries:
        cur_col = parse(ordered[-1]['dominant_colors'][0])
        best = min(entries, key=lambda e: dist(cur_col, parse(e['dominant_colors'][0])))
        entries.remove(best)
        ordered.append(best)
    return ordered

def ensure_spacing(schedule, min_sep=3):
    """
    Enforce a minimum separation between clips from the same source file.
    Uses the file_path prefix before '_scene' to identify siblings.
    """
    positions = {}
    for i, entry in enumerate(schedule):
        prefix = entry['path'].rsplit('_scene', 1)[0]
        if prefix in positions and i - positions[prefix] < min_sep:
            # swap with the next later entry that has a different prefix
            for j in range(i + 1, len(schedule)):
                other_prefix = schedule[j]['path'].rsplit('_scene', 1)[0]
                if other_prefix != prefix:
                    schedule[i], schedule[j] = schedule[j], schedule[i]
                    break
        positions[prefix] = i
    return schedule

def load_movelist(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def pick_clips(movelist, n, used):
    choices = [c for c in movelist if c['video_id'] not in used]
    # allow reuse only if looping permitted later
    return random.sample(choices, min(n, len(choices)))

def build_schedule(cfg, all_cfgs):
    # capture movement name for special-handling
    cfg_name = cfg['name']
    # compute time offset for split movements (part2)
    offset = 0
    if cfg_name.endswith('_part2'):
        base = cfg_name[:-6]  # remove '_part2'
        offset = all_cfgs[f"{base}_part1"]['total_duration']
    explanation_log = []
    if cfg_name == 'orientation':
        # custom orientation: burst & sustain cycles until total_duration
        movelist = []
        for m in ['elemental','built','people_part1','people_part2','blur_part1','blur_part2']:
            movelist.extend(load_movelist(all_cfgs[m]['csv']))
        schedule = []
        time = 0.0
        burst_count = 20
        burst_dur   = 0.5
        sustain_count = 3
        sustain_dur   = 8.0
        total = cfg['total_duration']
        # alternate burst of quick singles and sustain blocks
        while time < total:
            # (1) burst phase
            for _ in range(burst_count):
                clip = random.choice(movelist)
                schedule.append({
                    'movement': cfg_name,
                    'time_start': round(time + offset, 2),
                    'video_id': clip['video_id'],
                    'path': clip['file_path'],
                    'layer': 0,
                    'duration': burst_dur,
                    'loop': False,
                    'uuid': str(uuid.uuid4()),
                    'semantic_tags': [t.strip() for t in clip['semantic_tags'].split(',')],
                    'mood_tag': clip['mood_tag'],
                    'motion_tag': clip['motion_tag'],
                    'motion_variance': float(clip['motion_variance']),
                    'dominant_colors': [c.strip() for c in clip['dominant_colors'].split(',')],
                    'emotional_tags': [t.strip() for t in clip['emotional_tags'].split(',')],
                    'formal_tags': [t.strip() for t in clip['formal_tags'].split(',')],
                    'material_tags': [t.strip() for t in clip['material_tags'].split(',')],
                    'ai_description': clip.get('ai_description', ''),
                })
                # For orientation, explanation_log is not detailed (no scoring), but still log basic info
                explanation_log.append({
                    "movement": cfg_name,
                    "time_start": round(time + offset, 2),
                    "video_id": clip['video_id'],
                    "path": clip['file_path'],
                    "score_breakdown": {
                        "decay_weight": None,
                        "semantic_match": None,
                        "motion_variance": float(clip['motion_variance']),
                        "final_weight": None
                    }
                })
                time += burst_dur
                if time >= total:
                    break
            if time >= total:
                break
            # (2) sustain phase
            clips = random.sample(movelist, min(sustain_count, len(movelist)))
            for idx, clip in enumerate(clips):
                schedule.append({
                    'movement': cfg_name,
                    'time_start': round(time + offset, 2),
                    'video_id': clip['video_id'],
                    'path': clip['file_path'],
                    'layer': idx,
                    'duration': sustain_dur,
                    'loop': False,
                    'uuid': str(uuid.uuid4()),
                    'semantic_tags': [t.strip() for t in clip['semantic_tags'].split(',')],
                    'mood_tag': clip['mood_tag'],
                    'motion_tag': clip['motion_tag'],
                    'motion_variance': float(clip['motion_variance']),
                    'dominant_colors': [c.strip() for c in clip['dominant_colors'].split(',')],
                    'emotional_tags': [t.strip() for t in clip['emotional_tags'].split(',')],
                    'formal_tags': [t.strip() for t in clip['formal_tags'].split(',')],
                    'material_tags': [t.strip() for t in clip['material_tags'].split(',')],
                    'ai_description': clip.get('ai_description', ''),
                })
                explanation_log.append({
                    "movement": cfg_name,
                    "time_start": round(time + offset, 2),
                    "video_id": clip['video_id'],
                    "path": clip['file_path'],
                    "score_breakdown": {
                        "decay_weight": None,
                        "semantic_match": None,
                        "motion_variance": float(clip['motion_variance']),
                        "final_weight": None
                    }
                })
            time += sustain_dur
        return schedule, explanation_log
    else:
        movelist = load_movelist(cfg['csv'])

    # --- Per-phase filtering logic ---
    phase_tag_filters = {
        "elemental": {
            "semantic_tags": {
                "nature", "forest", "water", "serene", "calm", "trees", "stream", "sky", "rocks", "sunlight",
                "fog", "grass", "leaves", "branches", "flora", "wind", "dirt", "clay", "moss", "insect",
                "volcano", "ice", "river", "lake", "sun", "earth", "animal", "clouds", "natural_light",
                "diffused_light", "natural_color"
            },
        },
        "built": {
            "semantic_tags": {"interior", "architecture", "room", "urban", "furniture", "structure"},
        },
        "people": {
            "semantic_tags": {"people", "human", "relationship", "interaction", "hug", "kiss"},
            "emotional_tags": {"people", "human", "relationship", "interaction", "hug", "kiss"},
        },
        "blur": {
            "semantic_tags": {"blur", "abstract", "motion_blur"},
            "motion_variance": 15,  # threshold
        }
    }
    # Determine which phase logic to use
    phase_base = cfg_name.split("_part")[0] if "_part" in cfg_name else cfg_name
    filtered_movelist = movelist
    if phase_base in phase_tag_filters:
        before = len(filtered_movelist)
        if phase_base == "elemental":
            # Only keep clips with required tags in semantic_tags
            filtered_movelist = [
                c for c in filtered_movelist
                if any(tag in [t.strip() for t in c['semantic_tags'].split(',')] for tag in phase_tag_filters["elemental"]["semantic_tags"])
            ]
            if DEBUG_MODE:
                print(f"→ Phase '{phase_base}' post-filter movelist size: {len(filtered_movelist)}")
        elif phase_base == "built":
            # Loosen filtering: allow fallback for mood_tag in {"neutral", "dim"}
            filtered_movelist = [
                c for c in filtered_movelist
                if (
                    any(tag in [t.strip() for t in c['semantic_tags'].split(',')] for tag in phase_tag_filters["built"]["semantic_tags"])
                    or (str(c.get("mood_tag", "")).strip() in {"neutral", "dim"})
                )
            ]
            if DEBUG_MODE:
                print(f"→ Phase '{phase_base}' post-filter movelist size: {len(filtered_movelist)}")
        elif phase_base == "people":
            # Loosen filtering: allow fallback for motion_tag "moderate" or mood_tag in {"warm", "joyful"}
            filtered_movelist = [
                c for c in filtered_movelist
                if (
                    any(tag in [t.strip() for t in c['semantic_tags'].split(',')] for tag in phase_tag_filters["people"]["semantic_tags"])
                    or
                    any(tag in [t.strip() for t in c['emotional_tags'].split(',')] for tag in phase_tag_filters["people"]["emotional_tags"])
                    or (str(c.get("motion_tag", "")).strip() == "moderate")
                    or (str(c.get("mood_tag", "")).strip() in {"warm", "joyful"})
                )
            ]
            if DEBUG_MODE:
                print(f"→ Phase '{phase_base}' post-filter movelist size: {len(filtered_movelist)}")
        elif phase_base == "blur":
            filtered_movelist = [
                c for c in filtered_movelist
                if (
                    float(c.get('motion_variance', 0)) > phase_tag_filters["blur"]["motion_variance"]
                    or
                    any(tag in [t.strip() for t in c['semantic_tags'].split(',')] for tag in phase_tag_filters["blur"]["semantic_tags"])
                )
            ]
            if DEBUG_MODE:
                print(f"→ Phase '{phase_base}' post-filter movelist size: {len(filtered_movelist)}")
        after = len(filtered_movelist)
        if DEBUG_MODE:
            print(f"🔍 Phase '{cfg_name}' filtered movelist: {before} → {after}")
    # Add debug print after tag filtering for phase logic
    print(f"🧪 Phase '{cfg_name}': Filtered movelist: {len(movelist)} → Candidates: {len(filtered_movelist)}")
    movelist = filtered_movelist
    time = 0
    schedule = []
    used = set()
    last_played_time = {}
    # Add per-layer state tracking
    layer_times = [0.0] * cfg['simultaneous']
    layer_durations = [0.0] * cfg['simultaneous']
    master_time = 0.0
    # --- Add per-root state tracking for max scenes per root ---
    from collections import defaultdict
    root_counts = defaultdict(int)
    MAX_SCENES_PER_ROOT = 5

    # Allow per-movement min_sep (default 3, but can be set in cfg)
    min_sep = cfg.get('min_sep', 3)

    # --- Root usage penalty setup ---
    recent_root_usage = defaultdict(int)
    root_decay_weight = 0.75

    # Staggered per-layer scheduling loop
    while master_time < cfg['total_duration']:
        for layer in range(cfg['simultaneous']):
            if master_time >= layer_times[layer] + layer_durations[layer]:
                # Pick eligible clips
                clips = []
                for c in movelist:
                    clip_id = c['video_id']
                    if clip_id in used and not cfg['allow_loop']:
                        continue
                    time_since_last = master_time - last_played_time.get(clip_id, -math.inf)
                    # *** Increase temporal decay penalty ***
                    decay = 1 / (1 + time_since_last) if time_since_last > 0 else 1.0
                    base_weight = 1.0
                    # --- Root penalty logic ---
                    root_id = c['video_id'].split('_scene_')[0]
                    root_penalty = 1.0 / (1.0 + recent_root_usage[root_id])  # reduces weight the more it's used
                    # Prevent weight from being exactly zero: fallback to min 0.001
                    weight = max(0.001, base_weight * decay * root_penalty if cfg.get('use_temporal_decay') else base_weight * root_penalty)
                    c['_weight'] = weight
                sorted_choices = sorted([c for c in movelist if '_weight' in c], key=lambda x: -x['_weight'])

                # Debug print for decay/weight logic
                if DEBUG_MODE:
                    for c in sorted_choices[:10]:
                        root_id_dbg = c['video_id'].split('_scene_')[0]
                        print(f"🎯 Candidate: {c['video_id']} | Weight: {c['_weight']:.3f} | Last played: {last_played_time.get(c['video_id'], 'never')} | Root usage: {recent_root_usage[root_id_dbg]}")

                # Apply phase tag filters
                phase_choices = sorted_choices
                if phase_base in phase_tag_filters:
                    if phase_base == "people":
                        # Loosen filtering: allow fallback for motion_tag "moderate" or mood_tag in {"warm", "joyful"}
                        phase_choices = [c for c in phase_choices if (
                            any(tag in [t.strip() for t in c['semantic_tags'].split(',')] for tag in phase_tag_filters["people"]["semantic_tags"])
                            or
                            any(tag in [t.strip() for t in c['emotional_tags'].split(',')] for tag in phase_tag_filters["people"]["emotional_tags"])
                            or (str(c.get("motion_tag", "")).strip() == "moderate")
                            or (str(c.get("mood_tag", "")).strip() in {"warm", "joyful"})
                        )]
                    elif phase_base == "built":
                        # Loosen filtering: allow fallback for mood_tag in {"neutral", "dim"}
                        phase_choices = [c for c in phase_choices if (
                            any(tag in [t.strip() for t in c['semantic_tags'].split(',')] for tag in phase_tag_filters["built"]["semantic_tags"])
                            or (str(c.get("mood_tag", "")).strip() in {"neutral", "dim"})
                        )]
                    elif phase_base == "blur":
                        phase_choices = [c for c in phase_choices if (
                            float(c.get('motion_variance', 0)) > phase_tag_filters["blur"]["motion_variance"]
                            or
                            any(tag in [t.strip() for t in c['semantic_tags'].split(',')] for tag in phase_tag_filters["blur"]["semantic_tags"])
                        )]

                # --- Filter phase_choices by root count (max scenes per root) ---
                filtered_phase_choices = []
                for c in phase_choices:
                    root = c['video_id'].split('_scene_')[0]
                    if root_counts[root] < MAX_SCENES_PER_ROOT:
                        filtered_phase_choices.append(c)
                phase_choices = filtered_phase_choices

                # --- Prevent the same video_id from being scheduled at the same time across different layers ---
                used_this_time = {
                    e['video_id']
                    for e in schedule
                    if abs(e['time_start'] - (master_time + offset)) < 100.0  # Prevent reuse within 100 seconds
                }
                phase_choices = [c for c in phase_choices if c['video_id'] not in used_this_time]

                if not phase_choices:
                    continue

                # Weighted random selection
                weights = [c['_weight'] for c in phase_choices]
                if all(w <= 0 for w in weights):
                    if DEBUG_MODE:
                        print(f"⚠️ Skipping layer {layer} at time {master_time:.2f} due to all-zero weights")
                    continue
                clip = random.choices(phase_choices, weights=weights, k=1)[0]
                duration = random.uniform(*cfg['duration_range'])
                entry = {
                    'movement': cfg_name,
                    'time_start': round(master_time + offset, 2),
                    'video_id': clip['video_id'],
                    'path': clip['file_path'],
                    'layer': layer,
                    'duration': round(duration, 2),
                    'loop': cfg['allow_loop'],
                    'uuid': str(uuid.uuid4()),
                    'semantic_tags': [t.strip() for t in clip['semantic_tags'].split(',')],
                    'mood_tag': clip['mood_tag'],
                    'motion_tag': clip['motion_tag'],
                    'motion_variance': float(clip['motion_variance']),
                    'dominant_colors': [c.strip() for c in clip['dominant_colors'].split(',')],
                    'emotional_tags': [t.strip() for t in clip['emotional_tags'].split(',')],
                    'formal_tags': [t.strip() for t in clip['formal_tags'].split(',')],
                    'material_tags': [t.strip() for t in clip['material_tags'].split(',')],
                    'ai_description': clip.get('ai_description', ''),
                }
                # Prevent same clip playing on multiple layers at the same time
                overlapping = any(
                    e['video_id'] == clip['video_id'] and abs(e['time_start'] - (master_time + offset)) < 0.1
                    for e in schedule
                )
                if overlapping:
                    continue
                schedule.append(entry)
                used.add(clip['video_id'])
                last_played_time[clip['video_id']] = master_time
                layer_times[layer] = master_time
                layer_durations[layer] = duration
                # --- Increment recent_root_usage for selected root_id ---
                selected_root_id = clip['video_id'].split('_scene_')[0]
                recent_root_usage[selected_root_id] += 1
                # --- Increment root_counts for selected root ---
                root = clip['video_id'].split('_scene_')[0]
                root_counts[root] += 1
                # --- Optional debug print for root selection ---
                if DEBUG_MODE:
                    print(f"📦 Selected clip {clip['video_id']} from root {root} (used {root_counts[root]} times)")

                explanation_log.append({
                    "movement": cfg_name,
                    "time_start": round(master_time + offset, 2),
                    "video_id": clip['video_id'],
                    "path": clip['file_path'],
                    "score_breakdown": {
                        "decay_weight": clip['_weight'] if cfg.get('use_temporal_decay') else 1.0,
                        "semantic_match": True,
                        "motion_variance": float(clip['motion_variance']),
                        "final_weight": clip['_weight']
                    }
                })
        master_time += 0.1
        # --- Periodically decay recent_root_usage to allow older clips to re-enter ---
        if int(master_time * 10) % 50 == 0:
            for k in recent_root_usage:
                recent_root_usage[k] *= root_decay_weight

    # Post-process: ensure scenes from the same file are at least min_sep steps apart.
    schedule = ensure_spacing(schedule, min_sep=8)

    # --- Candidate balancing and debugging: print summary before returning ---
    if DEBUG_MODE:
        from collections import Counter
        counts = Counter([e['video_id'] for e in schedule])
        print("📊 Top used clips:", counts.most_common(10))

    # Semantic transitions per movement
    strat = cfg.get('transition_strategy')
    if strat:
        schedule = reorder_group(schedule, strat, 'semantic_tags')

    # Color harmony sequencing
    if cfg.get('use_color_harmony'):
        schedule = reorder_by_color(schedule)

    # Temporal decay: penalize recently used clips
    if cfg.get('use_temporal_decay'):
        for entry in schedule:
            clip_id = entry['video_id']
            last_played_time[clip_id] = entry['time_start']

    # --- Debug printout after final schedule ---
    if DEBUG_MODE:
        print(f"✅ Final schedule for {cfg_name}: {len(schedule)} entries")

    return schedule, explanation_log

if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description="Emit playback schedule")
    parser.add_argument("--movement", type=str, help="Optional: specify a single movement to regenerate")
    args = parser.parse_args()

    cfgs = yaml.safe_load(Path('movements.yml').read_text())
    all_rows = []
    all_explanation_logs = []
    for cfg_name, cfg in cfgs.items():
        if args.movement and cfg_name != args.movement:
            continue
        rows, explanation_log = build_schedule({**cfg, 'name':cfg_name}, cfgs)
        all_rows.extend(rows)
        all_explanation_logs.extend(explanation_log)

    df = pd.DataFrame(all_rows)
    movement_order = ["orientation","elemental","built","people_part1","people_part2","blur_part1","blur_part2"]
    df['movement'] = pd.Categorical(df['movement'], categories=movement_order, ordered=True)
    df.sort_values(['movement', 'time_start'], inplace=True)

    # Load existing schedule if partial regeneration
    if args.movement:
        existing_df = pd.read_csv('performance_schedule.csv')
        existing_df = existing_df[existing_df['movement'] != args.movement]
        df = pd.concat([existing_df, df], ignore_index=True)
        df['movement'] = pd.Categorical(df['movement'], categories=movement_order, ordered=True)
        df.sort_values(['movement', 'time_start'], inplace=True)

    df.to_csv('performance_schedule.csv', index=False)
    print("Wrote performance_schedule.csv with", len(df), "rows")

    # Write explanation log JSON
    with open("clip_selection_explanations.json", "w", encoding="utf-8") as f:
        json.dump(all_explanation_logs, f, indent=2)