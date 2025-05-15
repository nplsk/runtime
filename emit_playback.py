import csv, random, yaml, uuid
from pathlib import Path
import pandas as pd

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
            time += sustain_dur
        return schedule
    else:
        movelist = load_movelist(cfg['csv'])
    time = 0
    schedule = []
    used = set()

    while time < cfg['total_duration']:
        clips = pick_clips(movelist, cfg['simultaneous'], used)
        durations = [random.uniform(*cfg['duration_range']) for _ in clips]
        for clip, dur in zip(clips, durations):
            entry = {
                'movement': cfg_name,
                'time_start': round(time + offset, 2),
                'video_id': clip['video_id'],
                'path': clip['file_path'],
                'layer': clips.index(clip),           # which screen 0..N-1
                'duration': round(dur,2),
                'loop': cfg['allow_loop'],
                'uuid': str(uuid.uuid4()),            # unique row key
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
            schedule.append(entry)
            used.add(clip['video_id'])
        # advance time by the *shortest* clip so overlaps cascade
        time += min(durations)
        # if reused allowed, clear used set when exhausted
        if cfg['allow_loop'] and len(used) >= len(movelist):
            used.clear()

    # Post-process: ensure scenes from the same file are at least 3 steps apart
    schedule = ensure_spacing(schedule, min_sep=3)

    # Semantic transitions per movement
    strat = cfg.get('transition_strategy')
    if strat:
        schedule = reorder_group(schedule, strat, 'semantic_tags')

    # Color harmony sequencing
    if cfg.get('use_color_harmony'):
        schedule = reorder_by_color(schedule)

    # Temporal decay placeholder (e.g., penalize repeats in future runs)
    if cfg.get('use_temporal_decay'):
        # TODO: implement temporal decay weighting
        pass

    return schedule

if __name__ == '__main__':
    cfgs = yaml.safe_load(Path('movements.yml').read_text())
    all_rows = []
    for cfg_name, cfg in cfgs.items():
        rows = build_schedule({**cfg, 'name':cfg_name}, cfgs)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    movement_order = ["orientation","elemental","built","people_part1","people_part2","blur_part1","blur_part2"]
    df['movement'] = pd.Categorical(df['movement'], categories=movement_order, ordered=True)
    df.sort_values(['movement', 'time_start'], inplace=True)
    df.to_csv('performance_schedule.csv', index=False)
    print("Wrote performance_schedule.csv with", len(df), "rows")