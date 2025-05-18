"""
This script is a TouchDesigner CHOP Execute DAT callback that manages the playback of video clips based on a timeline and movement phases.
It is triggered whenever the timeline CHOP value changes and performs the following tasks:

1. Determines the current movement phase (e.g., orientation, elemental, built, people, blur).
2. Detects when the timeline wraps (resets from max back to 0) and advances the movement phase.
3. For each active layer, it searches the schedule DAT for the next clip to play based on the current movement and layer.
4. When the global time reaches the scheduled start time of a clip, it sets the file on the corresponding Movie TOP and advances the row index.

Input:
- A timeline CHOP that cycles from 0 to 300 seconds.
- A Constant CHOP ('project') that holds row indices and the current movement state.
- A schedule DAT ('scheduleDAT') that contains the clip schedule with columns for movement, start time, and file path.

Output:
- Updates the file parameter of Movie TOPs to play the scheduled clips.
- Advances the movement phase and row indices as the timeline progresses.
- Logs debugging information to the console.

Note: This script is designed to be used within TouchDesigner and relies on specific operator names and structures.
"""

# me - this DAT
# 
# channel - the Channel object which has changed
# sampleIndex - the index of the changed sample
# val - the numeric value of the changed sample
# prev - the previous sample value
# 
# Make sure the corresponding toggle is enabled in the CHOP Execute DAT.
sch = op('scheduleDAT')

def onOffToOn(channel, sampleIndex, val, prev):
	return

def whileOn(channel, sampleIndex, val, prev):
	#op('readout').bypass = 0
	return

def onOnToOff(channel, sampleIndex, val, prev):
	return

def whileOff(channel, sampleIndex, val, prev):
	#op('readout').bypass = 1
	return

def onValueChange(channel, sampleIndex, val, prev):
    # ENTRY POINT: callback triggered by timeline CHOP whenever its value changes
    print(f"onValueChange called: channel={channel.name}, val={val}, prev={prev}")
    # Called whenever the timeline CHOP value changes.
    # channel: CHOP channel triggering this callback
    # val: current global time in seconds (cycles 0–300)
    # prev: previous time value
    # Reference the Constant CHOP holding row indices and movement state
    project = op('project')  # Constant CHOP
    # DEBUG: print current rowIndex and movement channel values
    # dump key Constant CHOP channels
    for ch_name in ['rowIndex0','rowIndex1','rowIndex2','rowIndex3','rowIndex4','rowIndex5','movement']:
        ch = project[ch_name]
        print(f"  CHOP channel {ch_name}: {ch}")
    sch     = op('scheduleDAT')
    
    # Determine current movement phase from 'movement' channel (0-based index)
    # Determine which movement phase we're in (0=orientation, 1=elemental, etc.)
    try:
        movement = int(project['movement'])        
    except:
        movement = 0
    # DEBUG: show movement index and its corresponding name
    print(f"  Parsed movement (int): {movement}")

    # Log current movement state for debugging
    movement_names = [
        'orientation',
        'elemental',
        'built',
        'people',        
        'blur'
    ]
    current_move = movement_names[movement] if 0 <= movement < len(movement_names) else None

    print(f"Movement idx: {movement}, Movement name: {current_move}")
    # DETECT WRAP: timeline CHOP value reset from max back to 0
    print("  Checking for timer wrap...")
    # Detect timer wrap (300→0) and reset row indices & advance movement
    if val < prev:
        print(f"  Wrap detected! prev={prev}, val={val}")
        # Reset all per-layer rowIndex channels to start fresh for new movement cycle
        # Reset per-layer rowIndex channels to 0
        for L in (0, 1, 2, 3, 4, 5):
            ch = project[f'rowIndex{L}']
            if ch is not None:
                # write back to the corresponding const parameter
                project.par[f'const{L}value'] = 0

        # Increment movement channel to move to next performance segment
        mv_chan = int(project['movement'])
        print(f"Movement channel: {mv_chan}")
        if mv_chan is not None:
            cur_mv = int(mv_chan)
            print(f"Current movement index: {cur_mv}")
            new_mv = cur_mv + 1
            if new_mv >= len(movement_names):
                # End of performance: cue credits slide and stop playback
                for L in (0, 1, 2, 3, 4, 5):
                    op(f'movie{L}').par.file = ''					
                    op('timeline1').par.usetimecode = 0
                    op('timeline1').par.second = 0
                print("End of performance: credits slide cue")
                return
            else:
                project.par.const6value = new_mv
                print(f"  Movement bumped to {new_mv}")
                print(f"New movement index: {new_mv}")

    # If movement index is out of known names, skip scheduling but allow wrap logic
    if current_move is None:
        print("No valid movement name for current index; skipping scheduling")
        return

    # GATE: only run layer 0 during the orientation segment
    # During orientation (movement 0) only process layer 0, else process all layers
    if movement == 0:
        # only layer 0 during orientation
        layers_to_check = (0,)
    else:
        layers_to_check = (0, 1, 2, 3, 4, 5)
    print(f"  Layers to check for movement {movement}: {layers_to_check if 'layers_to_check' in locals() else 'TBD'}")

    # Iterate over each active layer to schedule its next clip
    # Iterate through each active layer and find its next clip
    for layer in layers_to_check:
        # print(f"  Layer {layer}: starting idx lookup")
        # Get current rowIndex for this layer from the Constant CHOP
        try:
            p = project[f'rowIndex{layer}']  # CHOP channel
        except:
            p = None
        idx = int(p) if p else 0
        # print(f"Param rowIndex{layer} exists: {p is not None}, p object: {p}, starting idx: {idx}")
        # print(f"Checking layer {layer}, starting idx {idx}, current time {val}")
        # print(f"  Searching schedule DAT for layer {layer}, movement {current_move}")
        # SEARCH SCHEDULE: find the next row matching current movement and layer
        # Search forward in the schedule for the next row matching this movement & layer
        while idx < sch.numRows:
            try:
                raw_move = sch[idx, 0].val
                # skip rows without a valid move string
                if raw_move is None:
                    idx += 1
                    continue
                row_move  = str(raw_move)
                row_layer = int(sch[idx, 4].val)
                row_t     = float(sch[idx, 1].val)
                # print(f"Row {idx}: move={row_move}, layer={row_layer}, time_start={row_t}")
            except:
                idx += 1
                continue

            # only match rows whose move starts with the current movement name
            if not row_move.startswith(current_move) or row_layer != layer:
                idx += 1
                continue
            break

        if idx < sch.numRows:
            t = row_t
            # TIME CHECK: launch clip when global time reaches row's start time
            # If it's time to launch this clip, set the file on the Movie TOP
            if val >= t:
                path = sch[idx, 3]
                print(f"[✔] PLAYING: Movement='{current_move}' | Layer={layer} | Row={idx} | Time={val:.2f}s\n      ↳ Clip: {path}")
                # print(f"Launching clip on layer {layer}: {path} at {val}s (scheduled {t}s)")
                # print(f"    Advancing rowIndex{layer} to {idx+1}")
                # set file on the proper Movie File In TOP and reload it
                m = op(f'movie{layer}')
                if m:
                    m.par.file = path
                    # force the TOP to reload the new file
                    try:
                        m.par.reload.pulse()
                    except:
                        pass
                # ADVANCE POINTER: update rowIndex channel so this row isn't reused
                # Advance this layer's rowIndex so we don't replay the same row
                project.par[f'const{layer}value'] = idx + 1
    return