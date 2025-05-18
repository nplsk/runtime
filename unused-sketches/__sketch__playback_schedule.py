# 3. How to Use in TouchDesigner
# 	1.	Import performance_schedule.csv into a Table DAT.
# 	2.	In your Movie File In TOP network, have parameters driven by columns:
# 	    •	file → me.inputCell.row('path')
# 	    •	Timer CHOP for each row’s duration.
# 	3.	Use a DAT Execute on row‐change to:
#	4.	Advance RowIndex whenever any Timer CHOP hits “done.”

# Why This Helps
# 	•	Single source of truth: one CSV, one Table DAT, one slice of Python inside TD.
# 	•	Per‐movement tuning: simply adjust movements.yml—no code edits.
# 	•	Textual tags drive clip selection only; mood/motion can later feed into offsets or color filters in TD, not clip choice.
# 	•	Layer column tells TD which MovieIn to drive, so you can arbitrarily map “screen A/B/C” per movement.

# With this in place you can instantly re-generate a new preview by rerunning the Python script—no deeper refactor, and you get a neatly tabulated schedule that TD can step through.


def onTableChange(dat):
    row = dat.rows[me.par.RowIndex]
    clip = op('moviein'+row.layer)
    clip.par.file = row.path
    run('me.par.TimerStart=1')  # or trigger Timer CHOP


# 1.	Load the schedule into a Table DAT
# 	•	Point a Table DAT at your performance_schedule.csv. You should see columns like semantic_caption, ai_description, dominant_colors, etc., alongside path, duration, and so on.
# 	2.	Use a DAT Execute or CHOP to pick the current row
# 	•	Drive a RowIndex parameter (on a Null CHOP, an Operator CHOP, or a custom parameter) from your playback timer.
# 	•	Hook a DAT Execute DAT on your Table:

def onTableChange(dat):
    r = int(me.parent().par.Rowindex)  # whatever you call your row pointer
    row = dat.rows[r]
    op('textDAT').text = f"""
    Path: {row['path'].val}
    Duration: {row['duration'].val:.2f}s
    Tags: {row['semantic_tags'].val}
    Mood: {row['mood_tag'].val}, Motion: {row['motion_tag'].val}
    Caption: {row['semantic_caption'].val}
    AI Notes: {row['ai_description'].val}
    """
    return

	# •	That script writes all of your “behind‑the‑scenes” fields into a Text DAT called textDAT.

	# 3.	Render the Text DAT on‑screen
	# •	Drop down a Text TOP and point its DAT parameter at textDAT.
	# •	Composite that Text TOP over your three Movie File In TOPs with a Composite TOP (or layer it in your GLSL).
	# 4.	Style & Layout
	# •	You can drive font size, alignment, or even color‑code bits of text by splitting your string into separate DATs (e.g. one for tags, one for captions) and mapping them to multiple Text TOPs.
	# •	If you want fancier layouts—columns, scrolling lists, dynamic opacity—use a Panel COMP or a Container COMP to hold your Text TOP and animate its position or alpha.

# Why this works
# 	•	Single source of truth: all your metadata lives in performance_schedule.csv. No need to re‑parse per‑movement CSVs.
# 	•	Flexibility: you can choose exactly which columns to surface as text. Want to show dominant colors as swatches instead? Use a TOP to draw small swatches based on the dominant_colors hex values.
# 	•	Simplicity: all your logic lives in the DAT Execute–Text DAT–Text TOP chain; your Movie File In network is untouched.