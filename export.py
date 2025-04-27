import os
import json

output_dir = "./output"
export_path = os.path.join(output_dir, "descriptions_review.txt")

with open(export_path, "w") as export_file:
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            export_file.write(f"File: {filename}\n")
            export_file.write(f"Semantic Caption: {data.get('semantic_caption', '')}\n")
            export_file.write(f"Main Description: {data.get('ai_description', '')}\n\n")

            styles = data.get("style_outputs", {})
            for style, content in styles.items():
                export_file.write(f"  [{style.upper()}]\n")
                export_file.write(f"  {content.get('ai_description', '')}\n\n")

            export_file.write("=" * 80 + "\n\n")

print(f"Exported all descriptions to {export_path}")