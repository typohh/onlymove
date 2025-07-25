import os
import re
import shutil
import json


def extract_difficulty(content):
    """Extract difficulty rank from SGF content."""
    patterns = [
        r"GN\[Move difficulty: (\d+[kd])±",
        r"GN\[Last move difficulty: (\d+[kd])±"
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    return None


if __name__ == "__main__":
    tsumegos_dir = "pwa/tsumegos"
    if not os.path.exists(tsumegos_dir):
        print(f"Error: {tsumegos_dir} directory not found")
        exit(1)

    metadata = []

    for filename in os.listdir(tsumegos_dir):
        if filename.endswith(".sgf"):
            file_path = os.path.join(tsumegos_dir, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            difficulty = extract_difficulty(content)

            if not difficulty:
                raise Exception(f"Warning: Could not determine difficulty for {filename}")

            metadata.append({
                "filename": filename,
                "rank": difficulty
            })

    # Write metadata to JSON file
    metadata_path = os.path.join("pwa", "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
