import hashlib
import json
import os
import re

from katago_queries import get_rank_number


def rename_tsumegos(tsumegos_dir):
    sgf_files = [f for f in os.listdir(tsumegos_dir) if f.endswith('.sgf')]
    for filename in sgf_files:
        filepath = os.path.join(tsumegos_dir, filename)
        with open(filepath, 'rb') as f:
            file_bytes = f.read()
            new_filename = hashlib.sha256(file_bytes).hexdigest()[:8] + ".sgf"
            os.rename(filepath, os.path.join(tsumegos_dir, new_filename))


def extract_difficulty(content):
    """Extract difficulty rank from SGF content."""
    patterns = [
        r"GN\[Move difficulty: (\d+[kd])±",
        r"GN\[Last move difficulty: (\d+[kd])±",
        r"Game difficulty: (\d+[kd])±",
        r"remaining game difficulty: (\d+[kd])±"
    ]

    total = 0
    count = 0

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            rank_txt = match.group(1)
            rank_number = get_rank_number(rank_txt)
            total += rank_number
            count += 1

    return total / count


if __name__ == "__main__":
    tsumegos_dir = "tsumegos"
    if not os.path.exists(tsumegos_dir):
        print(f"Error: {tsumegos_dir} directory not found")
        exit(1)

    rename_tsumegos(tsumegos_dir)

    metadata = []
    rank_distribution = []

    for filename in os.listdir(tsumegos_dir):
        if filename.endswith(".sgf"):
            file_path = os.path.join(tsumegos_dir, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            rank_num = extract_difficulty(content)

            if rank_num is None:
                raise Exception(f"Warning: Could not determine difficulty for {filename}")

            metadata.append({
                "filename": filename,
                "rank": rank_num
            })

            rank_distribution.append(rank_num)

    # Sort metadata by rank and assign groups
    metadata.sort(key=lambda x: x["rank"])
    for i, item in enumerate(metadata):
        item["group"] = int(10 * i / len(metadata))

    # Create zip files for each group
    import zipfile

    for group in range(10):
        group_files = [item["filename"] for item in metadata if item["group"] == group]
        zip_path = os.path.join("pwa", f"tsumegos{group}.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in group_files:
                file_path = os.path.join(tsumegos_dir, filename)
                zipf.write(file_path, filename)

    # Write metadata to JSON file
    metadata_path = os.path.join("pwa", "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # Print rank distribution sorted by rank_num
    rank_distribution = sorted(rank_distribution)
    print("\nRank Distribution:")
    for i in range(29):
        min_rank = i/29
        max_rank = (i+1)/29
        # print aproximately how many of rank_distrubitn is between min_rank and max_rank
        count = sum(1 for x in rank_distribution if min_rank <= x <= max_rank)
        print(f"Rank {i}: {count} problems")
