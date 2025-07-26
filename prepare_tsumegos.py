import json
import os
import re

from katago_queries import get_rank_number


def extract_difficulty(content):
    """Extract difficulty rank from SGF content."""
    patterns = [
        r"GN\[Move difficulty: (\d+[kd])±",
        r"GN\[Last move difficulty: (\d+[kd])±"
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            rank_txt = match.group(1)
            rank_number = get_rank_number(rank_txt)
            return rank_number, rank_txt

    return None, None


if __name__ == "__main__":
    tsumegos_dir = "pwa/tsumegos"
    if not os.path.exists(tsumegos_dir):
        print(f"Error: {tsumegos_dir} directory not found")
        exit(1)

    metadata = []
    rank_distribution = {}

    for filename in os.listdir(tsumegos_dir):
        if filename.endswith(".sgf"):
            file_path = os.path.join(tsumegos_dir, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            rank_num,rank_txt = extract_difficulty(content)

            if rank_num is None:
                raise Exception(f"Warning: Could not determine difficulty for {filename}")

            metadata.append({
                "filename": filename,
                "rank": rank_num
            })

            rank_distribution[rank_txt] = rank_distribution.get(rank_txt, 0) + 1

    # Print rank distribution sorted by rank_num
    print("\nRank Distribution:")
    sorted_ranks = sorted(rank_distribution.items(), key=lambda x: get_rank_number(x[0]))
    for rank, count in sorted_ranks:
        print(f"{rank}: {count}")

    # Write metadata to JSON file
    metadata_path = os.path.join("pwa", "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

