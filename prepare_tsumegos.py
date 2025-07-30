import hashlib
import json
import os
import re

from katago_queries import get_rank_number, get_rank_str


def fix_tsumegos(tsumegos_dir):
    sgf_files = [f for f in os.listdir(tsumegos_dir) if f.endswith('.sgf')]
    for filename in sgf_files:
        filepath = os.path.join(tsumegos_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            standardized = standardize_text(content)
            if standardized is not None:
                content = standardized

                pattern = r"GN\[Move: (\d+[kd]), Game: (\d+[kd])\]"
                match = re.search(pattern, content)
                if not match :
                    raise Exception(f"Error: Could not find difficulty in {content}")

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
        with open(filepath, 'rb') as f:
            file_bytes = f.read()
            new_filename = hashlib.sha256(file_bytes).hexdigest()[:8] + ".sgf"
            os.rename(filepath, os.path.join(tsumegos_dir, new_filename))



def count_black_moves(content):
    """Count number of black moves before first parenthesis."""
    content = content[1:]  # Skip first character
    first_paren = content.find('(')
    if first_paren == -1:
        return content.count(';B[')
    return content[:first_paren].count(';B[')


def find_difference_in_score(content):
    """Find difference between two highest scores in content."""
    pattern = r'Score: (-?\d+\.?\d*)'
    matches = re.findall(pattern, content)
    scores = sorted([float(score) for score in matches], reverse=True)
    return scores[0] - scores[1]


def standardize_text(content):
    # Pattern for "Move difficulty: Xk±0.0, Game difficulty: Yd±0.0" format
    pattern1 = r'GN\[Move difficulty: (\d+[kd])±\d+\.\d+, Game difficulty: (\d+[kd])±\d+\.\d+\]'
    # Pattern for "Last move difficulty: Xk±0.0, Last move + remaining game difficulty: Yd±0.0" format
    pattern2 = r'GN\[Last move difficulty: (\d+[kd])±\d+\.\d+, Last move \+ remaining game difficulty: (\d+[kd])±\d+\.\d+\]'

    new_content = content

    # Replace first pattern with simplified format
    match1 = re.search(pattern1, content)
    if match1:
        move_rank = match1.group(1)
        game_rank = match1.group(2)
        new_content = re.sub(pattern1, f'GN[Move: {move_rank}, Game: {game_rank}]', new_content)

    # Replace second pattern with simplified format    
    match2 = re.search(pattern2, content)
    if match2:
        move_rank = match2.group(1)
        game_rank = match2.group(2)
        new_content = re.sub(pattern2, f'GN[Move: {move_rank}, Game: {game_rank}]', new_content)

    return new_content if new_content != content else None


def extract_difficulty(content):
    """Extract move and game ranks from the GN tag and return their average."""
    pattern = r"GN\[Move: (\d+[kd]), Game: (\d+[kd])\]"
    match = re.search(pattern, content)

    if not match:
        raise Exception(f"Error: Could not find difficulty in {content}")

    move_rank = get_rank_number(match.group(1))
    game_rank = get_rank_number(match.group(2))
    return ((move_rank + game_rank) / 2)


if __name__ == "__main__":
    tsumegos_dir = "tsumegos"
    if not os.path.exists(tsumegos_dir):
        print(f"Error: {tsumegos_dir} directory not found")
        exit(1)

    fix_tsumegos(tsumegos_dir)


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
        min_rank = i / 29
        max_rank = (i + 1) / 29
        # print aproximately how many of rank_distrubitn is between min_rank and max_rank
        count = sum(1 for x in rank_distribution if min_rank <= x <= max_rank)
        print(f"{get_rank_str((min_rank + max_rank) / 2)}: {count} problems")
