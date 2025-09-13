import json
import csv
import statistics

# Load the JSON data
with open('t.json', 'r') as f:
    data = json.load(f)

# Extract IDs based on status
success_ids = []
fail_ids = []
total_items = len(data)
print(f"Total entries in t.json: {total_items}")

for item in data:
    filename = item['filename']
    id_ = filename.replace('.sgf', '')
    status = item['status']
    if status == 'success':
        success_ids.append(id_)
    else:
        fail_ids.append(id_)

print(f"Success IDs: {len(success_ids)}")
print(f"Fail IDs: {len(fail_ids)}")

# Load ratings from CSV
ratings = {}
with open('ratings/tsumego_ratings_final.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Columns are 'tsumego_id' and 'rating'
        ratings[row['tsumego_id']] = float(row['rating'])

# Load quality from CSV
quality = {}
with open('ratings/tsumego_quality_final.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        quality[row['tsumego_id']] = float(row['quality'])

print(f"Total ratings in CSV: {len(ratings)}")

# Collect ratings and quality for success and fail IDs
success_ratings = []
success_quality = []
missing_success = []
for id_ in success_ids:
    if id_ in ratings:
        success_ratings.append(ratings[id_])
        if id_ in quality:
            success_quality.append(quality[id_])
    else:
        missing_success.append(id_)

fail_ratings = []
fail_quality = []
missing_fail = []
for id_ in fail_ids:
    if id_ in ratings:
        fail_ratings.append(ratings[id_])
        if id_ in quality:
            fail_quality.append(quality[id_])
    else:
        missing_fail.append(id_)

print(f"Success ratings found: {len(success_ratings)}, missing: {len(missing_success)}")
print(f"Fail ratings found: {len(fail_ratings)}, missing: {len(missing_fail)}")

def print_distribution(ratings, label):
    if not ratings:
        print(f"{label}: No data")
        return
    ratings_sorted = sorted(ratings)
    n = len(ratings_sorted)
    mean = statistics.mean(ratings_sorted)
    std = statistics.stdev(ratings_sorted)
    min_val = ratings_sorted[0]
    max_val = ratings_sorted[-1]
    p25 = ratings_sorted[int(0.25 * (n-1))]
    p50 = ratings_sorted[int(0.5 * (n-1))]
    p75 = ratings_sorted[int(0.75 * (n-1))]
    print(f"{label}: count={n}, mean={mean:.2f}, std={std:.2f}, min={min_val:.2f}, 25%={p25:.2f}, 50%={p50:.2f}, 75%={p75:.2f}, max={max_val:.2f}")

# Compute and print statistics
print_distribution(success_ratings, "Success tsumegos ratings")
print_distribution(success_quality, "Success tsumegos quality")
print_distribution(fail_ratings, "Fail tsumegos ratings")
print_distribution(fail_quality, "Fail tsumegos quality")