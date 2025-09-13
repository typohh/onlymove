import csv

def load_ratings():
    ratings_list = []
    with open('ratings/tsumego_ratings_final.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings_list.append(float(row['rating']))
    ratings_list.sort()
    return ratings_list

def get_percentile_value(percentile, data):
    if not data:
        return float('inf')
    idx = int((percentile / 100) * (len(data) - 1))
    return data[idx]

def get_actual_percentile(value, data):
    if not data:
        return 0.0
    pos = 0
    for i, v in enumerate(data):
        if v > value:
            break
        pos = i + 1
    return (pos / len(data)) * 100

def find_x_by_rating(ratings_list, top_percent, value=2.77):
    total = len(ratings_list)
    bottom_1 = int(0.01 * total)
    remaining = ratings_list[bottom_1:]
    min_val = remaining[0]
    # value = min + (1 - top_percent/100) * (max - min)
    # value = min + (1 - factor) * (max - min)
    # value - min = (1 - factor) * (max - min)
    # (value - min) / (1 - factor) = max - min
    # max = min + (value - min) / (1 - factor)
    factor = top_percent / 100
    target_max = min_val + (value - min_val) / (1 - factor)
    # Find the largest idx where remaining[idx] <= target_max
    idx = 0
    for i, r in enumerate(remaining):
        if r > target_max:
            break
        idx = i
    top_x_count = len(remaining) - idx - 1
    x = (top_x_count / len(remaining)) * 100
    return round(x, 2), target_max

def get_final_data(ratings_list, x):
    total = len(ratings_list)
    bottom_1 = int(0.01 * total)
    remaining = ratings_list[bottom_1:]
    top_x = int((x / 100) * len(remaining))
    final = remaining[:-top_x] if top_x > 0 else remaining
    return final

if __name__ == "__main__":
    ratings_list = load_ratings()
    print(f"Total ratings: {len(ratings_list)}")

    # For top 17% by rating
    x_17, target_max_17 = find_x_by_rating(ratings_list, 17.0)
    final_17 = get_final_data(ratings_list, x_17)
    min_17 = min(final_17) if final_17 else 0
    max_17 = max(final_17) if final_17 else 0
    print(f"For rating {2.77} to be top 17% by rating: exclude top {x_17}%, target max={target_max_17:.2f}, actual max={max_17:.2f}, min={min_17:.2f}")

    # For top 14% by rating
    x_14, target_max_14 = find_x_by_rating(ratings_list, 14.0)
    final_14 = get_final_data(ratings_list, x_14)
    min_14 = min(final_14) if final_14 else 0
    max_14 = max(final_14) if final_14 else 0
    print(f"For rating {2.77} to be top 14% by rating: exclude top {x_14}%, target max={target_max_14:.2f}, actual max={max_14:.2f}, min={min_14:.2f}")