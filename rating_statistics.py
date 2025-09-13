# --- Histograms and Correlation ---
def print_histograms_and_correlation():
    import pandas as pd
    import numpy as np

    # Load data
    rating_df = pd.read_csv('ratings/tsumego_ratings_final.csv')
    quality_df = pd.read_csv('ratings/tsumego_quality_final.csv')
    # params_df = pd.read_csv('ratings/tsumego_parameters_final.csv')
    # df = rating_df.merge(quality_df, on='tsumego_id').merge(params_df, on='tsumego_id')

    df = rating_df.merge(quality_df, on='tsumego_id')

    columns = ['rating', 'quality']

    # Histograms
    for col in columns:
        data = df[col].dropna()
        if len(data) == 0:
            print(f"\nNo data for {col}")
            continue
        hist, bins = np.histogram(data, bins=10)
        print(f"\nHistogram for {col}:")
        max_count = max(hist) if max(hist) > 0 else 1
        for i in range(10):
            bar = '*' * int(hist[i] / max_count * 50)
            print(f"{bins[i]:.2f} - {bins[i+1]:.2f}: {bar} ({hist[i]})")

    # Correlation
    corr = df[columns].corr()
    print("\nCorrelation Matrix:")
    print(corr)

# --- Sample Tsumegos Near Rating with Categories ---
def sample_tsumegos_near_rating_with_categories(target_rating, num_closest=500, num_per_category=3):
    import os
    import pandas as pd
    import numpy as np

    tsumegos_dir = "tsumegos"

    # Load data
    rating_df = pd.read_csv('ratings/tsumego_ratings_final.csv')
    quality_df = pd.read_csv('ratings/tsumego_quality_final.csv')
    params_df = pd.read_csv('ratings/tsumego_parameters_final.csv')
    bot_df = pd.read_csv('ratings/bot_ratings_final.csv')

    # Merge tsumego data
    df = rating_df.merge(quality_df, on='tsumego_id').merge(params_df, on='tsumego_id')

    # Filter to monotonic (where difficulty is not nan)
    df = df.dropna(subset=['difficulty'])

    # Bot ratings dict
    bot_ratings_dict = dict(zip(bot_df['bot_id'], bot_df['rating']))

    # Filter to only include tsumegos that have SGF files
    df = df[df['tsumego_id'].apply(lambda tid: os.path.exists(os.path.join(tsumegos_dir, f"{tid}.sgf")))]

    # Filter to closest num_closest tsumegos
    df['abs_diff'] = (df['rating'] - target_rating).abs()
    near_rating_df = df.nsmallest(num_closest, 'abs_diff').drop(columns=['abs_diff'])

    if near_rating_df.empty:
        print(f"No tsumegos found near rating {target_rating}.")
        return

    # Compute medians for categorization
    median_disc = near_rating_df['discrimination'].median()
    median_guess = near_rating_df['guessing'].median()

    # Categorize
    near_rating_df = near_rating_df.copy()
    near_rating_df['low_disc'] = near_rating_df['discrimination'] < median_disc
    near_rating_df['low_guess'] = near_rating_df['guessing'] < median_guess

    # Categories
    categories = {
        'Low Disc, Low Guess': near_rating_df[(near_rating_df['low_disc']) & (near_rating_df['low_guess'])],
        'Low Disc, High Guess': near_rating_df[(near_rating_df['low_disc']) & (~near_rating_df['low_guess'])],
        'High Disc, Low Guess': near_rating_df[(~near_rating_df['low_disc']) & (near_rating_df['low_guess'])],
        'High Disc, High Guess': near_rating_df[(~near_rating_df['low_disc']) & (~near_rating_df['low_guess'])]
    }

    # Sample and print
    print(f"\n--- Sampled Tsumegos Near Rating {target_rating} (Closest {num_closest}) ---\n")
    for cat_name, cat_df in categories.items():
        print(f"{cat_name}:")
        if cat_df.empty:
            print("  No tsumegos in this category.")
            continue
        # Normalize discrimination and guessing within category
        min_disc = cat_df['discrimination'].min()
        max_disc = cat_df['discrimination'].max()
        if max_disc > min_disc:
            normalized_disc = (cat_df['discrimination'] - min_disc) / (max_disc - min_disc)
        else:
            normalized_disc = pd.Series([0.5] * len(cat_df), index=cat_df.index)

        min_guess = cat_df['guessing'].min()
        max_guess = cat_df['guessing'].max()
        if max_guess > min_guess:
            normalized_guess = (cat_df['guessing'] - min_guess) / (max_guess - min_guess)
        else:
            normalized_guess = pd.Series([0.5] * len(cat_df), index=cat_df.index)

        # Combine to represent the new direction
        combined_score = normalized_disc * normalized_guess

        cat_df = cat_df.copy()
        cat_df['combined_score'] = combined_score

        # Select the most extreme (highest combined score)
        num_to_select = min(num_per_category, len(cat_df))
        sampled = cat_df.nlargest(num_to_select, 'combined_score')
        for _, row in sampled.iterrows():
            tsumego_id = row['tsumego_id']
            rating_val = row['rating']
            # Find closest bot
            closest_bot = min(bot_ratings_dict, key=lambda b: abs(bot_ratings_dict[b] - rating_val))
            print(f"{tsumego_id} {rating_val:.2f} {row['quality']:.2f} {row['difficulty']:.2f} {row['discrimination']:.2f} {row['guessing']:.4f} {closest_bot} http://localhost:8000/?id={tsumego_id}")
        print()


# --- Sample Tsumegos Linearly ---
def sample_tsumegos(rating_start, rating_end, quality_start, quality_end, num_samples):
    """
    Finds and lists a specified number of tsumegos that best fit a linear interpolation
    between a start and end (rating, quality) pair.
    """
    import os
    import pandas as pd
    import numpy as np

    tsumegos_dir = "tsumegos"

    # Load data
    try:
        rating_df = pd.read_csv('ratings/tsumego_ratings_final.csv')
        quality_df = pd.read_csv('ratings/tsumego_quality_final.csv')
        bot_df = pd.read_csv('ratings/bot_ratings_final.csv')
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}. Make sure the ratings CSV files are in the 'ratings/' directory.")
        return

    # Merge tsumego data
    df = rating_df.merge(quality_df, on='tsumego_id')

    # Drop rows with NaN in rating or quality, as they can't be used
    df.dropna(subset=['rating', 'quality'], inplace=True)
    if df.empty:
        print("No tsumego data available after merging and dropping NaNs.")
        return

    # Bot ratings dict
    bot_ratings_dict = dict(zip(bot_df['bot_id'], bot_df['rating']))

    # Filter to only include tsumegos that have SGF files
    df = df[df['tsumego_id'].apply(lambda tid: os.path.exists(os.path.join(tsumegos_dir, f"{tid}.sgf")))]

    if df.empty:
        print("No tsumegos found with corresponding SGF files.")
        return

    # Create a copy to use as the pool of candidates
    candidate_df = df.copy()

    # Calculate normalization stddev from the entire dataset
    rating_range = df['rating'].std() / 5
    quality_range = df['quality'].std()

    # Avoid division by zero if all values are the same
    if pd.isna(rating_range) or rating_range == 0: rating_range = 1.0
    if pd.isna(quality_range) or quality_range == 0: quality_range = 1.0

    selected_tsumegos = []

    print(f"\n--- Searching for {num_samples} Tsumegos along the line from (R:{rating_start}, Q:{quality_start}) to (R:{rating_end}, Q:{quality_end}) ---\n")

    for i in range(num_samples):
        if candidate_df.empty:
            print("Warning: Ran out of tsumegos to sample from.")
            break

        # Calculate the target (rating, quality) point for this step
        if num_samples > 1:
            fraction = i / (num_samples - 1)
        else:
            fraction = 0.5  # For a single sample, target the midpoint

        target_rating = rating_start + fraction * (rating_end - rating_start)
        target_quality = quality_start + fraction * (quality_end - quality_start)

        # Calculate the squared normalized Euclidean distance for each candidate tsumego to the target point.
        normalized_rating_dist = (candidate_df['rating'] - target_rating) / rating_range
        normalized_quality_dist = (candidate_df['quality'] - target_quality) / quality_range

        candidate_df['distance'] = normalized_rating_dist ** 2 + normalized_quality_dist ** 2

        # Find the index of the tsumego with the minimum distance
        closest_idx = candidate_df['distance'].idxmin()

        # Get the closest tsumego
        closest_tsumego = candidate_df.loc[closest_idx]

        # Add it to our results list
        selected_tsumegos.append(closest_tsumego)

        # Remove the selected tsumego from the candidate pool to avoid re-selection
        candidate_df = candidate_df.drop(closest_idx)

    # Print the results
    if not selected_tsumegos:
        print("No tsumegos were selected.")
        return

    print(f"Found {len(selected_tsumegos)} samples:")
    print(f"{'tsumego_id':<10} {'rating':>7} {'quality':>7} {'closest_bot':<15} {'url'}")
    print("-" * 70)

    # Create a DataFrame from the list of Series for easier handling
    results_df = pd.DataFrame(selected_tsumegos)

    for _, row in results_df.iterrows():
        tsumego_id = row['tsumego_id']
        rating_val = row['rating']

        # Find closest bot
        closest_bot = min(bot_ratings_dict, key=lambda b: abs(bot_ratings_dict[b] - rating_val))

        print(f"{tsumego_id:<10} {rating_val:>7.2f} {row['quality']:>7.2f} {closest_bot:<15} http://localhost:8000/?id={tsumego_id}")
    print()



# --- Sample Tsumegos Near Rating by Quality ---
def sample_tsumegos_near_rating_by_quality(target_rating, num_closest=500, num_per_category=3):
    import os
    import pandas as pd
    import numpy as np

    tsumegos_dir = "tsumegos"

    # Load data
    rating_df = pd.read_csv('ratings/tsumego_ratings_final.csv')
    quality_df = pd.read_csv('ratings/tsumego_quality_final.csv')
    bot_df = pd.read_csv('ratings/bot_ratings_final.csv')

    # Merge tsumego data
    df = rating_df.merge(quality_df, on='tsumego_id')

    # Bot ratings dict
    bot_ratings_dict = dict(zip(bot_df['bot_id'], bot_df['rating']))

    # Filter to only include tsumegos that have SGF files
    df = df[df['tsumego_id'].apply(lambda tid: os.path.exists(os.path.join(tsumegos_dir, f"{tid}.sgf")))]

    # Filter to closest num_closest tsumegos
    df['abs_diff'] = (df['rating'] - target_rating).abs()
    near_rating_df = df.nsmallest(num_closest, 'abs_diff').drop(columns=['abs_diff'])

    if near_rating_df.empty:
        print(f"No tsumegos found near rating {target_rating}.")
        return

    # Compute median for categorization
    median_quality = near_rating_df['quality'].median()
    stddev_quality = near_rating_df['quality'].std()
    print(median_quality,stddev_quality)

    # Categorize
    near_rating_df = near_rating_df.copy()
    near_rating_df['low_quality'] = near_rating_df['quality'] <= median_quality

    # Categories
    categories = {
        'Low Quality': near_rating_df[near_rating_df['low_quality']],
        'High Quality': near_rating_df[~near_rating_df['low_quality']]
    }

    # Sample and print
    print(f"\n--- Sampled Tsumegos Near Rating {target_rating} by Quality (Closest {num_closest}) ---\n")
    for cat_name, cat_df in categories.items():
        print(f"{cat_name}:")
        if cat_df.empty:
            print("  No tsumegos in this category.")
            continue

        # Select the most extreme quality in each category
        if cat_name == 'Low Quality':
            sampled = cat_df.nsmallest(num_per_category, 'quality')
        else:
            sampled = cat_df.nlargest(num_per_category, 'quality')

        for _, row in sampled.iterrows():
            tsumego_id = row['tsumego_id']
            rating_val = row['rating']
            # Find closest bot
            closest_bot = min(bot_ratings_dict, key=lambda b: abs(bot_ratings_dict[b] - rating_val))
            print(f"{tsumego_id} {rating_val:.2f} {row['quality']:.2f} {closest_bot} http://localhost:8000/?id={tsumego_id}")
        print()


# --- Sample Tsumego Pairs with Similar Ratings but Different Qualities ---
def sample_tsumego_pairs_similar_rating_different_quality(num_pairs=10, bin_size=0.5):
    import os
    import pandas as pd
    import numpy as np

    tsumegos_dir = "tsumegos"

    # Load data
    try:
        rating_df = pd.read_csv('ratings/tsumego_ratings_final.csv')
        quality_df = pd.read_csv('ratings/tsumego_quality_final.csv')
        bot_df = pd.read_csv('ratings/bot_ratings_final.csv')
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}. Make sure the ratings CSV files are in the 'ratings/' directory.")
        return

    # Merge tsumego data
    df = rating_df.merge(quality_df, on='tsumego_id')

    # Drop rows with NaN in rating or quality
    df.dropna(subset=['rating', 'quality'], inplace=True)
    if df.empty:
        print("No tsumego data available after merging and dropping NaNs.")
        return

    # Bot ratings dict
    bot_ratings_dict = dict(zip(bot_df['bot_id'], bot_df['rating']))

    # Filter to only include tsumegos that have SGF files
    df = df[df['tsumego_id'].apply(lambda tid: os.path.exists(os.path.join(tsumegos_dir, f"{tid}.sgf")))]

    if df.empty:
        print("No tsumegos found with corresponding SGF files.")
        return

    # Sort by rating
    df = df.sort_values('rating').reset_index(drop=True)

    # Create bins
    df['bin'] = (df['rating'] // bin_size) * bin_size

    # Group by bin
    grouped = df.groupby('bin')

    pairs = []

    for bin_val, group in grouped:
        if len(group) < 2:
            continue
        # Find the pair with max quality difference
        max_diff = 0
        best_pair = None
        rows = group.to_dict('records')
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                diff = abs(rows[i]['quality'] - rows[j]['quality'])
                if diff > max_diff:
                    max_diff = diff
                    best_pair = (rows[i], rows[j])
        if best_pair:
            pairs.append((max_diff, best_pair[0], best_pair[1]))

    # Sort pairs by diff descending
    pairs.sort(key=lambda x: x[0], reverse=True)

    # Take top num_pairs
    selected_pairs = pairs[:num_pairs]

    # Sort selected pairs by average rating ascending
    selected_pairs.sort(key=lambda x: (x[1]['rating'] + x[2]['rating']) / 2)

    if not selected_pairs:
        print("No pairs found.")
        return

    print(f"\n--- Top {len(selected_pairs)} Tsumego Pairs with Similar Ratings but Large Quality Differences ---\n")

    for diff, row1, row2 in selected_pairs:
        print(f"Quality Difference: {diff:.2f}")
        for row in sorted([row1, row2], key=lambda r: r['quality']):
            tsumego_id = row['tsumego_id']
            rating_val = row['rating']
            quality_val = row['quality']
            closest_bot = min(bot_ratings_dict, key=lambda b: abs(bot_ratings_dict[b] - rating_val))
            print(f"{tsumego_id} {rating_val:.2f} {quality_val:.2f} {closest_bot} http://localhost:8000/?id={tsumego_id}")
        print()

if __name__ == '__main__':
    print_histograms_and_correlation()
    target_rating = -1
    # sample_tsumegos_near_rating_by_quality(-300,40,5)
    # sample_tsumegos(-2,3, 0, 0, 15)
    # sample_tsumegos(-2,3, 1, 1, 15)
    sample_tsumego_pairs_similar_rating_different_quality()
