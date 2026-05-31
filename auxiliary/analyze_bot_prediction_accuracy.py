"""
Analyzes how well each bot predicts human results in 2-month overlapping chunks.

This script:
1. Loads human attempt data from database.sql
2. Splits user profiles into overlapping 2-month chunks (same windowing as the rating pipeline)
3. For each bot, calculates correlation between bot success rates and human success rates
4. Evaluates prediction accuracy using multiple metrics
"""

import os
import sys
import re
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Configuration ---
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "attempts")
MIN_ATTEMPTS_PER_PROFILE = 30  # Minimum attempts required for a user profile to be included


def load_human_attempts_with_profiles(sql_file):
    """
    Loads human attempt data from database.sql and creates overlapping 2-month profiles.
    
    Returns:
        DataFrame with columns: user_profile_id, tsumego_id, success
        where user_profile_id is formatted as {user_id[:8]}_{YYYY-MM}
    """
    print(f"Loading human data from {sql_file}...")
    
    user_attempts = []
    insert_pattern = re.compile(r"INSERT INTO [\"`]?tsumego_reports[\"`]? VALUES\(\d+,'([^']+)','([^']+)',([01]),'([^']+)'\);")
    
    with open(sql_file, 'r') as f:
        for line in tqdm(f, desc="Parsing SQL"):
            match = insert_pattern.match(line)
            if match:
                user_id, tsumego_id, success, timestamp_str = match.groups()
                
                try:
                    # Parse timestamp
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    # Create overlapping profiles (current month and previous month)
                    profiles_to_add = []
                    
                    # 1. Current month profile
                    current_profile = f"{user_id[:8]}_{dt.year}-{dt.month:02d}"
                    profiles_to_add.append(current_profile)
                    
                    # 2. Previous month profile (for overlap)
                    prev_month = dt.month - 1
                    prev_year = dt.year
                    if prev_month == 0:
                        prev_month = 12
                        prev_year -= 1
                    prev_profile = f"{user_id[:8]}_{prev_year}-{prev_month:02d}"
                    profiles_to_add.append(prev_profile)
                    
                    for profile_id in profiles_to_add:
                        user_attempts.append({
                            'user_profile_id': profile_id,
                            'tsumego_id': tsumego_id,
                            'success': int(success)
                        })
                        
                except ValueError as e:
                    print(f"Skipping malformed timestamp: {timestamp_str} ({e})")
                    continue
    
    df = pd.DataFrame(user_attempts)
    print(f"Loaded {len(df)} total attempts (with overlapping profiles)")
    
    # Filter profiles by minimum attempts
    profile_counts = df.groupby('user_profile_id').size()
    valid_profiles = profile_counts[profile_counts >= MIN_ATTEMPTS_PER_PROFILE].index
    df = df[df['user_profile_id'].isin(valid_profiles)]
    
    print(f"After filtering (min {MIN_ATTEMPTS_PER_PROFILE} attempts): {len(df)} attempts across {len(valid_profiles)} profiles")
    
    return df


def load_bot_data(data_dir):
    """
    Loads bot evaluation data from .txt files.
    
    Returns:
        Dictionary: {bot_name: {tsumego_id: success_rate}}
    """
    print("\nLoading bot data...")
    
    bot_data = {}
    bot_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    if not bot_files:
        print(f"Warning: No bot data files (.txt) found in '{data_dir}'")
        return bot_data
    
    for bot_file in tqdm(bot_files, desc="Loading Bot Files"):
        bot_name = os.path.splitext(os.path.basename(bot_file))[0]
        bot_results = {}
        
        with open(bot_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    tsumego_name, success_rate = parts
                    bot_results[tsumego_name] = float(success_rate)
        
        bot_data[bot_name] = bot_results
        
    print(f"Loaded {len(bot_data)} bots")
    return bot_data


def calculate_profile_success_rates(human_df):
    """
    Calculates average success rate for each (user_profile, tsumego) pair.
    
    Returns:
        DataFrame with columns: user_profile_id, tsumego_id, human_success_rate
    """
    print("\nCalculating human success rates per profile...")
    
    # Group by profile and tsumego, calculate mean success rate
    profile_rates = human_df.groupby(['user_profile_id', 'tsumego_id'])['success'].mean().reset_index()
    profile_rates.rename(columns={'success': 'human_success_rate'}, inplace=True)
    
    print(f"Calculated success rates for {len(profile_rates)} (profile, tsumego) pairs")
    return profile_rates


def evaluate_bot_predictions(profile_rates, bot_data):
    """
    Evaluates how well each bot predicts human success rates.
    
    For each bot, calculates:
    - Pearson correlation
    - Spearman correlation
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Coverage (% of human attempts the bot has evaluated)
    
    Returns:
        DataFrame with bot evaluation metrics
    """
    print("\nEvaluating bot prediction accuracy...")
    print(f"{'Bot Name':<50} {'Pearson':<10} {'Spearman':<10} {'RMSE':<10} {'Coverage':<10} {'N':<10}")
    print("=" * 100)
    
    results = []
    
    # Sort bots alphabetically by name
    sorted_bot_names = sorted(bot_data.keys())
    
    for bot_name in sorted_bot_names:
        bot_results = bot_data[bot_name]
        # Merge bot predictions with human results
        merged_data = []
        
        for _, row in profile_rates.iterrows():
            tsumego_id = row['tsumego_id']
            human_rate = row['human_success_rate']
            
            if tsumego_id in bot_results:
                bot_rate = bot_results[tsumego_id]
                merged_data.append({
                    'human_rate': human_rate,
                    'bot_rate': bot_rate
                })
        
        if len(merged_data) < 10:  # Need at least 10 data points for meaningful correlation
            continue
        
        merged_df = pd.DataFrame(merged_data)
        
        # Calculate metrics
        human_rates = merged_df['human_rate'].values
        bot_rates = merged_df['bot_rate'].values
        
        # Correlations
        pearson_corr, pearson_p = pearsonr(human_rates, bot_rates)
        spearman_corr, spearman_p = spearmanr(human_rates, bot_rates)
        
        # Error metrics
        mse = mean_squared_error(human_rates, bot_rates)
        mae = mean_absolute_error(human_rates, bot_rates)
        rmse = np.sqrt(mse)
        
        # Coverage
        total_human_attempts = len(profile_rates)
        covered_attempts = len(merged_data)
        coverage = covered_attempts / total_human_attempts
        
        result = {
            'bot_name': bot_name,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'n_comparisons': len(merged_data)
        }
        results.append(result)
        
        # Print result immediately
        print(f"{bot_name:<50} {pearson_corr:<10.4f} {spearman_corr:<10.4f} {rmse:<10.4f} {coverage:<10.2%} {len(merged_data):<10}")
    
    results_df = pd.DataFrame(results)
    
    # Sort by Pearson correlation (descending)
    results_df = results_df.sort_values('pearson_correlation', ascending=False)
    
    return results_df


def analyze_by_profile_count(profile_rates, bot_data):
    """
    Analyzes bot prediction accuracy grouped by number of user profiles.
    
    This helps understand if bots perform better/worse on problems attempted by
    more or fewer users.
    """
    print("\nAnalyzing by profile count...")
    
    # Count how many profiles attempted each tsumego
    tsumego_profile_counts = profile_rates.groupby('tsumego_id')['user_profile_id'].nunique().reset_index()
    tsumego_profile_counts.rename(columns={'user_profile_id': 'n_profiles'}, inplace=True)
    
    # Merge with profile rates
    profile_rates_with_counts = profile_rates.merge(tsumego_profile_counts, on='tsumego_id')
    
    # Define bins for profile counts
    bins = [0, 5, 10, 20, 50, 100, float('inf')]
    labels = ['1-5', '6-10', '11-20', '21-50', '51-100', '100+']
    profile_rates_with_counts['profile_count_bin'] = pd.cut(
        profile_rates_with_counts['n_profiles'],
        bins=bins,
        labels=labels
    )
    
    # For each bot, calculate correlation within each bin
    bin_results = []
    
    for bot_name, bot_results in tqdm(bot_data.items(), desc="Analyzing by Profile Count"):
        for bin_label in labels:
            bin_data = profile_rates_with_counts[profile_rates_with_counts['profile_count_bin'] == bin_label]
            
            merged_data = []
            for _, row in bin_data.iterrows():
                tsumego_id = row['tsumego_id']
                human_rate = row['human_success_rate']
                
                if tsumego_id in bot_results:
                    bot_rate = bot_results[tsumego_id]
                    merged_data.append({
                        'human_rate': human_rate,
                        'bot_rate': bot_rate
                    })
            
            if len(merged_data) < 10:
                continue
            
            merged_df = pd.DataFrame(merged_data)
            human_rates = merged_df['human_rate'].values
            bot_rates = merged_df['bot_rate'].values
            
            pearson_corr, _ = pearsonr(human_rates, bot_rates)
            
            bin_results.append({
                'bot_name': bot_name,
                'profile_count_bin': bin_label,
                'pearson_correlation': pearson_corr,
                'n_comparisons': len(merged_data)
            })
    
    bin_results_df = pd.DataFrame(bin_results)
    
    return bin_results_df


def main():
    """Main execution function."""
    
    # Load data
    sql_file = os.path.join(DATA_DIRECTORY, 'database.sql')
    human_df = load_human_attempts_with_profiles(sql_file)
    bot_data = load_bot_data(DATA_DIRECTORY)
    
    if human_df.empty or not bot_data:
        print("Error: No data loaded. Exiting.")
        return
    
    # Calculate human success rates per profile
    profile_rates = calculate_profile_success_rates(human_df)
    
    # Evaluate bot predictions
    results_df = evaluate_bot_predictions(profile_rates, bot_data)
    
    # Save overall results
    output_file = os.path.join(PROJECT_ROOT, 'auxiliary', 'bot_prediction_accuracy.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n=== Overall Bot Prediction Accuracy ===")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_file}")
    
    # Analyze by profile count
    bin_results_df = analyze_by_profile_count(profile_rates, bot_data)
    
    # Save bin results
    bin_output_file = os.path.join(PROJECT_ROOT, 'auxiliary', 'bot_prediction_by_profile_count.csv')
    bin_results_df.to_csv(bin_output_file, index=False)
    print(f"\n=== Bot Prediction Accuracy by Profile Count ===")
    
    # Show pivot table for easier reading
    if not bin_results_df.empty:
        pivot = bin_results_df.pivot(
            index='bot_name',
            columns='profile_count_bin',
            values='pearson_correlation'
        )
        print(pivot.to_string())
        print(f"\nBin results saved to: {bin_output_file}")
    
    print("\n=== Summary Statistics ===")
    print(f"Best bot (Pearson): {results_df.iloc[0]['bot_name']} ({results_df.iloc[0]['pearson_correlation']:.4f})")
    print(f"Best bot (Spearman): {results_df.sort_values('spearman_correlation', ascending=False).iloc[0]['bot_name']} "
          f"({results_df.sort_values('spearman_correlation', ascending=False).iloc[0]['spearman_correlation']:.4f})")
    print(f"Best bot (RMSE): {results_df.sort_values('rmse').iloc[0]['bot_name']} "
          f"({results_df.sort_values('rmse').iloc[0]['rmse']:.4f})")
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
