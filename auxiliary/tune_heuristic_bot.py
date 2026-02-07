"""
Tunes the heuristic bot parameters to minimize MSE with weak human success rates.

The heuristic uses two features:
1. moves_remaining: fewer moves left = easier
2. score_diff: larger gap between best and second-best move = more obvious
"""

import os
import sys
import re
import math
import json
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

import sgfmill.sgf

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

TSUMEGO_DIR = os.path.join(PROJECT_ROOT, "tsumegos")
ATTEMPTS_DIRECTORY = os.path.join(PROJECT_ROOT, "attempts")
RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")
CACHE_FILE = os.path.join(PROJECT_ROOT, "auxiliary", "problem_features_cache.json")

# Weak human threshold
WEAK_HUMAN_MAX_RATING = 0.75 # 6 kyu
WEAK_HUMAN_MIN_RATING = 0.0 # 13 kyu


def load_user_ratings():
    """Loads user ratings from user_profiles_final.csv."""
    filepath = os.path.join(RATINGS_DIR, "user_profiles_final.csv")
    df = pd.read_csv(filepath)
    # Filter out bots
    human_df = df[~df['bot_id'].str.contains('^b18c|^kata|^heuristic', regex=True)]
    return human_df.set_index('bot_id')['rating'].to_dict()


def load_human_attempts():
    """Loads human attempts from database.sql with profile info."""
    from datetime import datetime
    sql_file = os.path.join(ATTEMPTS_DIRECTORY, 'database.sql')
    
    attempts = []
    insert_pattern = re.compile(
        r"INSERT INTO [\"`]?tsumego_reports[\"`]? VALUES\(\d+,'([^']+)','([^']+)',([01]),'([^']+)'\);"
    )
    
    print("Loading human attempts from database.sql...")
    with open(sql_file, 'r') as f:
        for line in f:
            match = insert_pattern.match(line)
            if match:
                user_id, tsumego_id, success, timestamp_str = match.groups()
                try:
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    profile_id = f"{user_id[:8]}_{dt.year}-{dt.month:02d}"
                    attempts.append({
                        'profile_id': profile_id,
                        'tsumego_id': tsumego_id,
                        'success': int(success)
                    })
                except ValueError:
                    continue
    
    return pd.DataFrame(attempts)


def load_or_extract_features(tsumego_ids):
    """Loads features from cache or extracts them from SGF files."""
    # Try to load from cache
    if os.path.exists(CACHE_FILE):
        print(f"Loading features from cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'r') as f:
            cached = json.load(f)
        
        # Check if we need to extract new ones
        missing = [tid for tid in tsumego_ids if tid not in cached]
        if missing:
            print(f"Extracting {len(missing)} missing features...")
            new_features = extract_problem_features(missing)
            cached.update(new_features)
            # Save updated cache
            with open(CACHE_FILE, 'w') as f:
                json.dump(cached, f)
        
        return cached
    else:
        print("No cache found, extracting all features...")
        features = extract_problem_features(tsumego_ids)
        # Save cache
        with open(CACHE_FILE, 'w') as f:
            json.dump(features, f)
        print(f"Saved cache to {CACHE_FILE}")
        return features


def extract_problem_features(tsumego_ids):
    """Extracts features from tsumego SGF files."""
    features = {}
    
    for tsumego_id in tqdm(tsumego_ids, desc="Extracting features"):
        filepath = os.path.join(TSUMEGO_DIR, f"{tsumego_id}.sgf")
        if not os.path.exists(filepath):
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            game = sgfmill.sgf.Sgf_game.from_string(content)
            
            root_node = game.get_root()
            current_node = root_node
            
            while len(current_node) == 1:
                current_node = current_node[0]
            
            variation_nodes = list(current_node)
            if not variation_nodes:
                continue
            
            # Extract scores
            scores = []
            for node in variation_nodes:
                if not (node.has_property("B") or node.has_property("W")):
                    continue
                for comment_bytes in node.get_raw_list("C"):
                    comment = comment_bytes.decode('utf-8')
                    match = re.search(r"(\d+\.?\d*)%/([-]?\d+\.?\d*)", comment)
                    if match:
                        scores.append(float(match.group(2)))
                        break
            
            if len(scores) < 1:
                continue
            
            scores.sort(reverse=True)
            score_diff = scores[0] - scores[1] if len(scores) > 1 else 100.0
            
            # Count moves remaining
            moves_remaining = 1
            current = variation_nodes[0]
            while len(current) > 0:
                current = current[0]
                moves_remaining += 1
            
            features[tsumego_id] = {
                'moves_remaining': moves_remaining,
                'score_diff': score_diff
            }
            
        except Exception:
            continue
    
    return features


def calculate_heuristic_prob(moves_remaining, score_diff, params):
    """Calculates heuristic probability."""
    p_score = 1.0 - math.exp(-score_diff / params['score_decay'])
    p_moves = math.exp(-moves_remaining / params['moves_decay'])
    
    w_sum = params['score_weight'] + params['moves_weight']
    base = params.get('base_prob', 0.0)
    combined = (params['score_weight'] * p_score + params['moves_weight'] * p_moves) / w_sum
    return base + (1.0 - base) * combined


def main():
    print("=== Heuristic Bot Tuning (MSE Optimization) ===\n")
    
    # Load user ratings
    user_ratings = load_user_ratings()
    print(f"Loaded {len(user_ratings)} user profile ratings\n")
    
    # Load human attempts
    attempts_df = load_human_attempts()
    print(f"Loaded {len(attempts_df)} attempts\n")
    
    # Add ratings
    attempts_df['rating'] = attempts_df['profile_id'].map(user_ratings)
    attempts_df = attempts_df.dropna(subset=['rating'])
    print(f"Attempts with ratings: {len(attempts_df)}\n")
    
    # Filter to weak humans only
    weak_attempts = attempts_df[attempts_df['rating'] < WEAK_HUMAN_MAX_RATING]
    weak_attempts = weak_attempts[weak_attempts['rating'] > WEAK_HUMAN_MIN_RATING]
    print(f"Weak human attempts ({WEAK_HUMAN_MIN_RATING} < rating < {WEAK_HUMAN_MAX_RATING}): {len(weak_attempts)}\n")
    
    # Calculate success rate per tsumego for weak humans
    success_rates = weak_attempts.groupby('tsumego_id')['success'].agg(['mean', 'count'])
    success_rates = success_rates.rename(columns={'mean': 'weak_human_rate', 'count': 'n_attempts'})
    print(f"Tsumegos with weak human attempts: {len(success_rates)}\n")
    
    # Load/extract features
    features_dict = load_or_extract_features(success_rates.index.tolist())
    print(f"Features available for {len(features_dict)} tsumegos\n")
    
    # Build analysis dataframe
    data = []
    for tsumego_id, row in success_rates.iterrows():
        if tsumego_id not in features_dict:
            continue
        data.append({
            'tsumego_id': tsumego_id,
            'weak_human_rate': row['weak_human_rate'],
            'n_attempts': row['n_attempts'],
            'moves_remaining': features_dict[tsumego_id]['moves_remaining'],
            'score_diff': features_dict[tsumego_id]['score_diff']
        })
    
    df = pd.DataFrame(data)
    print(f"Analysis dataset: {len(df)} tsumegos\n")
    
    # Feature stats
    print("=== Feature Statistics ===")
    print(f"Moves remaining: mean={df['moves_remaining'].mean():.1f}, median={df['moves_remaining'].median():.0f}")
    print(f"Score diff: mean={df['score_diff'].mean():.1f}, median={df['score_diff'].median():.1f}")
    print(f"Weak human rate: mean={df['weak_human_rate'].mean():.3f}")
    
    # Vectorized calculation for speed
    moves_arr = df['moves_remaining'].values
    score_arr = df['score_diff'].values
    target_arr = df['weak_human_rate'].values
    
    def compute_mse(params_array):
        params = {
            'score_decay': max(0.1, params_array[0]),
            'moves_decay': max(0.1, params_array[1]),
            'score_weight': max(0.01, params_array[2]),
            'moves_weight': max(0.01, params_array[3]),
            'base_prob': max(0.0, min(0.9, params_array[4])) if len(params_array) > 4 else 0.0
        }
        
        p_score = 1.0 - np.exp(-score_arr / params['score_decay'])
        p_moves = np.exp(-moves_arr / params['moves_decay'])
        
        w_sum = params['score_weight'] + params['moves_weight']
        combined = (params['score_weight'] * p_score + params['moves_weight'] * p_moves) / w_sum
        pred = params['base_prob'] + (1.0 - params['base_prob']) * combined
        
        mse = np.mean((pred - target_arr) ** 2)
        return mse
    
    # Current parameters
    current_params = [3.0, 10.0, 1.0, 1.0, 0.0]
    current_mse = compute_mse(current_params)
    print(f"\n=== Current Parameters ===")
    print(f"score_decay=3.0, moves_decay=10.0, weights=1:1")
    print(f"MSE: {current_mse:.6f}, RMSE: {np.sqrt(current_mse):.4f}")
    
    # Grid search
    print("\n=== Grid Search ===")
    best_mse = float('inf')
    best_params = None
    
    score_decays = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    moves_decays = [2.0, 5.0, 10.0, 20.0, 50.0]
    weights = [0.1, 0.5, 1.0, 2.0, 5.0]
    base_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.5, 0.52, 0.55, 0.6, 0.7]
    
    for sd in tqdm(score_decays, desc="Grid search"):
        for md in moves_decays:
            for sw in weights:
                for mw in weights:
                    for bp in base_probs:
                        mse = compute_mse([sd, md, sw, mw, bp])
                        if mse < best_mse:
                            best_mse = mse
                            best_params = [sd, md, sw, mw, bp]
    
    print(f"Best grid: score_decay={best_params[0]}, moves_decay={best_params[1]}")
    print(f"           score_weight={best_params[2]}, moves_weight={best_params[3]}, base_prob={best_params[4]}")
    print(f"MSE: {best_mse:.6f}, RMSE: {np.sqrt(best_mse):.4f}")
    
    # Fine-tune
    result = minimize(compute_mse, best_params, method='Nelder-Mead', options={'maxiter': 2000})
    
    final = {
        'score_decay': max(0.1, result.x[0]),
        'moves_decay': max(0.1, result.x[1]),
        'score_weight': max(0.01, result.x[2]),
        'moves_weight': max(0.01, result.x[3]),
        'base_prob': max(0.0, min(0.9, result.x[4]))
    }
    
    print(f"\n=== Optimized Parameters ===")
    print(f"score_decay = {final['score_decay']:.4f}")
    print(f"moves_decay = {final['moves_decay']:.4f}")
    print(f"score_weight = {final['score_weight']:.4f}")
    print(f"moves_weight = {final['moves_weight']:.4f}")
    print(f"base_prob = {final['base_prob']:.4f}")
    print(f"MSE: {result.fun:.6f}, RMSE: {np.sqrt(result.fun):.4f}")
    
    # Calculate predictions with optimized params
    p_score = 1.0 - np.exp(-score_arr / final['score_decay'])
    p_moves = np.exp(-moves_arr / final['moves_decay'])
    w_sum = final['score_weight'] + final['moves_weight']
    combined = (final['score_weight'] * p_score + final['moves_weight'] * p_moves) / w_sum
    df['heuristic_pred'] = final['base_prob'] + (1.0 - final['base_prob']) * combined
    
    # Show examples
    print("\n=== Sample Predictions ===")
    sample = df.nlargest(10, 'n_attempts')[
        ['tsumego_id', 'weak_human_rate', 'heuristic_pred', 'moves_remaining', 'score_diff', 'n_attempts']
    ]
    print(sample.to_string(index=False))
    
    # Save
    output_file = os.path.join(PROJECT_ROOT, 'auxiliary', 'heuristic_tuning_results.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    print(f"\n=== Suggested Code ===")
    print(f"""
class HeuristicBot:
    def __init__(self):
        self.score_decay = {final['score_decay']:.4f}
        self.moves_decay = {final['moves_decay']:.4f}
        self.score_weight = {final['score_weight']:.4f}
        self.moves_weight = {final['moves_weight']:.4f}
        self.base_prob = {final['base_prob']:.4f}
""")


if __name__ == '__main__':
    main()
