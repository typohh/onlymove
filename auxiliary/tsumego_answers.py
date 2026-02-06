#!/usr/bin/env python3
"""
Tsumego Answers Viewer

Takes a tsumego ID and shows:
- All humans who have answered it, with their result, rating, and rank
- All bots who have answered it, with their success rate, rating, and rank

User profiles are date-based (overlapping 2-month windows), so we find the
appropriate profile based on when the tsumego was answered.
"""

import os
import re
import sys
from datetime import datetime

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
from utils.katago_queries import get_rank_str


def load_human_attempts(tsumego_id):
    """Load all human attempts for a specific tsumego from database.sql."""
    sql_file = os.path.join(PROJECT_ROOT, 'attempts', 'database.sql')
    attempts = []
    
    # Regex to parse INSERT statements
    insert_pattern = re.compile(
        r"INSERT INTO [\"`]?tsumego_reports[\"`]? VALUES\(\d+,'([^']+)','([^']+)',([01]),'([^']+)'\);"
    )
    
    with open(sql_file, 'r') as f:
        for line in f:
            match = insert_pattern.match(line)
            if match:
                user_id, problem, success, timestamp_str = match.groups()
                if problem == tsumego_id:
                    try:
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        attempts.append({
                            'user_id': user_id,
                            'success': int(success),
                            'timestamp': dt
                        })
                    except ValueError:
                        continue
    
    return attempts


def load_bot_attempts(tsumego_id):
    """Load all bot attempts for a specific tsumego from bot txt files."""
    attempts_dir = os.path.join(PROJECT_ROOT, 'attempts')
    attempts = []
    
    for filename in os.listdir(attempts_dir):
        if not filename.endswith('.txt'):
            continue
        
        bot_name = os.path.splitext(filename)[0]
        filepath = os.path.join(attempts_dir, filename)
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    tid, success_rate = parts
                    if tid == tsumego_id:
                        attempts.append({
                            'bot_id': bot_name,
                            'success_rate': float(success_rate)
                        })
                        break  # Only one entry per tsumego per bot
    
    return attempts


def load_ratings():
    """
    Load user profiles/bot ratings and calculate the scaling parameters
    to convert raw ratings to normalized 0-1 scale for rank display.
    
    Uses the same anchor logic as package_for_pwa.py:
    - b18c384nbt-humanv0_8k anchors to 8k (index 12.5/29)
    - b18c384nbt-humanv0_3d anchors to 3d (index 22.5/29)
    """
    profiles_file = os.path.join(PROJECT_ROOT, 'ratings', 'user_profiles_final.csv')
    
    try:
        profiles_df = pd.read_csv(profiles_file)
        ratings_dict = dict(zip(profiles_df['bot_id'], profiles_df['rating']))
        
        # Calculate scaling parameters using anchor bots
        bot_8k_id = 'b18c384nbt-humanv0_8k'
        bot_3d_id = 'b18c384nbt-humanv0_3d'
        
        if bot_8k_id in ratings_dict and bot_3d_id in ratings_dict:
            bot_8k_rating = ratings_dict[bot_8k_id]
            bot_3d_rating = ratings_dict[bot_3d_id]
            
            # 8k is index 12 (0-indexed from 20k), 3d is index 22
            # Using 12.5/29 and 22.5/29 as the normalized positions
            f1 = 12.5 / 29  # 8k position
            f2 = 22.5 / 29  # 3d position
            
            # Linear scaling: normalized = a * raw + b
            a = (f2 - f1) / (bot_3d_rating - bot_8k_rating)
            b = f1 - a * bot_8k_rating
            
            return ratings_dict, (a, b)
        else:
            print("Warning: Anchor bots not found. Rank display may be inaccurate.")
            return ratings_dict, None
            
    except FileNotFoundError:
        print(f"Warning: {profiles_file} not found.")
        return {}, None


def rating_to_normalized(raw_rating, scale_params):
    """Convert a raw rating to a normalized 0-1 value for rank display."""
    if scale_params is None:
        return None
    a, b = scale_params
    normalized = a * raw_rating + b
    # Clamp to 0-1 range
    return max(0.0, min(1.0, normalized))


def get_user_profile_id(user_id, timestamp):
    """
    Generate the profile ID for a user at a given timestamp.
    
    Profiles are 2-month overlapping windows, so an attempt in month M
    is attributed to both the profile starting in month M and the profile
    starting in month M-1. We return the "current month" profile for display.
    """
    return f"{user_id[:8]}_{timestamp.year}-{timestamp.month:02d}"


def show_tsumego_answers(tsumego_id):
    """Display all players who answered a specific tsumego."""
    print(f"\n{'='*70}")
    print(f"Tsumego ID: {tsumego_id}")
    print(f"{'='*70}")
    
    # Load ratings and scaling parameters
    ratings, scale_params = load_ratings()
    
    # --- Human Attempts ---
    print(f"\n--- Human Attempts ---\n")
    human_attempts = load_human_attempts(tsumego_id)
    
    if not human_attempts:
        print("No human attempts found.")
    else:
        # Sort by timestamp
        human_attempts.sort(key=lambda x: x['timestamp'])
        
        print(f"{'User ID':<26} {'Result':<10} {'Profile ID':<20} {'Rating':>8} {'Rank':>6}")
        print("-" * 70)
        
        for attempt in human_attempts:
            user_id = attempt['user_id']
            success = "Correct" if attempt['success'] else "Wrong"
            profile_id = get_user_profile_id(user_id, attempt['timestamp'])
            
            if profile_id in ratings:
                raw_rating = ratings[profile_id]
                normalized = rating_to_normalized(raw_rating, scale_params)
                rank = get_rank_str(normalized) if normalized is not None else "N/A"
            else:
                # Try the previous month profile as fallback
                prev_month = attempt['timestamp'].month - 1
                prev_year = attempt['timestamp'].year
                if prev_month == 0:
                    prev_month = 12
                    prev_year -= 1
                alt_profile_id = f"{user_id[:8]}_{prev_year}-{prev_month:02d}"
                
                if alt_profile_id in ratings:
                    raw_rating = ratings[alt_profile_id]
                    normalized = rating_to_normalized(raw_rating, scale_params)
                    rank = get_rank_str(normalized) if normalized is not None else "N/A"
                    profile_id = alt_profile_id
                else:
                    raw_rating = None
                    rank = "N/A"
            
            rating_str = f"{raw_rating:.2f}" if raw_rating is not None else "N/A"
            print(f"{user_id:<26} {success:<10} {profile_id:<20} {rating_str:>8} {rank:>6}")
    
    # --- Bot Attempts ---
    print(f"\n--- Bot Attempts ---\n")
    bot_attempts = load_bot_attempts(tsumego_id)
    
    if not bot_attempts:
        print("No bot attempts found.")
    else:
        # Sort by success rate descending
        bot_attempts.sort(key=lambda x: x['success_rate'], reverse=True)
        
        print(f"{'Bot ID':<50} {'Success%':>10} {'Rating':>8} {'Rank':>6}")
        print("-" * 76)
        
        for attempt in bot_attempts:
            bot_id = attempt['bot_id']
            success_rate = attempt['success_rate'] * 100
            
            if bot_id in ratings:
                raw_rating = ratings[bot_id]
                normalized = rating_to_normalized(raw_rating, scale_params)
                rank = get_rank_str(normalized) if normalized is not None else "N/A"
            else:
                raw_rating = None
                rank = "N/A"
            
            rating_str = f"{raw_rating:.2f}" if raw_rating is not None else "N/A"
            print(f"{bot_id:<50} {success_rate:>9.1f}% {rating_str:>8} {rank:>6}")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python tsumego_answers.py <tsumego_id>")
        print("Example: python tsumego_answers.py e2843e18")
        sys.exit(1)
    
    tsumego_id = sys.argv[1]
    show_tsumego_answers(tsumego_id)


if __name__ == '__main__':
    main()
