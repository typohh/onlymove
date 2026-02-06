import os
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def analyze_attempts(sql_path):
    print(f"Reading {sql_path}...")
    
    # regex from pipeline/rank_tsumegos_and_bots.py
    insert_pattern = re.compile(r"INSERT INTO [\"`]?tsumego_reports[\"`]? VALUES\(\d+,'([^']+)','([^']+)',([01]),'([^']+)'\);")
    
    records = []
    with open(sql_path, 'r') as f:
        for line in tqdm(f, desc="Parsing SQL"):
            match = insert_pattern.match(line)
            if match:
                user_id, problem_id, success, timestamp_str = match.groups()
                records.append({
                    'user_id': user_id,
                    'problem_id': problem_id,
                    'success': int(success),
                    'timestamp': timestamp_str
                })
    
    if not records:
        print("No records found.")
        return

    df = pd.DataFrame(records)
    df['dt'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['dt'].dt.strftime('%Y-%m')

    print("\n--- 1. Users by Completed Tsumegos (Bucketed) ---")
    completed_per_user = df[df['success'] == 1].groupby('user_id')['problem_id'].nunique()
    
    bins = [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
    labels = ['1-10', '11-50', '51-100', '101-500', '501-1000', '1001-5000', '5001+']
    
    buckets = pd.cut(completed_per_user, bins=bins, labels=labels)
    bucket_counts = buckets.value_counts().sort_index()
    print(bucket_counts)

    print("\n--- 2. Users Active per Month ---")
    users_per_month = df.groupby('month')['user_id'].nunique()
    print(users_per_month)

    print("\n--- 3. Active Users per Month (>= 30 attempts) ---")
    attempts_per_user_month = df.groupby(['month', 'user_id']).size().reset_index(name='count')
    active_users = attempts_per_user_month[attempts_per_user_month['count'] >= 30]
    active_users_per_month = active_users.groupby('month')['user_id'].nunique()
    print(active_users_per_month)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sql_file = os.path.join(base_dir, 'attempts', 'database.sql')
    if os.path.exists(sql_file):
        analyze_attempts(sql_file)
    else:
        print(f"Error: {sql_file} not found.")
