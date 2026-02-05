import argparse
import os
from datetime import timedelta

import numpy as np
import pandas as pd

def preprocess_interactions(df):
    """
    Apply unified schema and map event types.
    """
    # Parse timestamps
    df['event_ts'] = pd.to_datetime(df['event_ts'])
    
    # Map event types to descriptive relevance (Initial guess)
    # event_type 1 -> Start/Wishlist (Relevance 1)
    # event_type 2 -> Finish (Relevance 2)
    # If rating exists and high -> (Relevance 3)
    
    df['relevance'] = 1
    df.loc[df['event_type'] == 2, 'relevance'] = 2
    # Boost for high ratings (assuming scale 1-5)
    df.loc[df['rating'] >= 4, 'relevance'] = 3
    
    # Sort chronologically for sequence modeling
    df = df.sort_values(['user_id', 'event_ts'])
    
    return df

def create_time_split(df, val_days=14):
    """
    Create a chronological split based on the last interactions.
    """
    max_date = df['event_ts'].max()
    split_date = max_date - timedelta(days=val_days)
    
    train_df = df[df['event_ts'] < split_date]
    val_df = df[df['event_ts'] >= split_date]
    
    print(f"Split Date: {split_date}")
    print(f"Train events: {len(train_df)}")
    print(f"Val events: {len(val_df)}")
    
    return train_df, val_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/kaggle/input/though-pages/data")
    ap.add_argument("--submit-dir", default="/kaggle/input/though-pages/submit")
    ap.add_argument("--out-dir", default="/kaggle/working/through-pages/experiments/data_v1")
    ap.add_argument("--val-days", type=int, default=14)
    args = ap.parse_args()

    print("Loading data...")
    interactions = pd.read_csv(os.path.join(args.data_dir, "interactions.csv"))
    editions = pd.read_csv(os.path.join(args.data_dir, "editions.csv"))
    targets_users = pd.read_csv(os.path.join(args.submit_dir, "targets.csv"))["user_id"].unique()
    
    print("Preprocessing interactions...")
    df = preprocess_interactions(interactions)
    
    # Save the whole cleaned interactions for features
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_parquet(os.path.join(args.out_dir, "clean_interactions.parquet"), index=False)
    
    print("Creating chronological split...")
    train_df, val_df = create_time_split(df, val_days=args.val_days)
    
    train_df.to_parquet(os.path.join(args.out_dir, "train_interactions.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.out_dir, "val_interactions.parquet"), index=False)
    
    print("Phase 0 Preprocessing Complete.")

if __name__ == "__main__":
    main()
