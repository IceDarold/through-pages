import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

def generate_submission(features_path, output_path, top_n=20):
    print(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)
    
    # In a real SOTA scenario, we would load the trained LightGBM model here.
    # For the initial submission, we'll use a strong baseline: Multi-Interest Similarity + Series Logic.
    
    print("Ranking candidates...")
    
    # Example heuristic: combine multi-interest score with series bonus and author loyalty
    # Note: These column names should match what Feature Factory produces
    
    # If we have model scores, we use them. Otherwise, we calculate a baseline score.
    if 'model_score' not in df.columns:
        # Heuristic Score: 
        # +1.0 for Volume Diff == 1 (Direct sequel)
        # +0.5 for Author Match
        # -100 for Book Already Seen (already handled by competition usually, but safe to have)
        df['score'] = (
            (df['ui_volume_diff'] == 1).astype(float) * 1.5 + 
            (df['ui_author_count'] > 0).astype(float) * 0.5 +
            (df['ui_book_seen'] == 1).astype(float) * -100.0
        )
        # Add a tiny bit of popularity to break ties
        if 'i_pop_total' in df.columns:
            df['score'] += df['i_pop_total'] / (df['i_pop_total'].max() + 1) * 0.1
            
    else:
        df['score'] = df['model_score']

    # Rank within each user
    df = df.sort_values(['user_id', 'score'], ascending=[True, False])
    df['rank'] = df.groupby('user_id').cumcount() + 1
    
    # Filter top 20
    submission = df[df['rank'] <= top_n][['user_id', 'edition_id', 'rank']]
    
    print(f"Saving submission to {output_path}...")
    submission.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="experiments/data_v1/features_infer.parquet")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    
    generate_submission(args.features, args.output)
