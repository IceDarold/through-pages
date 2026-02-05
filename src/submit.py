import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

def generate_submission(features_path, output_path, model_path=None, top_n=20):
    print(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading LightGBM model from {model_path}...")
        model = lgb.Booster(model_file=model_path)
        
        # Select features (must match training)
        drop_cols = ['user_id', 'edition_id', 'label', 'author_id', 'book_id', 'genres', 'title']
        features = [c for c in df.columns if c not in drop_cols]
        
        print("Calculating model scores...")
        df['score'] = model.predict(df[features])
    else:
        print("Model not found. Falling back to heuristic baseline.")
        # Heuristic Score: 
        # +1.5 for Volume Diff == 1 (Direct sequel)
        # +0.5 for Author Match
        # + Similarity score from vectors
        df['score'] = (
            (df['ui_volume_diff'] == 1).astype(float) * 1.5 + 
            (df['ui_author_count'] > 0).astype(float) * 0.5 +
            df.get('ui_max_interest_sim', 0.0) * 1.0 +
            (df['ui_book_seen'] == 1).astype(float) * -100.0
        )

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
    parser.add_argument("--model-path", default="experiments/data_v1/lgbm_reranker.txt")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    
    generate_submission(args.features, args.output, model_path=args.model_path)
