import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import argparse

def train_lgbm(exp_dir):
    print("Loading features for training...")
    # This assumes we ran feature_factory on the train_reranker_pairs.csv
    train_path = os.path.join(exp_dir, "features_train.parquet")
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Did you run feature_factory in train mode?")
        return
        
    df = pd.read_parquet(train_path)
    
    # Define features to use
    drop_cols = ['user_id', 'edition_id', 'label', 'author_id', 'book_id', 'genres', 'title']
    features = [c for c in df.columns if c not in drop_cols]
    
    print(f"Training on {len(features)} features: {features}")
    
    # Sort by user_id for Grouping in LGBMRanker
    df = df.sort_values('user_id')
    groups = df.groupby('user_id').size().values
    
    X = df[features]
    y = df['label']
    
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        importance_type='gain',
        verbose=-1
    )
    
    model.fit(X, y, group=groups)
    
    model_path = os.path.join(exp_dir, "lgbm_reranker.txt")
    model.booster_.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Print importance
    importance = pd.DataFrame({'feature': features, 'gain': model.feature_importances_})
    print("\nTop Features by Gain:")
    print(importance.sort_values('gain', ascending=False).head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="/kaggle/working/through-pages/experiments/data_v1")
    args = parser.parse_args()
    train_lgbm(args.exp_dir)
