import pandas as pd
import numpy as np
import os
import argparse
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(data_dir, exp_dir):
    print("Loading data for feature engineering...")
    interactions = pd.read_parquet(os.path.join(exp_dir, "clean_interactions.parquet"))
    items = pd.read_parquet(os.path.join(exp_dir, "item_features_enriched.parquet"))
    
    # Load embeddings if available (for similarity features)
    embeddings = None
    if os.path.exists(os.path.join(exp_dir, "item_embeddings.parquet")):
        embeddings = pd.read_parquet(os.path.join(exp_dir, "item_embeddings.parquet"))
        
    return interactions, items, embeddings

def build_user_features(interactions):
    print("Building Global User Features...")
    user_stats = interactions.groupby('user_id').agg({
        'edition_id': 'count',
        'event_type': lambda x: (x == 2).mean(), # Finish rate
        'rating': ['mean', 'std', 'max']
    }).reset_index()
    
    user_stats.columns = [
        'user_id', 'u_total_interactions', 'u_finish_rate', 
        'u_avg_rating', 'u_std_rating', 'u_max_rating'
    ]
    
    # Add Recency: how many events in the last available week of the training set
    max_ts = interactions['event_ts'].max()
    interactions['days_from_end'] = (max_ts - interactions['event_ts']).dt.days
    
    last_week_counts = interactions[interactions['days_from_end'] <= 7].groupby('user_id').size().reset_index(name='u_last_week_activity')
    user_stats = user_stats.merge(last_week_counts, on='user_id', how='left').fillna(0)
    
    return user_stats

def build_item_features(interactions, items):
    print("Building Global Item Features...")
    item_stats = interactions.groupby('edition_id').agg({
        'user_id': 'count',
        'event_type': lambda x: (x == 2).mean(), # Sticky rate
        'rating': ['mean', 'std']
    }).reset_index()
    
    item_stats.columns = [
        'edition_id', 'i_pop_total', 'i_sticky_rate', 
        'i_avg_rating', 'i_std_rating'
    ]
    
    # Author stats
    item_with_author = items[['edition_id', 'author_id']].merge(item_stats, on='edition_id', how='left')
    author_stats = item_with_author.groupby('author_id').agg({
        'i_pop_total': 'sum',
        'i_avg_rating': 'mean'
    }).reset_index()
    author_stats.columns = ['author_id', 'a_total_pop', 'a_avg_rating']
    
    items = items.merge(item_stats, on='edition_id', how='left')
    items = items.merge(author_stats, on='author_id', how='left')
    
    return items.fillna(0)

def extract_volume(title):
    # Search for "том 1", "часть 2", "книга 3" etc.
    match = re.search(r'(?:том|часть|книга|volume|book|#)\s*(\d+)', title.lower())
    if match:
        return int(match.group(1))
    return 1 # Default to 1 if not specified

def generate_interaction_features(pairs, interactions, items, user_interests=None, item_embs=None):
    """
    pairs: DataFrame with [user_id, edition_id]
    """
    print("Generating User-Item Interaction Features...")
    
    # 0. Series/Volume Extraction
    items['volume'] = items['title'].apply(extract_volume)
    
    # 1. Author Affinity: How many times user read this author
    user_author_history = interactions.merge(items[['edition_id', 'author_id']], on='edition_id')
    user_author_counts = user_author_history.groupby(['user_id', 'author_id']).size().reset_index(name='ui_author_count')
    
    pairs = pairs.merge(items[['edition_id', 'author_id', 'genres', 'format_id', 'volume', 'title']], on='edition_id', how='left')
    pairs = pairs.merge(user_author_counts, on=['user_id', 'author_id'], how='left').fillna(0)
    
    # 2. Sequential Logic: Distance from last volumes of authors
    print("Calculating Sequential Volume Distance...")
    # Find max volume read per author by user
    user_last_volumes = user_author_history.merge(items[['edition_id', 'volume']], on='edition_id')
    user_last_volumes = user_last_volumes.groupby(['user_id', 'author_id'])['volume'].max().reset_index()
    user_last_volumes.columns = ['user_id', 'author_id', 'u_max_volume_author']
    
    pairs = pairs.merge(user_last_volumes, on=['user_id', 'author_id'], how='left').fillna(0)
    pairs['ui_volume_diff'] = pairs['volume'] - pairs['u_max_volume_author']
    
    # 3. Genre Affinity (Dice Coefficient / Overlap)
    print("Calculating Genre Affinity...")
    user_genres = interactions.merge(items[['edition_id', 'genres']], on='edition_id')
    # This is a bit slow, but let's do a simplified version
    # Expand genres to one-hot or use Jaccard
    
    # Vectorizing genres
    # We can pre-calculate user genre preference vector
    # [TO BE IMPLEMENTED IN PRODUCTION SCALE]

    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="experiments/data_v1")
    parser.add_argument("--mode", choices=['train', 'infer'], default='infer')
    args = parser.parse_args()
    
    interactions, items, embeddings = load_data('data', args.exp_dir)
    
    user_stats = build_user_features(interactions)
    item_stats = build_item_features(interactions, items)
    
    if args.mode == 'infer':
        print("Processing Candidates for Inference...")
        pairs = pd.read_csv('submit/candidates.csv')
    else:
        # For training, we need to generate pairs (target + negatives)
        # This will be handled in a separate step to keepFE clean
        print("Training mode FE not implemented here yet.")
        return

    # Merge basic features
    df = pairs.merge(user_stats, on='user_id', how='left')
    df = df.merge(item_stats.drop(columns=['author_id', 'genres', 'format_id', 'book_id'], errors='ignore'), on='edition_id', how='left')
    
    # Generate complex interaction features
    df = generate_interaction_features(df, interactions, items)
    
    output_path = os.path.join(args.exp_dir, f"features_{args.mode}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Features saved to {output_path}")

if __name__ == "__main__":
    main()
