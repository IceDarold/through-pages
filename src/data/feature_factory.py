import pandas as pd
import numpy as np
import os
import argparse
import re
import tqdm
import torch
import torch.nn.functional as F
import sys
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src to path for absolute imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.multi_interest import MultiInterestEncoder

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

def generate_interaction_features(pairs, interactions, items, exp_dir):
    """
    pairs: DataFrame with [user_id, edition_id]
    """
    print("Generating User-Item Interaction Features...")
    
    # 0. Series/Volume Extraction
    if 'volume' not in items.columns:
        items['volume'] = items['title'].apply(extract_volume)
    
    # 1. Author Affinity: How many times user read this author
    user_author_history = interactions.merge(items[['edition_id', 'author_id']], on='edition_id')
    user_author_counts = user_author_history.groupby(['user_id', 'author_id']).size().reset_index(name='ui_author_count')
    
    pairs = pairs.merge(items[['edition_id', 'author_id', 'genres', 'format_id', 'volume', 'title', 'book_id']], on='edition_id', how='left')
    pairs = pairs.merge(user_author_counts, on=['user_id', 'author_id'], how='left').fillna(0)
    
    # 2. Sequential Logic: Distance from last volumes of authors
    print("Calculating Sequential Volume Distance...")
    user_last_volumes = user_author_history.merge(items[['edition_id', 'volume']], on='edition_id')
    user_last_volumes = user_last_volumes.groupby(['user_id', 'author_id'])['volume'].max().reset_index()
    user_last_volumes.columns = ['user_id', 'author_id', 'u_max_volume_author']
    
    pairs = pairs.merge(user_last_volumes, on=['user_id', 'author_id'], how='left').fillna(0)
    pairs['ui_volume_diff'] = pairs['volume'] - pairs['u_max_volume_author']
    
    # 3. Item Seen Check
    user_book_history = interactions.merge(items[['edition_id', 'book_id']], on='edition_id')
    user_book_seen = user_book_history[['user_id', 'book_id']].drop_duplicates()
    user_book_seen['ui_book_seen'] = 1
    pairs = pairs.merge(user_book_seen, on=['user_id', 'book_id'], how='left').fillna(0)

    interests_path = os.path.join(exp_dir, "user_interests.npy")
    embs_path = os.path.join(exp_dir, "item_embeddings.parquet")
    model_path = os.path.join(exp_dir.replace("data_v1", "models_v1"), "best_multi_interest.pth")
    
    # Check alternate Kaggle location
    if not os.path.exists(interests_path):
        interests_path = "/kaggle/input/though-pages/user_interests.npy"
    if not os.path.exists(embs_path):
        embs_path = "/kaggle/input/though-pages/item_embeddings.parquet"
    if not os.path.exists(model_path):
        model_path = "/kaggle/working/through-pages/experiments/models_v1/best_multi_interest.pth"

    if os.path.exists(interests_path) and os.path.exists(embs_path) and os.path.exists(model_path):
        print("Calculating Multi-Interest Similarity Scores...")
        user_data = np.load(interests_path, allow_pickle=True).item()
        user_id_to_idx = {uid: i for i, uid in enumerate(user_data['user_ids'])}
        user_interests = user_data['interests'] # [N, 6, 256]
        
        emb_df = pd.read_parquet(embs_path)
        item_id_to_idx = {eid: i for i, eid in enumerate(emb_df['edition_id'])}
        item_embs_raw = emb_df.drop(columns=['edition_id']).values # [M, 768]
        
        # Load model to get the projector
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiInterestEncoder(item_emb_dim=768, d_model=256).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Project all item embeddings to 256-D
        print("Projecting item embeddings to 256-D...")
        with torch.no_grad():
            item_embs_tensor = torch.from_numpy(item_embs_raw).float().to(device)
            # Project in chunks to avoid OOM
            item_embs_proj = []
            chunk_size = 10000
            for k in range(0, len(item_embs_tensor), chunk_size):
                chunk = item_embs_tensor[k:k+chunk_size]
                proj = model.item_projector(chunk)
                proj = F.normalize(proj, p=2, dim=-1)
                item_embs_proj.append(proj.cpu().numpy())
            item_embs = np.concatenate(item_embs_proj, axis=0) # [M, 256]

        # Result buffers
        max_sims = []
        
        for idx_row, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Sim Calculation"):
            uid, eid = row['user_id'], row['edition_id']
            if uid in user_id_to_idx and eid in item_id_to_idx:
                u_vecs = user_interests[user_id_to_idx[uid]] # [6, 256]
                i_vec = item_embs[item_id_to_idx[eid]]      # [256]
                # Cosine similarity
                sims = np.dot(u_vecs, i_vec)
                max_sims.append(np.max(sims))
            else:
                max_sims.append(0.0)
        
        pairs['ui_max_interest_sim'] = max_sims

    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="/kaggle/working/through-pages/experiments/data_v1")
    parser.add_argument("--data-dir", default="/kaggle/input/though-pages")
    parser.add_argument("--mode", choices=['train', 'infer'], default='infer')
    args = parser.parse_args()
    
    interactions, items, embeddings = load_data(args.data_dir, args.exp_dir)
    
    user_stats = build_user_features(interactions)
    item_stats = build_item_features(interactions, items)
    
    if args.mode == 'infer':
        print("Processing Candidates for Inference...")
        pairs_path = os.path.join(args.data_dir, "candidates.csv")
        if not os.path.exists(pairs_path):
            pairs_path = "submit/candidates.csv" # local fallback
        pairs = pd.read_csv(pairs_path)
    else:
        print("Processing Pairs for Training...")
        pairs_path = os.path.join(args.exp_dir, "train_reranker_pairs.csv")
        if not os.path.exists(pairs_path):
            print("Please use make_reranker_train_data.py first.")
            return
        pairs = pd.read_csv(pairs_path)

    # Initial merge with stats to create 'df'
    print("Merging global stats...")
    df = pairs.merge(user_stats, on='user_id', how='left')
    df = df.merge(item_stats.drop(columns=['author_id', 'genres', 'format_id', 'book_id', 'volume', 'title'], errors='ignore'), on='edition_id', how='left')
    
    # Interaction Features (Volume, Author affinity, Vector Sim)
    df = generate_interaction_features(df, interactions, items, args.exp_dir)
    
    # FINAL ROBUST CLEANUP: Keep only numeric features + IDs
    # This prevents ANY PyArrow schema issues with titles, authors, descriptions, etc.
    cols_to_keep = ['user_id', 'edition_id', 'label']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    final_cols = list(set(cols_to_keep + numeric_cols))
    final_cols = [c for c in final_cols if c in df.columns]
    
    df = df[final_cols].copy()
    
    # Optional: Fill NaNs with 0 to be safe for LightGBM
    df = df.fillna(0)
    
    # Ensure all features are float32 to avoid Arrow type complexity
    for col in df.columns:
        if col not in ['user_id', 'edition_id', 'label']:
            df[col] = df[col].astype(np.float32)
    
    output_path = os.path.join(args.exp_dir, f"features_{args.mode}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Features saved to {output_path}")
    print(f"Features saved to {output_path}")

if __name__ == "__main__":
    main()
