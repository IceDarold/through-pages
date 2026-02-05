import pandas as pd
import numpy as np
import os
import argparse
import random
import json
import torch
import sys
from tqdm import tqdm

# Add src to path for absolute imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.multi_interest import MultiInterestEncoder

def make_train_pairs(exp_dir, n_negatives=100):
    print("Generating HARD labeled pairs for Reranker training...")
    
    # Paths
    val_path = os.path.join(exp_dir, "val_sequences.jsonl")
    interests_path = os.path.join(exp_dir, "user_interests.npy")
    embs_path = os.path.join(exp_dir, "item_embeddings.parquet")
    model_path = os.path.join(exp_dir.replace("data_v1", "models_v1"), "best_multi_interest.pth")
    
    if not all(os.path.exists(p) for p in [val_path, interests_path, embs_path]):
        print("Required files missing. Check user_interests.npy and item_embeddings.parquet")
        return

    # Load data
    val_sequences = []
    with open(val_path, 'r') as f:
        for line in f:
            val_sequences.append(json.loads(line))
            
    items_df = pd.read_parquet(os.path.join(exp_dir, "item_features_enriched.parquet"))
    all_edition_ids = items_df['edition_id'].values
    
    user_data = np.load(interests_path, allow_pickle=True).item()
    user_id_to_idx = {uid: i for i, uid in enumerate(user_data['user_ids'])}
    user_interests = user_data['interests'] # [N, 6, 256]
    
    emb_df = pd.read_parquet(embs_path)
    item_id_to_idx = {eid: i for i, eid in enumerate(emb_df['edition_id'])}
    item_embs_raw = emb_df.drop(columns=['edition_id']).values

    # Project item embeddings to 256-D (same as feature_factory)
    from models.multi_interest import MultiInterestEncoder
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiInterestEncoder(item_emb_dim=768, d_model=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        item_embs = model.item_projector(torch.from_numpy(item_embs_raw).float().to(device))
        item_embs = F.normalize(item_embs, p=2, dim=-1).cpu().numpy()

    labeled_pairs = []
    
    # We'll sample 100 negatives for each user: 
    # 50 random + 50 "hard" (most similar but not the one)
    for seq in tqdm(val_sequences, desc="Sampling Pairs"):
        user_id = seq['user_id']
        history = seq['history']
        if len(history) < 2 or user_id not in user_id_to_idx:
            continue
            
        positive_id = history[-1]['edition_id']
        history_ids = set(h['edition_id'] for h in history)
        
        # Positive sample
        labeled_pairs.append({'user_id': user_id, 'edition_id': positive_id, 'label': 1})
        
        # Get user interest vectors [6, 256]
        u_vecs = user_interests[user_id_to_idx[user_id]]
        
        # To find hard negatives efficiently, we'll sample a larger pool of items (e.g. 5000)
        # and pick the most similar ones from that pool
        pool_indices = np.random.choice(len(item_embs), 2000, replace=False)
        pool_embs = item_embs[pool_indices]
        
        # Calculate max similarity for pool [2000, 6] -> [2000]
        sims = np.dot(pool_embs, u_vecs.T)
        max_sims = np.max(sims, axis=1)
        
        # Sort pool by similarity
        sorted_indices = np.argsort(-max_sims)
        
        neg_count = 0
        for idx in sorted_indices:
            eid = emb_df.iloc[pool_indices[idx]]['edition_id']
            if eid not in history_ids and eid != positive_id:
                labeled_pairs.append({'user_id': user_id, 'edition_id': eid, 'label': 0})
                neg_count += 1
                if neg_count >= n_negatives:
                    break
                
    df = pd.DataFrame(labeled_pairs)
    output_path = os.path.join(exp_dir, "train_reranker_pairs.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} pairs for {len(val_sequences)} users. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="/kaggle/working/through-pages/experiments/data_v1")
    args = parser.parse_args()
    make_train_pairs(args.exp_dir)
