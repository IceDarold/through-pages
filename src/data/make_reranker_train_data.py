import pandas as pd
import numpy as np
import os
import argparse
import random

def make_train_pairs(exp_dir, n_negatives=100):
    print("Generating labeled pairs for Reranker training...")
    
    # Load ground truth from validation sequences
    val_sequences = []
    import json
    with open(os.path.join(exp_dir, "val_sequences.jsonl"), 'r') as f:
        for line in f:
            val_sequences.append(json.loads(line))
            
    # Load all items for negative sampling
    items = pd.read_parquet(os.path.join(exp_dir, "item_features_enriched.parquet"))
    all_edition_ids = items['edition_id'].unique()
    
    labeled_pairs = []
    
    for seq in val_sequences:
        user_id = seq['user_id']
        history = seq['history']
        
        if len(history) < 2:
            continue
            
        positive_id = history[-1]['edition_id']
        
        # Positive sample
        labeled_pairs.append({'user_id': user_id, 'edition_id': positive_id, 'label': 1})
        
        # Negative samples
        history_ids = set(h['edition_id'] for h in history)
        negatives = []
        while len(negatives) < n_negatives:
            cand = random.choice(all_edition_ids)
            if cand not in history_ids and cand != positive_id:
                negatives.append(cand)
                labeled_pairs.append({'user_id': user_id, 'edition_id': cand, 'label': 0})
                
    df = pd.DataFrame(labeled_pairs)
    output_path = os.path.join(exp_dir, "train_reranker_pairs.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} pairs for {len(val_sequences)} users. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="/kaggle/working/through-pages/experiments/data_v1")
    args = parser.parse_args()
    make_train_pairs(args.exp_dir)
