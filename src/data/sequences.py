import pandas as pd
import numpy as np
import os
import json

def generate_sequences(interactions_path, output_path, min_len=2):
    print(f"Loading interactions from {interactions_path}...")
    df = pd.read_parquet(interactions_path)
    
    print("Generating user sequences...")
    # Group by user and collect histories
    # We store them as list of dicts for flexibility
    
    sequences = []
    
    # Using groupby is slow for large datasets, but 200k interactions is fine.
    for user_id, group in df.groupby('user_id'):
        if len(group) < min_len:
            continue
            
        # Sort is already handled in preprocess.py, but just to be safe
        group = group.sort_values('event_ts')
        
        history = []
        for _, row in group.iterrows():
            history.append({
                'edition_id': int(row['edition_id']),
                'relevance': int(row['relevance']),
                'ts': row['event_ts'].isoformat()
            })
            
        sequences.append({
            'user_id': int(user_id),
            'history': history
        })
    
    print(f"Generated {len(sequences)} sequences.")
    
    # Save as JSONL (better for large sequence datasets)
    with open(output_path, 'w') as f:
        for seq in sequences:
            f.write(json.dumps(seq) + '\n')
            
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_sequences(
        'experiments/data_v1/train_interactions.parquet',
        'experiments/data_v1/train_sequences.jsonl'
    )
    generate_sequences(
        'experiments/data_v1/val_interactions.parquet',
        'experiments/data_v1/val_sequences.jsonl',
        min_len=1 # Validation might have shorter segments after split
    )
