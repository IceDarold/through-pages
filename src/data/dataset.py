import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json

class InteractionDataset(Dataset):
    def __init__(self, sequences_path, embeddings_path, features_path, max_len=50):
        self.max_len = max_len
        
        print(f"Loading item data from {features_path} and {embeddings_path}...")
        # 1. Load item features for format_id and map them
        items_df = pd.read_parquet(features_path)
        self.item_meta = items_df.set_index('edition_id')[['format_id']].to_dict('index')
        
        # 2. Load embeddings and convert to a faster lookup structure
        emb_df = pd.read_parquet(embeddings_path)
        # We'll store embeddings in a large float32 array and map edition_id to its index
        self.edition_ids = emb_df['edition_id'].values
        self.id_to_idx = {eid: i for i, eid in enumerate(self.edition_ids)}
        
        # Drop edition_id to keep only raw embedding values
        self.embeddings = emb_df.drop(columns=['edition_id']).values.astype('float32')
        self.emb_dim = self.embeddings.shape[1]
        
        # 3. Load sequences
        print(f"Loading sequences from {sequences_path}...")
        self.sequences = []
        with open(sequences_path, 'r') as f:
            for line in f:
                self.sequences.append(json.loads(line))
        
        print(f"Dataset initialized with {len(self.sequences)} users.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        history = seq_data['history']
        
        # For training: we use all items except the last one as history,
        # and the last item as the target to predict.
        if len(history) < 2:
            # Fallback for very short validation sequences
            history_src = history
            target_item = history[-1]
        else:
            history_src = history[:-1]
            target_item = history[-1]
            
        # Truncate if history is too long
        history_src = history_src[-self.max_len:]
        
        seq_len = len(history_src)
        
        # Prepare buffers
        hist_embs = np.zeros((self.max_len, self.emb_dim), dtype='float32')
        hist_relevance = np.zeros(self.max_len, dtype='int64')
        hist_formats = np.zeros(self.max_len, dtype='int64')
        mask = np.zeros(self.max_len, dtype='bool')
        
        # Fill buffers
        for i, item in enumerate(history_src):
            eid = item['edition_id']
            if eid in self.id_to_idx:
                idx_emb = self.id_to_idx[eid]
                hist_embs[i] = self.embeddings[idx_emb]
                hist_relevance[i] = item['relevance'] # 1-3
                hist_formats[i] = self.item_meta.get(eid, {'format_id': 0})['format_id']
                mask[i] = True
        
        # Prepare target
        target_eid = target_item['edition_id']
        if target_eid in self.id_to_idx:
            target_emb = self.embeddings[self.id_to_idx[target_eid]]
        else:
            # Fallback for items missing in embeddings (unlikely)
            target_emb = np.zeros(self.emb_dim, dtype='float32')
            
        return {
            'hist_embs': torch.from_numpy(hist_embs),
            'hist_relevance': torch.from_numpy(hist_relevance),
            'hist_formats': torch.from_numpy(hist_formats),
            'mask': torch.from_numpy(mask),
            'target_emb': torch.from_numpy(target_emb)
        }

if __name__ == "__main__":
    # Test loading a sample batch
    # Need to make sure files exist or handle error
    try:
        ds = InteractionDataset(
            'experiments/data_v1/train_sequences.jsonl',
            'experiments/data_v1/item_embeddings.parquet',
            'experiments/data_v1/item_features_enriched.parquet',
            max_len=50
        )
        sample = ds[0]
        print("\nSample batch check:")
        print(f"History Embs shape: {sample['hist_embs'].shape}")
        print(f"Mask sum: {sample['mask'].sum()}")
        print(f"Target Emb shape: {sample['target_emb'].shape}")
    except FileNotFoundError as e:
        print(f"Skipping test run: {e}")
