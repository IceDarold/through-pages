import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json

class InteractionDataset(Dataset):
    def __init__(self, sequences_path, embeddings_path, features_path, max_len=50, mode='train'):
        self.max_len = max_len
        self.mode = mode
        
        print(f"Loading item data from {features_path} and {embeddings_path}...")
        items_df = pd.read_parquet(features_path)
        self.item_meta = items_df.set_index('edition_id')[['format_id']].to_dict('index')
        
        emb_df = pd.read_parquet(embeddings_path)
        self.edition_ids = emb_df['edition_id'].values
        self.id_to_idx = {eid: i for i, eid in enumerate(self.edition_ids)}
        self.embeddings = emb_df.drop(columns=['edition_id']).values.astype('float32')
        self.emb_dim = self.embeddings.shape[1]
        
        print(f"Loading sequences from {sequences_path}...")
        self.samples = []
        with open(sequences_path, 'r') as f:
            for line in f:
                seq_data = json.loads(line)
                user_id = seq_data['user_id']
                history = seq_data['history']
                
                if self.mode == 'train':
                    # Sliding window: (1)->2, (1,2)->3, ..., (1...N-1)->N
                    # We skip very short sequences and limit the number of samples per user if needed
                    for i in range(1, len(history)):
                        self.samples.append({
                            'user_id': user_id,
                            'history_src': history[:i],
                            'target_item': history[i]
                        })
                else:
                    # Inference/Validation: only predict the very last item or use full history
                    self.samples.append({
                        'user_id': user_id,
                        'history_src': history[:-1] if len(history) > 1 else history,
                        'target_item': history[-1]
                    })
        
        print(f"Dataset initialized with {len(self.samples)} samples (Mode: {mode}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        history_src = sample['history_src'][-self.max_len:]
        target_item = sample['target_item']
        
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
                hist_relevance[i] = item['relevance']
                hist_formats[i] = self.item_meta.get(eid, {'format_id': 0})['format_id']
                mask[i] = True
        
        # Prepare target
        target_eid = target_item['edition_id']
        target_emb = np.zeros(self.emb_dim, dtype='float32')
        if target_eid in self.id_to_idx:
            target_emb = self.embeddings[self.id_to_idx[target_eid]]
            
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
