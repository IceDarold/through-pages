import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
import os
import tqdm
import json

from models.multi_interest import MultiInterestEncoder
from data.dataset import InteractionDataset

def generate_user_vectors(model, loader, device, output_path):
    model.eval()
    user_ids = []
    interest_vectors = []
    
    with torch.no_grad():
        for i in tqdm.trange(len(loader.dataset), desc="Inference"):
            # We process 1 by 1 for simplicity and to preserve user_id mapping
            batch = loader.dataset[i]
            
            # Add batch dimension
            hist_embs = batch['hist_embs'].unsqueeze(0).to(device)
            relevance = batch['hist_relevance'].unsqueeze(0).to(device)
            formats = batch['hist_formats'].unsqueeze(0).to(device)
            mask = batch['mask'].unsqueeze(0).to(device)
            
            # Get interests [1, K, D]
            interests = model(hist_embs, relevance, formats, mask)
            
            # Use 'samples' which is the new name in InteractionDataset
            user_id = loader.dataset.samples[i]['user_id']
            user_ids.append(user_id)
            interest_vectors.append(interests.cpu().numpy()[0])
            
    # Save as numpy array [N, K, D]
    interest_vectors = np.array(interest_vectors)
    np.save(output_path, {
        'user_ids': np.array(user_ids),
        'interests': interest_vectors
    })
    print(f"Saved user interests to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default="experiments/data_v1/train_sequences.jsonl")
    parser.add_argument("--emb-path", default="experiments/data_v1/item_embeddings.parquet")
    parser.add_argument("--feature-path", default="experiments/data_v1/item_features_enriched.parquet")
    parser.add_argument("--model-path", default="experiments/models_v1/best_multi_interest.pth")
    parser.add_argument("--output", default="experiments/data_v1/user_interests.npy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset (FORCE mode='infer' to get 1 sample per user)
    ds = InteractionDataset(args.sequences, args.emb_path, args.feature_path, mode='infer')
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Model
    model = MultiInterestEncoder(
        item_emb_dim=ds.emb_dim, 
        d_model=256, 
        n_interests=6, 
        n_heads=8, 
        n_layers=3
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    generate_user_vectors(model, loader, device, args.output)

if __name__ == "__main__":
    main()
