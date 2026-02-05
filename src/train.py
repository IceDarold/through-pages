import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import tqdm

from models.multi_interest import MultiInterestEncoder
from data.dataset import InteractionDataset

def train_epoch(model, loader, optimizer, device, temperature=0.07):
    model.train()
    total_loss = 0
    
    for batch in tqdm.tqdm(loader, desc="Training"):
        # Move to device
        hist_embs = batch['hist_embs'].to(device)
        relevance = batch['hist_relevance'].to(device)
        formats = batch['hist_formats'].to(device)
        mask = batch['mask'].to(device)
        target_emb = batch['target_emb'].to(device) # [B, D_raw]
        
        optimizer.zero_grad()
        
        # Forward: Get user interests [B, K, D]
        # We assume targets are already normalized in the loss calculation
        user_interests = model(hist_embs, relevance, formats, mask)

        # Project targets into the same space as user_interests
        target_proj = model.item_projector(target_emb)
        target_proj = F.normalize(target_proj, p=2, dim=-1)
        
        # In-batch Negative Sampling Loss
        # 1. Similarity of all K interests with all targets in batch
        # interests: [B, K, D], targets: [N, D]
        # scores: [B, K, N]
        scores = torch.einsum('bkd,nd->bkn', user_interests, target_proj)
        
        # 2. For each user-target pair (within or outside batch), select the best interest
        # resultant_scores: [B, N]
        resultant_scores, _ = torch.max(scores, dim=1)
        
        # 3. CrossEntropy: match user B with target B
        resultant_scores = resultant_scores / temperature
        labels = torch.arange(resultant_scores.size(0)).to(device)
        
        loss = nn.CrossEntropyLoss()(resultant_scores, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Validating"):
            hist_embs = batch['hist_embs'].to(device)
            relevance = batch['hist_relevance'].to(device)
            formats = batch['hist_formats'].to(device)
            mask = batch['mask'].to(device)
            target_emb = batch['target_emb'].to(device)
            
            user_interests = model(hist_embs, relevance, formats, mask)
            target_proj = model.item_projector(target_emb)
            target_proj = F.normalize(target_proj, p=2, dim=-1)
            scores = torch.einsum('bkd,nd->bkn', user_interests, target_proj)
            resultant_scores, _ = torch.max(scores, dim=1)
            labels = torch.arange(resultant_scores.size(0)).to(device)
            loss = nn.CrossEntropyLoss()(resultant_scores / 0.07, labels)
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seq", default="experiments/data_v1/train_sequences.jsonl")
    parser.add_argument("--val-seq", default="experiments/data_v1/val_sequences.jsonl")
    parser.add_argument("--emb-path", default="experiments/data_v1/item_embeddings.parquet")
    parser.add_argument("--feature-path", default="experiments/data_v1/item_features_enriched.parquet")
    parser.add_argument("--output-dir", default="experiments/models_v1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_ds = InteractionDataset(args.train_seq, args.emb_path, args.feature_path)
    val_ds = InteractionDataset(args.val_seq, args.emb_path, args.feature_path)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = MultiInterestEncoder(
        item_emb_dim=train_ds.emb_dim, 
        d_model=256, 
        n_interests=6, 
        n_heads=8, 
        n_layers=3
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_multi_interest.pth"))
            print("Model saved!")

if __name__ == "__main__":
    main()
