import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiInterestEncoder(nn.Module):
    def __init__(self, item_emb_dim=768, d_model=256, n_interests=6, n_heads=8, n_layers=3, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.n_interests = n_interests
        
        # 1. Linear projection of content embeddings (768 -> d_model)
        self.item_projector = nn.Linear(item_emb_dim, d_model)
        
        # 2. Context Embeddings
        # 0: Pad, 1: Type 1, 2: Type 2, 3: Type 3 (rating boost handled in relevance)
        self.relevance_emb = nn.Embedding(5, d_model, padding_idx=0) 
        # 0: Standard, 1: Manga, 2: Comics
        self.format_emb = nn.Embedding(5, d_model, padding_idx=0)    
        # Position embeddings
        self.pos_emb = nn.Embedding(max_len + 1, d_model, padding_idx=0)
        
        # 3. Transformer Encoder (Standard sequence reasoning)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model*4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Multi-Interest Extraction Layer (Learned Attention Centroids)
        # These vectors "query" the history to find different clusters of behavior
        self.interest_queries = nn.Parameter(torch.randn(n_interests, d_model))
        
        # 5. Output projection (optional, to match retrieval space)
        self.output_layer = nn.Linear(d_model, d_model)
        
    def forward(self, item_embs, relevance, formats, mask=None):
        """
        item_embs: [Batch, SeqLen, 768] - Content vectors from Phase 2
        relevance: [Batch, SeqLen] - Pre-calculated 1-3 grades
        formats:   [Batch, SeqLen] - Format IDs (0: Std, 1: Manga, 2: Comics)
        mask:      [Batch, SeqLen] - Boolean mask (True for actual items, False for padding)
        """
        batch_size, seq_len, _ = item_embs.shape
        
        # Project item content
        x = self.item_projector(item_embs) # [B, L, D]
        
        # Add context signals
        x = x + self.relevance_emb(relevance)
        x = x + self.format_emb(formats)
        
        # Add positions (reversed: the most recent item is always index 1)
        # This makes it easier for the model to generalize across sequence lengths
        positions = torch.arange(seq_len, 0, -1, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        if mask is not None:
             positions = positions * mask.long()
        x = x + self.pos_emb(positions)
        
        # Transformer reasoning
        # src_key_padding_mask requires False for valid, True for padding
        padding_mask = ~mask if mask is not None else None
        seq_output = self.transformer(x, src_key_padding_mask=padding_mask) # [B, L, D]
        
        # Multi-interest pooling via Attention
        # queries: [K, d_model], sequence: [Batch, SeqLen, d_model]
        q = self.interest_queries.unsqueeze(0).repeat(batch_size, 1, 1) # [B, K, D]
        
        # Dot-product attention weights [B, K, SeqLen]
        attn_weights = torch.bmm(q, seq_output.transpose(1, 2))
        
        if mask is not None:
            # Mask out padding items so they don't contribute to interests
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(1), -1e9)
        
        attn_probs = F.softmax(attn_weights, dim=-1) # [B, K, SeqLen]
        user_interests = torch.bmm(attn_probs, seq_output) # [Batch, K, d_model]
        
        # Final pass and normalization
        user_interests = self.output_layer(user_interests)
        user_interests = F.normalize(user_interests, p=2, dim=-1) # L2 Norm for cosine similarity
        
        return user_interests

def compute_multi_interest_loss(user_interests, target_item_emb, temperature=0.1):
    """
    user_interests: [Batch, K, D]
    target_item_emb: [Batch, D] (Normalized)
    """
    # 1. Similarity with all K interests
    # user_interests: [B, K, D], target: [B, D, 1]
    scores = torch.bmm(user_interests, target_item_emb.unsqueeze(2)).squeeze(2) # [Batch, K]
    
    # 2. Hard selection: only the best interest is responsible for predicting this item
    best_score, _ = torch.max(scores, dim=1)
    
    # 3. InfoNCE style loss (Simplified here, usually needs negatives)
    loss = -torch.mean(best_score / temperature)
    
    return loss
