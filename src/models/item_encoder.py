import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import argparse
import os

def generate_embeddings(input_path, output_path, model_name='paraphrase-multilingual-mpnet-base-v2', batch_size=64):
    print(f"Loading item features from {input_path}...")
    df = pd.read_parquet(input_path)
    
    print("Preparing text for fusion...")
    # Fill NaN to avoid string concatenation issues
    df['title'] = df['title'].fillna('')
    df['author_name'] = df['author_name'].fillna('Unknown Author')
    df['genres'] = df['genres'].fillna('')
    df['description'] = df['description'].fillna('')
    
    # Create the fused text: [TITLE] + [AUTHOR] + [GENRES] + [DESCRIPTION]
    # We truncate description to first 300 characters to keep it relevant and fit transformer context better
    df['text_to_encode'] = (
        df['title'] + " " + 
        "автор " + df['author_name'] + ". " + 
        "жанры: " + df['genres'] + ". " + 
        df['description'].str[:300]
    )
    
    print(f"Initializing model: {model_name}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    
    print(f"Encoding {len(df)} items on {device} (batch_size={batch_size})...")
    embeddings = model.encode(
        df['text_to_encode'].tolist(), 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    print(f"Saving embeddings to {output_path}...")
    # We save as a dataframe with edition_id to ensure alignment
    emb_df = pd.DataFrame(embeddings)
    emb_df.columns = [f'emb_{i}' for i in range(embeddings.shape[1])]
    emb_df['edition_id'] = df['edition_id'].values
    
    # Reorder columns to have edition_id first
    cols = ['edition_id'] + [c for c in emb_df.columns if c != 'edition_id']
    emb_df = emb_df[cols]
    
    emb_df.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='experiments/data_v1/item_features.parquet')
    parser.add_argument("--output", type=str, default='experiments/data_v1/item_embeddings.parquet')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="Force re-encoding even if cached embeddings exist")
    args = parser.parse_args()
    
    cached_path = "/kaggle/input/though-pages/item_embeddings.parquet"
    if not args.force:
        if os.path.exists(cached_path):
            print(f"Found cached embeddings at {cached_path}. Skipping encoding. Use --force to regenerate.")
            raise SystemExit(0)
        if os.path.exists(args.output):
            print(f"Found embeddings at {args.output}. Skipping encoding. Use --force to regenerate.")
            raise SystemExit(0)

    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    generate_embeddings(args.input, args.output, batch_size=args.batch_size)
