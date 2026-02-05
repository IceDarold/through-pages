import pandas as pd
import os
import argparse

def enrich_item_features(input_path, output_path):
    print(f"Loading features from {input_path}...")
    df = pd.read_parquet(input_path)
    
    print("Mapping formats and content types...")
    # Define keywords for format detection
    # 0: Standard Book
    # 1: Manga / Manhua / Manhwa
    # 2: Comics / Graphic Novel
    # 3: Audiobooks (placeholder if we find info)
    
    df['format_id'] = 0 # Default: Standard
    
    # Simple keyword-based mapping from genres
    df['genres_lower'] = df['genres'].str.lower().fillna('')
    
    df.loc[df['genres_lower'].str.contains('манга|маньхуа|манхва|manga'), 'format_id'] = 1
    df.loc[df['genres_lower'].str.contains('комикс|графический роман|comic'), 'format_id'] = 2
    
    # Feature for age restriction (useful context)
    # We can normalize age into few buckets
    df['age_bucket'] = 0
    df.loc[df['age_restriction'] >= 12, 'age_bucket'] = 1
    df.loc[df['age_restriction'] >= 16, 'age_bucket'] = 2
    df.loc[df['age_restriction'] >= 18, 'age_bucket'] = 3

    print(f"Format distribution:\n{df['format_id'].value_counts()}")
    
    # Cleanup and save
    df = df.drop(columns=['genres_lower'])
    df.to_parquet(output_path, index=False)
    print(f"Enriched features saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiments/data_v1/item_features.parquet")
    parser.add_argument("--output", default="experiments/data_v1/item_features_enriched.parquet")
    args = parser.parse_args()
    
    enrich_item_features(args.input, args.output)
