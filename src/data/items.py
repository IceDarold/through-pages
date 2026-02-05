import argparse
import os

import pandas as pd

def create_item_features(data_dir="/kaggle/input/though-pages/data", out_dir="/kaggle/working/through-pages/experiments/data_v1"):
    print("Loading raw item data...")
    editions = pd.read_csv(os.path.join(data_dir, "editions.csv"))
    authors = pd.read_csv(os.path.join(data_dir, "authors.csv"))
    book_genres = pd.read_csv(os.path.join(data_dir, "book_genres.csv"))
    genres = pd.read_csv(os.path.join(data_dir, "genres.csv"))
    
    # Merge authors
    print("Merging authors...")
    authors = authors.rename(columns={'id': 'author_id', 'name': 'author_name'})
    editions = editions.merge(authors[['author_id', 'author_name']], on='author_id', how='left')
    
    # Merge genres (multiple per book)
    print("Aggregating genres...")
    book_genres = book_genres.merge(genres, on='genre_id', how='left')
    genres_agg = book_genres.groupby('book_id')['genre_name'].apply(lambda x: ', '.join(x.dropna())).reset_index()
    genres_agg = genres_agg.rename(columns={'genre_name': 'genres'})
    
    editions = editions.merge(genres_agg, on='book_id', how='left')
    
    print(f"Final Item catalog size: {len(editions)}")
    os.makedirs(out_dir, exist_ok=True)
    editions.to_parquet(os.path.join(out_dir, "item_features.parquet"), index=False)
    print("Item features saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/kaggle/input/though-pages/data")
    ap.add_argument("--out-dir", default="/kaggle/working/through-pages/experiments/data_v1")
    args = ap.parse_args()
    create_item_features(data_dir=args.data_dir, out_dir=args.out_dir)
