import pandas as pd
import os

def create_item_features():
    print("Loading raw item data...")
    editions = pd.read_csv('data/editions.csv')
    authors = pd.read_csv('data/authors.csv')
    book_genres = pd.read_csv('data/book_genres.csv')
    genres = pd.read_csv('data/genres.csv')
    
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
    editions.to_parquet('experiments/data_v1/item_features.parquet', index=False)
    print("Item features saved.")

if __name__ == "__main__":
    create_item_features()
