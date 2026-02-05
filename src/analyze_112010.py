import pandas as pd
import json
import os

# Paths
interactions_path = 'data/interactions.csv'
candidates_json_path = 'exports/candidates_json/user_112010_candidates.json'

# Load interactions
df_inter = pd.read_csv(interactions_path)
user_inter = df_inter[df_inter['user_id'] == 112010].sort_values('event_ts')

print(f"User 112010 has {len(user_inter)} interactions.")
print("Columns:", user_inter.columns.tolist())
print("\nFirst 5 interactions:")
print(user_inter.head(5))
print("\nLast 5 interactions:")
print(user_inter.tail(5))

# Load candidates
with open(candidates_json_path, 'r') as f:
    candidates_data = json.load(f)

print(f"\nUser 112010 has {len(candidates_data['candidates'])} candidates.")

# Check overlap
candidate_ids = set(c['edition_id'] for c in candidates_data['candidates'])
history_ids = set(user_inter['edition_id'].unique())

overlap = candidate_ids.intersection(history_ids)
print(f"\nOverlap between history and candidates: {len(overlap)} items.")
if len(overlap) > 0:
    print("Overlap items:", list(overlap)[:10])

# Check event types
print("\nEvent types distribution:")
print(user_inter['event_type'].value_counts())

# Generate a sample "User Representation"
# We want to format this as a sequence of events
history_sequence = []
for _, row in user_inter.iterrows():
    history_sequence.append({
        'item_id': int(row['edition_id']),
        'timestamp': row['event_ts'],
        'event_type': int(row['event_type']),
        'rating': row['rating'] if not pd.isna(row['rating']) else None
    })

# Format candidates with placeholder scores/labels
formatted_candidates = []
for c in candidates_data['candidates']:
    # Mock feature extraction (in reality would come from item encoder)
    formatted_candidates.append({
        'id': c['edition_id'],
        'features': {
            'title': c['title'],
            'author': c['author_name'],
            'genres': c['genres_list']
        },
        'label': '?' # To be filled by ground truth in training, or predicted in inference
    })

sample_dataset = {
    'user_id': 112010,
    'history_length': len(history_sequence),
    'history': history_sequence, # The input sequence
    'candidates': formatted_candidates # The items to rank
}

# Just print the structure summary
print("\nGenerated Sample Dataset Structure:")
print(json.dumps({
    'user_id': sample_dataset['user_id'],
    'history_example (last 2)': sample_dataset['history'][-2:],
    'candidate_example (first 1)': sample_dataset['candidates'][:1]
}, indent=2, ensure_ascii=False))
