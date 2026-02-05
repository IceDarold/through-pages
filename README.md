# BookRec Hackathon — Participants Package

## Layout
- `data/`: data files for model training
- `submit/`: files needed for submission

## data/
- `interactions.csv`: train-only interactions (event_ts < test_start_ts)
- `users.csv`, `editions.csv`, `authors.csv`, `genres.csv`, `book_genres.csv`

## submit/
- `targets.csv`: list of users (user_id) to score
- `candidates.csv`: 200 candidate editions per user
- `example_submission.csv`: random baseline submission

## Submission format
`submission.csv` with columns:

```
user_id,edition_id,rank
```

Requirements:
- 20 rows per user
- rank 1..20, unique per user
- edition_id unique per user
- edition_id must be in candidates for that user
