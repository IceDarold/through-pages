import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # src

from baseline_utils import (
    log,
    load_data,
    build_mappings,
    add_indices,
    add_weights,
    time_split,
    build_genre_maps,
    build_val_candidates,
    build_submission,
    score_submission,
    save_submission,
)


def score_candidates(candidates_df, train_df, user2idx, item2idx, ed2genres):
    # build user -> genre counts
    ed2genres_idx = {item2idx[ed]: ed2genres.get(ed, set()) for ed in item2idx.keys()}
    user_genre = {}
    for u, grp in train_df.groupby("u"):
        gcount = {}
        for it in grp["i"].values:
            for g in ed2genres_idx.get(it, set()):
                gcount[g] = gcount.get(g, 0) + 1
        user_genre[u] = gcount

    cand = candidates_df.copy()
    cand["u"] = cand["user_id"].map(user2idx).astype(int)
    cand["i"] = cand["edition_id"].map(item2idx).astype(int)

    scores = []
    for u, it in zip(cand["u"].values, cand["i"].values):
        gset = ed2genres_idx.get(it, set())
        if not gset:
            scores.append(0.0)
            continue
        gcount = user_genre.get(u, {})
        s = 0.0
        for g in gset:
            s += gcount.get(g, 0)
        scores.append(s)

    cand["score"] = scores
    return cand[["user_id", "edition_id", "score"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--submit-dir", default="submit")
    ap.add_argument("--out", default="submission_user_genre.csv")
    ap.add_argument("--do-local-split", action="store_true")
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=0.5)
    args = ap.parse_args()

    log("loading data")
    ds = load_data(args.data_dir, args.submit_dir)
    maps = build_mappings(ds)
    ds = add_indices(ds, maps)
    ds.interactions = add_weights(ds.interactions)

    if args.do_local_split:
        split = time_split(ds.interactions, days=30)
    else:
        split = type("Split", (), {"train_df": ds.interactions, "val_df": None})

    ed2genres = build_genre_maps(ds.editions, ds.book_genres, maps.item2idx)

    log("scoring candidates")
    cand_scored = score_candidates(ds.candidates, split.train_df, maps.user2idx, maps.item2idx, ed2genres)

    log("building submission")
    submission = build_submission(cand_scored, ed2genres, topk=20, lam=args.lam, gamma=args.gamma)
    save_submission(submission, args.out)

    if split.val_df is not None and len(split.val_df) > 0:
        log("building validation candidates")
        val_candidates = build_val_candidates(split.val_df, maps, ed2genres, split.train_df)

        log("scoring validation candidates")
        val_scored = score_candidates(val_candidates, split.train_df, maps.user2idx, maps.item2idx, ed2genres)
        val_submission = build_submission(val_scored, ed2genres, topk=20, lam=args.lam, gamma=args.gamma)

        ndcg, div, score = score_submission(val_submission, split.val_df, ed2genres)
        log(f"local NDCG@20: {ndcg:.6f}")
        log(f"local Diversity@20: {div:.6f}")
        log(f"local Score: {score:.6f}")


if __name__ == "__main__":
    main()
