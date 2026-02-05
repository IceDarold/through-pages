import os
import math
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    users: pd.DataFrame
    interactions: pd.DataFrame
    editions: pd.DataFrame
    book_genres: pd.DataFrame
    targets: pd.DataFrame
    candidates: pd.DataFrame


@dataclass
class Mappings:
    user2idx: dict
    item2idx: dict
    idx2item: dict
    n_users: int
    n_items: int


@dataclass
class Split:
    train_df: pd.DataFrame
    val_df: pd.DataFrame | None


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def load_data(data_dir="data", submit_dir="submit") -> Dataset:
    users = pd.read_csv(os.path.join(data_dir, "users.csv"))
    interactions = pd.read_csv(os.path.join(data_dir, "interactions.csv"), parse_dates=["event_ts"])
    editions = pd.read_csv(os.path.join(data_dir, "editions.csv"))
    book_genres = pd.read_csv(os.path.join(data_dir, "book_genres.csv"))

    targets = pd.read_csv(os.path.join(submit_dir, "targets.csv"))
    candidates = pd.read_csv(os.path.join(submit_dir, "candidates.csv"))

    log(f"users: {users.shape}, interactions: {interactions.shape}, editions: {editions.shape}")
    log(f"book_genres: {book_genres.shape}, targets: {targets.shape}, candidates: {candidates.shape}")

    return Dataset(
        users=users,
        interactions=interactions,
        editions=editions,
        book_genres=book_genres,
        targets=targets,
        candidates=candidates,
    )


def build_mappings(ds: Dataset) -> Mappings:
    all_user_ids = pd.Index(pd.concat([ds.users["user_id"], ds.interactions["user_id"], ds.targets["user_id"]]).unique())
    all_edition_ids = pd.Index(pd.concat([ds.editions["edition_id"], ds.interactions["edition_id"], ds.candidates["edition_id"]]).unique())

    user2idx = {u: i for i, u in enumerate(all_user_ids)}
    item2idx = {it: i for i, it in enumerate(all_edition_ids)}
    idx2item = {i: it for it, i in item2idx.items()}

    n_users = len(all_user_ids)
    n_items = len(all_edition_ids)

    log(f"n_users: {n_users}, n_items: {n_items}")
    return Mappings(user2idx=user2idx, item2idx=item2idx, idx2item=idx2item, n_users=n_users, n_items=n_items)


def add_indices(ds: Dataset, maps: Mappings) -> Dataset:
    ds = Dataset(**ds.__dict__)
    ds.interactions = ds.interactions.copy()
    ds.interactions["u"] = ds.interactions["user_id"].map(maps.user2idx).astype(np.int64)
    ds.interactions["i"] = ds.interactions["edition_id"].map(maps.item2idx).astype(np.int64)
    return ds


def add_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    w = np.zeros(len(df), dtype=np.float32)
    is_wish = df["event_type"] == 1
    is_read = df["event_type"] == 2

    w[is_wish] = 1.0
    w[is_read] = 3.0

    r = df["rating"].astype("float32").fillna(0.0).clip(0.0, 5.0) / 5.0
    w[is_read] += 0.2 * r[is_read].to_numpy()

    df["w"] = w
    return df


def time_split(interactions: pd.DataFrame, days=30) -> Split:
    interactions = interactions.sort_values(["u", "event_ts"])
    max_ts = interactions.groupby("u")["event_ts"].transform("max")
    cutoff = max_ts - pd.Timedelta(days=days)

    train_df = interactions[interactions["event_ts"] < cutoff].copy()
    val_df = interactions[interactions["event_ts"] >= cutoff].copy()

    log(f"train events: {len(train_df)}, val events: {len(val_df)}")
    return Split(train_df=train_df, val_df=val_df)


def build_genre_maps(editions: pd.DataFrame, book_genres: pd.DataFrame, item2idx: dict) -> dict:
    ed2book = dict(zip(editions["edition_id"].values, editions["book_id"].values))
    bg = book_genres.groupby("book_id")["genre_id"].apply(lambda s: set(s.values)).to_dict()

    ed2genres = {}
    for ed, idx in item2idx.items():
        b = ed2book.get(ed, None)
        ed2genres[ed] = bg.get(b, set())
    return ed2genres


# ---------- Local metric ----------

def build_relevance(val_df: pd.DataFrame) -> dict:
    rel = defaultdict(dict)
    if val_df is None or len(val_df) == 0:
        return rel
    for (u, ed), grp in val_df.groupby(["user_id", "edition_id"]):
        if (grp["event_type"] == 2).any():
            rel[u][ed] = 3
        elif (grp["event_type"] == 1).any():
            rel[u][ed] = 1
    return rel


def ndcg_at_20(ranked_items: list, rel_u: dict) -> float:
    gains = []
    for k, ed in enumerate(ranked_items, start=1):
        r = rel_u.get(ed, 0)
        gains.append(r / math.log2(k + 1))
    dcg = sum(gains)

    ideal_rels = sorted(rel_u.values(), reverse=True)[:20]
    idcg = 0.0
    for k, r in enumerate(ideal_rels, start=1):
        idcg += r / math.log2(k + 1)
    return 0.0 if idcg == 0 else dcg / idcg


def jaccard_dist(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - (inter / union if union else 0.0)


def diversity_at_20(ranked_items: list, rel_u: dict, ed2genres: dict) -> float:
    rel_items = [ed for ed in ranked_items if rel_u.get(ed, 0) > 0]

    w = [1.0 / math.log2(k + 1) for k in range(1, 21)]

    S = set()
    num = 0.0
    den = 0.0
    for k, ed in enumerate(ranked_items, start=1):
        g = ed2genres.get(ed, set())
        if len(g) == 0:
            den += w[k - 1] * 0.0
            continue
        if rel_u.get(ed, 0) > 0:
            new = len(g - S) / len(g)
            num += w[k - 1] * new
            S |= g
        den += w[k - 1] * len(g)
    coverage = 0.0 if den == 0 else num / den

    if len(rel_items) < 2:
        ild = 0.0
    else:
        dsum = 0.0
        cnt = 0
        for i in range(len(rel_items)):
            for j in range(i + 1, len(rel_items)):
                dsum += jaccard_dist(ed2genres.get(rel_items[i], set()), ed2genres.get(rel_items[j], set()))
                cnt += 1
        ild = dsum / cnt if cnt > 0 else 0.0

    return 0.5 * coverage + 0.5 * ild


def score_submission(pred_df: pd.DataFrame, val_df: pd.DataFrame, ed2genres: dict) -> tuple[float, float, float]:
    rel = build_relevance(val_df)
    users = pred_df["user_id"].unique()

    ndcgs = []
    divs = []
    for u in users:
        ranked = pred_df[pred_df["user_id"] == u].sort_values("rank")["edition_id"].tolist()
        rel_u = rel.get(u, {})
        ndcgs.append(ndcg_at_20(ranked, rel_u))
        divs.append(diversity_at_20(ranked, rel_u, ed2genres))

    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    div = float(np.mean(divs)) if divs else 0.0
    score = 0.7 * ndcg + 0.3 * div
    return ndcg, div, score


# ---------- Validation candidates ----------

def build_val_candidates(
    val_df: pd.DataFrame,
    maps: Mappings,
    ed2genres: dict,
    train_df: pd.DataFrame,
    per_user=200,
    pop_k=200,
    genre_k=200,
    coread_k=200,
    max_seed_items=5,
    max_users_per_item=200,
    max_items_per_user=20,
    seed=42,
):
    rng = np.random.default_rng(seed)
    log("building validation candidates: start")

    item_pop = train_df.groupby("i").size().sort_values(ascending=False)
    item_pop_rank = item_pop.index.to_numpy()

    user_items = train_df.groupby("u")["i"].apply(list).to_dict()
    item_users = train_df.groupby("i")["u"].apply(list).to_dict()

    ed2genres_idx = {maps.item2idx[ed]: ed2genres.get(ed, set()) for ed in maps.item2idx.keys()}

    genre_to_items = {}
    for item_idx in range(maps.n_items):
        gset = ed2genres_idx.get(item_idx, set())
        for g in gset:
            genre_to_items.setdefault(g, []).append(item_idx)

    for g, items in genre_to_items.items():
        items.sort(key=lambda x: item_pop.get(x, 0), reverse=True)

    val_users = val_df["user_id"].unique()
    val_pairs = val_df.groupby("user_id")["edition_id"].apply(set).to_dict()

    rows = []
    for u in val_users:
        u_idx = maps.user2idx[u]
        train_items = set(user_items.get(u_idx, []))
        positives = set(maps.item2idx[ed] for ed in val_pairs.get(u, set()) if ed in maps.item2idx)

        cand = set(positives)

        # popularity
        for it in item_pop_rank[:pop_k]:
            if it not in train_items:
                cand.add(it)
            if len(cand) >= per_user:
                break

        # genre-based
        gcount = Counter()
        for it in list(train_items)[:200]:
            for g in ed2genres_idx.get(it, set()):
                gcount[g] += 1

        top_genres = [g for g, _ in gcount.most_common(10)]
        for g in top_genres:
            for it in genre_to_items.get(g, [])[:genre_k]:
                if it not in train_items:
                    cand.add(it)
                if len(cand) >= per_user:
                    break
            if len(cand) >= per_user:
                break

        # co-read
        seed_items = list(train_items)[:max_seed_items]
        for it in seed_items:
            users = item_users.get(it, [])
            if len(users) > max_users_per_item:
                users = rng.choice(users, size=max_users_per_item, replace=False)
            for v in users:
                v_items = user_items.get(v, [])[:max_items_per_user]
                for it2 in v_items:
                    if it2 not in train_items:
                        cand.add(it2)
                    if len(cand) >= per_user:
                        break
                if len(cand) >= per_user:
                    break
            if len(cand) >= per_user:
                break

        # fill random
        if len(cand) < per_user:
            pool = [it for it in item_pop_rank if it not in train_items]
            need = per_user - len(cand)
            if need > 0:
                extra = rng.choice(pool, size=min(need, len(pool)), replace=False)
                cand.update(extra)

        cand_list = list(cand)[:per_user]
        for it in cand_list:
            rows.append((u, maps.idx2item[it]))

    df = pd.DataFrame(rows, columns=["user_id", "edition_id"])
    log(f"building validation candidates: done, rows={len(df)} users={len(val_users)}")
    return df


# ---------- Diversity rerank ----------

def rerank_diverse(df_user: pd.DataFrame, ed2genres: dict, topk=20, lam=0.15, gamma=0.5) -> list:
    items = df_user.sort_values("score", ascending=False)[["edition_id", "score"]].to_records(index=False)

    chosen = []
    chosen_genres = set()

    for _ in range(topk):
        best = None
        best_val = -1e18

        for ed, s in items:
            if ed in chosen:
                continue

            g = ed2genres.get(ed, set())
            if len(g) > 0:
                new = len(g - chosen_genres) / len(g)
            else:
                new = 0.0

            if not chosen:
                ild = 0.0
            else:
                dsum = 0.0
                for prev in chosen:
                    dsum += jaccard_dist(g, ed2genres.get(prev, set()))
                ild = dsum / len(chosen)

            val = float(s) + lam * (new + gamma * ild)
            if val > best_val:
                best_val = val
                best = ed

        chosen.append(best)
        chosen_genres |= ed2genres.get(best, set())

    return chosen


def build_submission(cand_scored: pd.DataFrame, ed2genres: dict, topk=20, lam=0.15, gamma=0.5) -> pd.DataFrame:
    pred_rows = []
    for u, grp in cand_scored.groupby("user_id"):
        chosen = rerank_diverse(grp, ed2genres, topk=topk, lam=lam, gamma=gamma)
        for r, ed in enumerate(chosen, start=1):
            pred_rows.append((u, ed, r))

    submission = pd.DataFrame(pred_rows, columns=["user_id", "edition_id", "rank"])
    return submission


def save_submission(submission: pd.DataFrame, out_path: str):
    submission.to_csv(out_path, index=False)

    ok_20 = submission.groupby("user_id").size().eq(20).all()
    unique_ed = submission.groupby("user_id")["edition_id"].nunique().eq(20).all()
    unique_rank = submission.groupby("user_id")["rank"].nunique().eq(20).all()

    log(f"saved: {out_path}")
    log(f"20 rows per user: {ok_20}")
    log(f"unique edition_id: {unique_ed}")
    log(f"unique rank: {unique_rank}")
