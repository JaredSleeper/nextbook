"""Offline held-out-user eval for nextbook.

Protocol: users in data_eval/holdout_users.npy (excluded from CF training). For each sampled
user with >= MIN_FIVE 5-star ratings, 10 random 5-star books are the "top ten favorites";
the rest of their 5-star books are the targets. We rank K books and measure:
  recall@K  = hits / min(K, #targets)
  hit@K     = share of users with at least one hit
  median ratings_count of recommended books (lower = less mainstream)
  coverage  = distinct books recommended across all users / catalog size

Claude-only baseline (--claude N): ask claude-sonnet-4-6 for K books from the same 10 titles,
map titles back to the catalog, score the same way, alongside our hybrid on those same users.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

K = 20
MIN_FIVE = 15


def norm(s: str) -> str:
    s = re.sub(r"\s*\([^)]*#[^)]*\)\s*$", "", s or "")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def load_users(ratings_path, holdout, n_users, seed):
    five = defaultdict(list)
    hold = set(int(u) for u in holdout)
    with open(ratings_path) as f:
        for row in csv.DictReader(f):
            u = int(row["user_id"])
            if u in hold and int(row["rating"]) == 5:
                five[u].append(int(row["book_id"]))
    rng = np.random.default_rng(seed)
    users = sorted(u for u, b in five.items() if len(b) >= MIN_FIVE)
    users = [users[i] for i in rng.permutation(len(users))[:n_users]]
    out = []
    for u in users:
        books = sorted(set(five[u]))
        perm = rng.permutation(len(books))
        seeds = [books[i] for i in perm[:10]]
        targets = set(books[i] for i in perm[10:])
        out.append((u, seeds, targets))
    return out


def score(recs_by_user, users, counts, n_books):
    recalls, hits, pops, seen = [], 0, [], set()
    for (_, _, targets), all_recs in zip(users, recs_by_user, strict=True):
        recs = all_recs[:K]
        h = len(set(recs) & targets)
        recalls.append(h / min(K, len(targets)))
        hits += h > 0
        pops.extend(counts[b] for b in recs)
        seen.update(recs)
    return {
        "recall@20": round(float(np.mean(recalls)), 4),
        "hit@20": round(hits / len(users), 3),
        "median_ratings_count": int(np.median(pops)) if pops else 0,
        "coverage": round(len(seen) / n_books, 3),
    }


def claude_recs(seed_books, catalog, client):
    lines = "\n".join(f"- {catalog[b]['title']} by {catalog[b]['author']}" for b in seed_books)
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": (
                    f"These are my ten favorite books:\n{lines}\n\nRecommend {K} other books "
                    "I'm likely to love, best first. Don't include books from the same "
                    "series as the ones listed. Reply with only JSON: "
                    '{"books":[{"title":"...","author":"..."}]}'
                ),
            }
        ],
    )
    text = msg.content[0].text
    text = text[text.find("{") : text.rfind("}") + 1]
    return json.loads(text)["books"], msg.usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-eval", default="data_eval")
    ap.add_argument("--ratings", default="data/raw/ratings.csv")
    ap.add_argument("--users", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--claude", type=int, default=0, help="users for the Claude-only baseline")
    args = ap.parse_args()

    from nextbook.recommend import Params, load

    eng = load(args.data_eval)
    holdout = np.load(os.path.join(args.data_eval, "holdout_users.npy"))
    users = load_users(args.ratings, holdout, args.users, args.seed)
    print(f"{len(users)} held-out users with >= {MIN_FIVE} five-star ratings", flush=True)

    n_books = len(eng.catalog)

    def to_idx(book_id):
        return eng.id_to_index[book_id]

    def to_bid(index):
        return int(eng.catalog[index]["id"])

    counts = {to_bid(i): int(eng.catalog[i].get("ratings_count", 0)) for i in range(n_books)}

    def run(params):
        out = []
        for _, seeds, _ in users:
            idx = [to_idx(b) for b in seeds]
            out.append([to_bid(i) for i, _ in eng.recommend(idx, k=K, params=params)])
        return out

    def popularity():
        order = sorted(counts, key=counts.get, reverse=True)
        out = []
        for _, seeds, _ in users:
            s = set(seeds)
            out.append([b for b in order if b not in s][:K])
        return out

    variants = {
        "popularity": None,
        "cf ease": Params(cf_method="ease", w_cf=1.0, w_content=0.0),
        "cf cosine": Params(cf_method="cosine", w_cf=1.0, w_content=0.0),
        "content only": Params(w_cf=0.0, w_content=1.0),
        "hybrid ease .7/.3": Params(pop_beta=0.0),
        "hybrid ease .85/.15": Params(w_cf=0.85, w_content=0.15),
        "hybrid ease .5/.5": Params(w_cf=0.5, w_content=0.5),
        "hybrid ease .7/.3 pop.15": Params(pop_beta=0.15),
        "hybrid ease .7/.3 pop.3": Params(pop_beta=0.3),
        "hybrid cosine .7/.3": Params(cf_method="cosine"),
    }
    results = {}
    for name, p in variants.items():
        t = time.time()
        recs = popularity() if p is None else run(p)
        results[name] = score(recs, users, counts, n_books)
        ms = (time.time() - t) / len(users) * 1000
        print(f"{name:28s} {results[name]}  ({ms:.0f} ms/user)", flush=True)

    if args.claude:
        import anthropic

        client = anthropic.Anthropic()
        cat = {to_bid(i): eng.catalog[i] for i in range(n_books)}
        by_title = defaultdict(list)
        for b, info in cat.items():
            by_title[norm(info["title"])].append(b)
        sub = users[: args.claude]
        mapped = total = 0
        tin = tout = 0
        claude_out = []
        for _, seeds, _ in sub:
            try:
                books, usage = claude_recs(seeds, cat, client)
                tin += usage.input_tokens
                tout += usage.output_tokens
            except Exception as e:  # noqa: BLE001
                print("claude error", e, flush=True)
                books = []
            recs = []
            for bk in books[:K]:
                total += 1
                cands = by_title.get(norm(bk.get("title", "")), [])
                last = norm(bk.get("author", "")).split(" ")[-1:] or [""]
                cands = [c for c in cands if last[0] in norm(cat[c]["author"])] or cands[:1]
                if cands and cands[0] not in seeds:
                    recs.append(cands[0])
                    mapped += 1
            claude_out.append(recs)
        users_backup = users
        users = sub
        ours = run(Params())
        print(f"\nSame {len(sub)} users:")
        print(f"{'claude only':28s} {score(claude_out, sub, counts, n_books)}")
        print(f"{'ours (hybrid default)':28s} {score(ours, sub, counts, n_books)}")
        print(f"claude titles found in catalog: {mapped}/{total} ({mapped / max(total, 1):.0%})")
        print(f"claude tokens: in {tin}, out {tout}")
        users = users_backup


if __name__ == "__main__":
    main()
