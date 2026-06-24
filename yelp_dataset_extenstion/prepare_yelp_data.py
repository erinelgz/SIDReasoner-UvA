"""
prepare_yelp_data.py

Converts the Yelp academic dataset into the SIDReasoner data format.

Usage (on Snellius, after extracting the tar):
    python prepare_yelp_data.py \
        --tar_path /gpfs/home2/scur1223/yelp_dataset.tar \
        --out_dir   ./data/Yelp \
        --category  Restaurants \
        --min_interactions 5

Output structure mirrors data/Amazon/:
    data/Yelp/
      train/Yelp_Restaurants_5core.csv
      valid/Yelp_Restaurants_5core.csv
      test/Yelp_Restaurants_5core.csv
      info/Yelp_Restaurants_5core.txt
      index/Yelp_Restaurants.index.json
      index/Yelp_Restaurants.item.json
      index/Yelp_Restaurants.item_enhanced_v2.json
      index/Yelp_Restaurants.integrated_narrative.csv
      general/sampled_data.arrow  (symlink to Amazon version)
      rec_reasoning_verl/Yelp_Restaurants/train.parquet
      rec_reasoning_verl/Yelp_Restaurants/test.parquet

Semantic IDs:
    Uses a three-level category-based assignment without RQVAE training:
      <a_X>  top-level Yelp category cluster  (up to 256 codes)
      <b_Y>  second-level category cluster    (up to 256 codes per a)
      <c_Z>  sequential index within (a, b)   (up to 256 codes)

    Items that share top-level and sub-level categories receive similar IDs,
    which is semantically meaningful for the recommendation task.
    For better quality, replace with RQVAE-trained codes on item embeddings.

LLM-enriched fields:
    item_enhanced_v2.json → "llm_stage2" filled with template text using
    available Yelp metadata (name, categories, city, stars).
    integrated_narrative.csv → "integrated_narrative" filled with template
    reasoning text. These placeholders are functional; replace with
    actual LLM-generated text for best results.
"""

import argparse
import ast
import collections
import json
import os
import re
import tarfile
from pathlib import Path

import pandas as pd


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tar_path", default="/gpfs/home2/scur1223/yelp_dataset.tar")
    p.add_argument("--out_dir", default="./data/Yelp")
    p.add_argument(
        "--category", default="Restaurants", help="Top-level Yelp category filter, or 'all' to keep everything"
    )
    p.add_argument(
        "--min_interactions", type=int, default=5, help="k-core filter: keep users/items with >= k interactions"
    )
    p.add_argument("--max_seq_len", type=int, default=10, help="Truncate user history to this many items (most recent)")
    p.add_argument(
        "--max_users",
        type=int,
        default=-1,
        help="Randomly subsample this many users after k-core filtering "
        "(-1 = keep all). Item index files always cover all items.",
    )
    p.add_argument(
        "--amazon_general_path",
        default="./data/Amazon/general/sampled_data.arrow",
        help="Path to the Amazon general reasoning file to symlink",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────


def load_businesses_from_tar(tar_path, category_filter):
    """Return {business_id: meta_dict} for businesses matching the category."""
    print(f"Loading businesses (category filter: {category_filter!r}) ...")
    businesses = {}
    with tarfile.open(tar_path) as t:
        f = t.extractfile("yelp_academic_dataset_business.json")
        for line in f:
            biz = json.loads(line)
            cats = biz.get("categories") or ""
            if category_filter.lower() == "all" or category_filter in cats:
                businesses[biz["business_id"]] = biz
    print(f"  {len(businesses):,} businesses after category filter")
    return businesses


def load_reviews_from_tar(tar_path, valid_business_ids):
    """Return DataFrame with columns [user_id, business_id, date] sorted by user+date."""
    print("Loading reviews ...")
    rows = []
    with tarfile.open(tar_path) as t:
        f = t.extractfile("yelp_academic_dataset_review.json")
        for line in f:
            rev = json.loads(line)
            if rev["business_id"] in valid_business_ids:
                rows.append((rev["user_id"], rev["business_id"], rev["date"]))
    df = pd.DataFrame(rows, columns=["user_id", "business_id", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    # Keep the last interaction per (user, business) pair to avoid duplicates
    df = df.drop_duplicates(subset=["user_id", "business_id"], keep="last")
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    print(f"  {len(df):,} interactions after dedup")
    return df


def kcore_filter(df, k):
    """Iteratively remove users/items with fewer than k interactions."""
    print(f"Applying {k}-core filter ...")
    while True:
        item_counts = df["business_id"].value_counts()
        user_counts = df["user_id"].value_counts()
        mask = (df["business_id"].map(item_counts) >= k) & (df["user_id"].map(user_counts) >= k)
        filtered = df[mask]
        if len(filtered) == len(df):
            break
        df = filtered.reset_index(drop=True)
    print(
        f"  {df['user_id'].nunique():,} users, "
        f"{df['business_id'].nunique():,} items, "
        f"{len(df):,} interactions after {k}-core"
    )
    return df


# ── ID mapping ────────────────────────────────────────────────────────────────


def build_id_maps(df, businesses):
    """Map business_id → int item_id, user_id → int user_id (for CSV)."""
    item_ids = sorted(df["business_id"].unique())
    user_ids = sorted(df["user_id"].unique())
    item2int = {bid: i for i, bid in enumerate(item_ids)}
    user2int = {uid: i for i, uid in enumerate(user_ids)}
    return item2int, user2int, item_ids


# ── Semantic ID assignment ────────────────────────────────────────────────────


def _extract_categories(biz):
    """Return list of non-empty category strings for a business."""
    raw = biz.get("categories") or ""
    return [c.strip() for c in raw.split(",") if c.strip()]


def assign_semantic_ids(item_ids, businesses, codebook_size=256):
    """
    Assign hierarchical semantic IDs <a_X><b_Y><c_Z> to each item.

    Strategy:
      a  ← index of the item's primary category in the global ranked list
           (capped at codebook_size-1)
      b  ← index of the item's secondary category within the (a) group
           (capped at codebook_size-1)
      c  ← sequential counter within the (a, b) bucket
           (capped at codebook_size-1; wraps after 256 items per bucket)

    Items with no category get a=0, b=0, c=sequential.
    """
    print("Assigning category-based semantic IDs ...")

    # Build global category vocab ordered by frequency
    cat_counter = collections.Counter()
    for bid in item_ids:
        biz = businesses.get(bid, {})
        for cat in _extract_categories(biz):
            cat_counter[cat] += 1

    # Top-level (a) vocab: most common categories first
    top_cats = [cat for cat, _ in cat_counter.most_common()]
    # Reserve code 0 for "unknown"
    top_cat2a = {cat: min(i + 1, codebook_size - 1) for i, cat in enumerate(top_cats)}

    # Per-(a) sub-category vocab for (b) assignment
    sub_cat_counter = collections.defaultdict(collections.Counter)
    for bid in item_ids:
        biz = businesses.get(bid, {})
        cats = _extract_categories(biz)
        a_code = top_cat2a.get(cats[0], 0) if cats else 0
        for cat in cats[1:]:
            sub_cat_counter[a_code][cat] += 1

    sub_cat2b = {}
    for a_code, ctr in sub_cat_counter.items():
        sub_cats = [cat for cat, _ in ctr.most_common()]
        # Reserve code 0 for "no sub-category"
        sub_cat2b[a_code] = {cat: min(i + 1, codebook_size - 1) for i, cat in enumerate(sub_cats)}

    # Assign IDs
    bucket_counters = collections.defaultdict(int)
    sid_map = {}  # int_item_id → (a, b, c)

    for int_id, bid in enumerate(item_ids):
        biz = businesses.get(bid, {})
        cats = _extract_categories(biz)

        a = top_cat2a.get(cats[0], 0) if cats else 0
        b_vocab = sub_cat2b.get(a, {})
        b = b_vocab.get(cats[1], 0) if len(cats) > 1 else 0

        c = bucket_counters[(a, b)] % codebook_size
        bucket_counters[(a, b)] += 1

        sid_map[int_id] = (a, b, c)

    print(f"  Assigned {len(sid_map)} items to semantic ID buckets")
    print(f"  Unique (a,b) pairs: {len(bucket_counters)}")
    return sid_map


def sid_str(a, b, c):
    return f"<a_{a}><b_{b}><c_{c}>"


def sid_tokens(a, b, c):
    return [f"<a_{a}>", f"<b_{b}>", f"<c_{c}>"]


# ── Sequence splitting ────────────────────────────────────────────────────────


def build_user_sequences(df, item2int, user2int, max_seq_len):
    """
    Return dict: user_int → list of int item_ids in interaction order.
    Also keep a mapping user_int → original user_id for reference.
    """
    seqs = {}
    for uid_orig, grp in df.groupby("user_id"):
        uid = user2int[uid_orig]
        items = [item2int[b] for b in grp["business_id"]]
        # Keep most recent max_seq_len items
        seqs[uid] = items[-max_seq_len:] if len(items) > max_seq_len else items
    return seqs


# ── CSV row generation ────────────────────────────────────────────────────────


def _history_titles(history_ids, int2title):
    return [int2title[i] for i in history_ids]


def _history_sids(history_ids, sid_map):
    return [sid_str(*sid_map[i]) for i in history_ids]


def build_split_rows(seqs, int2title, sid_map, split):
    """
    Build DataFrame rows for the requested split.

    split='train': all steps except last 2 items per user
    split='valid': one row per user using second-to-last item as target
    split='test':  one row per user using last item as target
    """
    rows = []
    for uid, seq in seqs.items():
        if len(seq) < 3:
            continue  # need at least train + valid + test items

        if split == "train":
            # Generate one row per step in [1 .. len-2)
            for t in range(1, len(seq) - 2):
                hist = seq[:t]
                target = seq[t]
                rows.append(
                    {
                        "user_id": uid,
                        "history_item_title": str(_history_titles(hist, int2title)),
                        "item_title": int2title[target],
                        "history_item_id": str(hist),
                        "item_id": target,
                        "history_item_sid": str(_history_sids(hist, sid_map)),
                        "item_sid": sid_str(*sid_map[target]),
                    }
                )

        elif split == "valid":
            hist = seq[:-2]
            target = seq[-2]
            rows.append(
                {
                    "user_id": uid,
                    "history_item_title": str(_history_titles(hist, int2title)),
                    "item_title": int2title[target],
                    "history_item_id": str(hist),
                    "item_id": target,
                    "history_item_sid": str(_history_sids(hist, sid_map)),
                    "item_sid": sid_str(*sid_map[target]),
                }
            )

        elif split == "test":
            hist = seq[:-1]
            target = seq[-1]
            rows.append(
                {
                    "user_id": uid,
                    "history_item_title": str(_history_titles(hist, int2title)),
                    "item_title": int2title[target],
                    "history_item_id": str(hist),
                    "item_id": target,
                    "history_item_sid": str(_history_sids(hist, sid_map)),
                    "item_sid": sid_str(*sid_map[target]),
                }
            )

    return pd.DataFrame(rows)


# ── LLM-enriched placeholder generation ──────────────────────────────────────


def build_item_json(item_ids, businesses):
    """Build item.json: {str(int_id): {title, description, brand, categories}}."""
    data = {}
    for int_id, bid in enumerate(item_ids):
        biz = businesses.get(bid, {})
        name = biz.get("name", "Unknown Business")
        cats = biz.get("categories") or ""
        city = biz.get("city", "")
        state = biz.get("state", "")
        stars = biz.get("stars", "")
        review_count = biz.get("review_count", "")
        address = biz.get("address", "")

        description = (
            f"{name} is a business located at {address}, {city}, {state}. "
            f"It belongs to the following categories: {cats}. "
            f"It has an average rating of {stars} stars based on {review_count} reviews."
        )
        data[str(int_id)] = {
            "title": name,
            "description": description,
            "brand": "",
            "categories": cats,
        }
    return data


def build_item_enhanced_v2(item_ids, businesses, sid_map):
    """
    Build item_enhanced_v2.json.

    Adds 'llm_stage2' field: a paragraph that naturally weaves the SID
    tokens into a description of the business, mimicking the Amazon version.
    SidTextInterleaveDataset_v2 trains an LM objective on this text.
    """
    data = {}
    for int_id, bid in enumerate(item_ids):
        biz = businesses.get(bid, {})
        name = biz.get("name", "Unknown Business")
        cats = biz.get("categories") or "various services"
        city = biz.get("city", "")
        state = biz.get("state", "")
        stars = biz.get("stars", "")
        review_count = biz.get("review_count", "")
        address = biz.get("address", "")

        a, b, c = sid_map[int_id]
        sid = sid_str(a, b, c)

        description = (
            f"{name} is a business located at {address}, {city}, {state}. "
            f"It belongs to the following categories: {cats}. "
            f"It has an average rating of {stars} stars based on {review_count} reviews."
        )

        llm_stage2 = (
            f"{sid} refers to {name}, a well-known establishment in {city}, {state} "
            f"offering {cats}. With an average rating of {stars} stars from {review_count} "
            f"reviews, {sid} has established itself as a popular choice for customers "
            f"seeking quality experiences in its category. Whether you are looking for "
            f"convenience, ambiance, or excellent service, {sid} delivers on multiple fronts. "
            f"Customers frequently appreciate {name} for its consistency and the variety of "
            f"options it provides within the {cats} space. Located at {address}, {city}, "
            f"{state}, {sid} remains a go-to destination for local residents and visitors alike."
        )

        data[str(int_id)] = {
            "title": name,
            "description": description,
            "brand": "",
            "categories": cats,
            "index": [f"<a_{a}>", f"<b_{b}>", f"<c_{c}>"],
            "llm_enhanced": {
                "detailed_description": description,
                "use_cases": [f"Visiting {name} for {cats}."],
                "target_audience": "Local residents and visitors.",
                "key_features": [f"Located in {city}, {state}.", f"Categories: {cats}."],
                "related_keywords": [name, cats, city],
            },
            "llm_stage2": llm_stage2,
        }
    return data


def build_integrated_narrative_csv(train_df, int2title, sid_map):
    """
    Build integrated_narrative.csv.

    Adds an 'integrated_narrative' column: a passage weaving history SIDs
    with reasoning toward the target SID. Used by SidTextInterleaveSequenceDataset
    as a plain LM training signal.
    """
    rows = []
    for _, row in train_df.iterrows():
        hist_sids_raw = row["history_item_sid"]
        target_sid = row["item_sid"]
        target_title = row["item_title"]

        # Parse history SIDs
        try:
            hist_sids = ast.literal_eval(hist_sids_raw) if isinstance(hist_sids_raw, str) else hist_sids_raw
        except Exception:
            hist_sids = []

        hist_str = ", ".join(hist_sids) if hist_sids else "no prior interactions"

        narrative = (
            f"Reviewing the user's interaction sequence {hist_str}, a clear preference "
            f"pattern emerges. The user has consistently engaged with businesses that share "
            f"common traits in terms of category, location, and quality level. Based on "
            f"these interactions, the user's next choice is likely to follow the same pattern. "
            f"Considering the sequence {hist_str}, a natural continuation would be {target_sid}, "
            f"corresponding to {target_title}. This recommendation aligns with the user's "
            f"demonstrated preferences and provides a consistent next step in their "
            f"exploration journey. The transition to {target_sid} is supported by both "
            f"the categorical overlap and the quality signals evident in the prior interactions."
        )

        rows.append(
            {
                **row.to_dict(),
                "reasoning_path": (
                    f"Given the history {hist_str}, the user shows interest in similar businesses. "
                    f"The next item {target_sid} fits this pattern."
                ),
                "integrated_narrative": narrative,
            }
        )

    return pd.DataFrame(rows)


# ── Write helpers ─────────────────────────────────────────────────────────────


def write_info_file(path, item_ids, businesses, sid_map):
    """Write info/<DATASET>.txt: one line per item: <sid>\t<name>\t<int_id>"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for int_id, bid in enumerate(item_ids):
            biz = businesses.get(bid, {})
            name = biz.get("name", "Unknown").replace("\t", " ")
            a, b, c = sid_map[int_id]
            f.write(f"{sid_str(a, b, c)}\t{name}\t{int_id}\n")
    print(f"  Wrote {len(item_ids)} items → {path}")


# ── RL parquet creation ───────────────────────────────────────────────────────


def build_rl_parquet(df, category_name, split, out_path):
    """Convert a split CSV into the VERL parquet format for RL training."""
    import pandas as pd

    data_source = f"rec/{category_name}"
    rows = []
    for _, row in df.iterrows():
        hist_sids_raw = row["history_item_sid"]
        target_sid = row["item_sid"]

        instruction = (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the request.\n"
            "Can you recommend the next item for the user based on their interaction history?\n"
        )
        prompt_text = (
            f"The user has sequentially interacted with items {hist_sids_raw}. "
            "Can you recommend the next item for him? Let's think step by step before "
            "making recommendation. Directly output the item SID after thinking."
        )

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt_text},
        ]

        rows.append(
            {
                "data_source": data_source,
                "prompt": messages,
                "ability": "Recommendation",
                "reward_model": {"style": "rule", "ground_truth": target_sid},
                "extra_info": {
                    "split": split,
                    "index": len(rows),
                    "answer": target_sid,
                    "question": messages,
                },
            }
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"  Wrote {len(rows)} rows → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    out = Path(args.out_dir)

    # ── 1. Load & filter businesses ──
    businesses = load_businesses_from_tar(args.tar_path, args.category)

    # ── 2. Load reviews ──
    df = load_reviews_from_tar(args.tar_path, set(businesses.keys()))

    # ── 3. k-core filter ──
    df = kcore_filter(df, args.min_interactions)

    # ── 4. ID maps (built from full k-core dataset so all items are indexed) ──
    item2int, user2int, item_ids = build_id_maps(df, businesses)
    int2title = {i: businesses[bid].get("name", "Unknown") for i, bid in enumerate(item_ids)}
    n_items = len(item_ids)
    print(f"Full k-core dataset: {len(user2int):,} users, {n_items:,} items")

    # ── 5. Semantic IDs (assigned over all items) ──
    sid_map = assign_semantic_ids(item_ids, businesses)

    # ── 6. User subsampling (interactions only; item index files keep all items) ──
    import random as _random

    _random.seed(args.seed)
    all_users = list(user2int.keys())
    if args.max_users > 0 and args.max_users < len(all_users):
        sampled_users = set(_random.sample(all_users, args.max_users))
        df_interactions = df[df["user_id"].isin(sampled_users)].reset_index(drop=True)
        print(f"Subsampled to {len(sampled_users):,} users ({len(df_interactions):,} interactions)")
    else:
        df_interactions = df
        sampled_users = set(all_users)
    n_users = len(sampled_users)

    # ── 7. User sequences (from subsampled interactions) ──
    seqs = build_user_sequences(df_interactions, item2int, user2int, args.max_seq_len)

    # Dataset name
    category_slug = f"Yelp_{args.category.replace(' ', '_').replace('&', 'and')}"
    dataset_name = f"{category_slug}_{args.min_interactions}core"

    # ── 8. Build splits ──
    print("Building train/valid/test splits ...")
    train_df = build_split_rows(seqs, int2title, sid_map, "train")
    valid_df = build_split_rows(seqs, int2title, sid_map, "valid")
    test_df = build_split_rows(seqs, int2title, sid_map, "test")
    print(f"  train: {len(train_df):,}  valid: {len(valid_df):,}  test: {len(test_df):,}")

    # ── 8. Write CSVs ──
    for split, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        p = out / split / f"{dataset_name}.csv"
        os.makedirs(p.parent, exist_ok=True)
        split_df.to_csv(p, index=False)
        print(f"  Wrote {split} → {p}")

    # ── 9. Write info file ──
    info_path = out / "info" / f"{dataset_name}.txt"
    write_info_file(str(info_path), item_ids, businesses, sid_map)

    # ── 10. Write index files ──
    index_dir = out / "index"
    os.makedirs(index_dir, exist_ok=True)

    index_json = {str(i): sid_tokens(*sid_map[i]) for i in range(n_items)}
    with open(index_dir / f"{category_slug}.index.json", "w") as f:
        json.dump(index_json, f)
    print(f"  Wrote index.json ({n_items} items)")

    item_json = build_item_json(item_ids, businesses)
    with open(index_dir / f"{category_slug}.item.json", "w") as f:
        json.dump(item_json, f)
    print("  Wrote item.json")

    item_enhanced = build_item_enhanced_v2(item_ids, businesses, sid_map)
    with open(index_dir / f"{category_slug}.item_enhanced_v2.json", "w") as f:
        json.dump(item_enhanced, f)
    print("  Wrote item_enhanced_v2.json")

    # ── 11. Integrated narrative CSV ──
    narrative_df = build_integrated_narrative_csv(train_df, int2title, sid_map)
    narrative_path = index_dir / f"{category_slug}.integrated_narrative.csv"
    narrative_df.to_csv(narrative_path, index=False)
    print(f"  Wrote integrated_narrative.csv ({len(narrative_df):,} rows)")

    # ── 12. General reasoning symlink ──
    general_dir = out / "general"
    os.makedirs(general_dir, exist_ok=True)
    symlink_src = os.path.abspath(args.amazon_general_path)
    symlink_dst = general_dir / "sampled_data.arrow"
    if symlink_dst.exists() or symlink_dst.is_symlink():
        symlink_dst.unlink()
    if os.path.exists(symlink_src):
        os.symlink(symlink_src, symlink_dst)
        print(f"  Symlinked general data → {symlink_src}")
    else:
        print(
            f"  WARNING: Amazon general data not found at {symlink_src}. "
            f"Set --amazon_general_path or provide the file manually."
        )

    # ── 13. RL parquet files ──
    print("Building RL parquet files ...")
    rl_dir = out / "rec_reasoning_verl" / category_slug
    build_rl_parquet(train_df, category_slug, "train", str(rl_dir / "train.parquet"))
    build_rl_parquet(test_df, category_slug, "test", str(rl_dir / "test.parquet"))

    # ── 14. Print summary ──
    print()
    print("=" * 60)
    print("Dataset preparation complete!")
    print(f"  Dataset name : {dataset_name}")
    print(f"  Category slug: {category_slug}")
    print(f"  Users        : {n_users:,}")
    print(f"  Items        : {n_items:,}")
    print(f"  Train rows   : {len(train_df):,}")
    print(f"  Valid rows   : {len(valid_df):,}")
    print(f"  Test rows    : {len(test_df):,}")
    print()
    print("Next steps:")
    print(f"  1. sbatch scripts/sft_Qwen3_enrich_yelp.sh")
    print(f"  2. sbatch scripts/sft_reasoning_activation_yelp.sh")
    print(f"  3. sbatch scripts/RL_training_script_yelp.sh")
    print(f"  4. sbatch scripts/merge_fsdp_ckpt.sh <checkpoint_dir>")
    print(f"  5. sbatch scripts/evaluate_Qwen3_yelp.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
