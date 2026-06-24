"""
Convert SIDReasoner Amazon data to original SASrec format.

Strategy: use the last test row's history_item_id as the full chronological
interaction sequence. This correctly includes all training + validation items
(including intermediate valid interactions that groupby-last on valid CSV would miss).

Item IDs are remapped to a dense range [1, n_unique_items] so that the SASRec
item embedding matrix only covers items actually seen in the data (not sparse
original IDs that may go up to 3x the actual item count).

Output: data/<dataset>.txt  with rows "user_id item_id" (1-indexed integers,
chronological order per user).
"""

import argparse
import ast
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", required=True, help="Path to SIDReasoner Amazon data dir (e.g. /path/to/SIDReasoner-UvA/data/Amazon)"
)
args = parser.parse_args()
DATA_DIR = args.data_dir

DATASETS = [
    "Office_Products_5_2016-10-2018-11",
    "Industrial_and_Scientific_5_2016-10-2018-11",
    "Video_Games_5_2016-10-2018-11",
]

for dataset in DATASETS:
    train_df = pd.read_csv(f"{DATA_DIR}/train/{dataset}.csv")
    valid_df = pd.read_csv(f"{DATA_DIR}/valid/{dataset}.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test/{dataset}.csv")

    # Only keep users present in all three splits
    common = set(train_df.user_id) & set(valid_df.user_id) & set(test_df.user_id)
    test_df = test_df[test_df.user_id.isin(common)]

    # Last test row per user: its history_item_id is the full chronological
    # interaction history up to (but not including) the test target.
    last_test = test_df.groupby("user_id").last().reset_index()

    user_ids = sorted(last_test["user_id"].unique())
    user_map = {uid: i + 1 for i, uid in enumerate(user_ids)}

    # First pass: collect all non-zero item IDs to build a dense mapping.
    # Original ID 0 is a padding placeholder in history_item_id — skip it.
    raw_items = []
    for _, row in last_test.iterrows():
        history = ast.literal_eval(row["history_item_id"])
        test_item = int(row["item_id"])
        for iid in history + [test_item]:
            if iid != 0:
                raw_items.append(iid)

    unique_items = sorted(set(raw_items))
    item_map = {iid: idx + 1 for idx, iid in enumerate(unique_items)}  # dense 1-indexed

    # Second pass: write interactions using remapped IDs.
    rows = []
    for _, row in last_test.iterrows():
        uid = user_map[row["user_id"]]
        history = ast.literal_eval(row["history_item_id"])
        test_item = int(row["item_id"])
        for iid in history + [test_item]:
            if iid != 0:
                rows.append((uid, item_map[iid]))

    out_path = f"data/{dataset}.txt"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        for uid, iid in rows:
            f.write(f"{uid} {iid}\n")

    n_users = len(user_ids)
    n_items = len(unique_items)
    n_interactions = len(rows)
    print(f"{dataset}: {n_users} users, {n_items} unique items, {n_interactions} interactions -> {out_path}")
