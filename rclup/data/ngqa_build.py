"""Build NGQA-Interactive roots from the released NGQA CSV.

The native NGQA graph contains the user's health profile (statuses, dietary
habits) and the condition->tag match/contradict edges as *input*. In the
interactive adaptation the food subgraph is public evidence, the user's
profile is private to a simulated user, and the condition->tag guideline is
public domain knowledge (it is deterministic across the whole dataset, which
this script verifies). Gold answers are the native answers.

Root id: ngqa:<row>. Test roots are chosen by a hash of the *food name* so test
foods never appear in training. Training roots are partitioned per App. B.1.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import random

from ..common import DATA, REPO, bucket, partition, write_json, write_jsonl

STATUS_ORDER = [
    "obesity", "hypertension", "diabetes", "opioid_misuse",
    "Low fat/Low cholesterol diet", "Low carbohydrate diet", "Sugar free/Low sugar diet",
    "High protein diet", "Weight loss/Low calorie diet", "Low salt/Low sodium diet",
    "Diabetic diet", "Weight gain/Muscle building diet", "Renal/Kidney diet",
]


def parse_rows(path):
    csv.field_size_limit(10 ** 9)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def decompose(row):
    nodes = ast.literal_eval(row["node_list"])
    edges = ast.literal_eval(row["edge_list"])
    attr = {n[0]: n[1] for n in nodes}
    food = {
        "name": attr[1]["attr"], "fndds_code": attr[1]["name"],
        "category": next((attr[e[2]]["attr"] for e in edges if e[0] == 1 and attr[e[2]].get("name") == "category"), None),
        "ingredients": [attr[e[2]]["attr"] for e in edges if e[0] == 1 and attr[e[2]].get("name") == "ingredient"],
        "nutrition_tags": [attr[e[2]]["attr"] for e in edges if e[0] == 1 and attr[e[2]].get("name") == "food_nutrition_tag"],
    }
    statuses = [attr[e[2]]["attr"] for e in edges if e[0] == 0 and e[1] == "has" and attr[e[2]].get("name") == "status"]
    habits = [attr[e[2]]["attr"] for e in edges if e[0] == 0 and e[1] == "has" and attr[e[2]].get("name") == "dietary habit"]
    needs = [attr[e[2]]["attr"] for e in edges if attr[e[0]].get("name") == "status" and e[1] == "need"]
    relations = [(attr[e[0]]["attr"], e[1], attr[e[2]]["attr"]) for e in edges if e[1] in ("match", "contradict")]
    return food, {"user_id": attr[0]["name"], "statuses": statuses, "habits": habits, "needs": sorted(set(needs))}, relations


def build(csv_path, out_dir, test_share=20, seed=20261018):
    rows = parse_rows(csv_path)
    guideline = collections.defaultdict(dict)
    roots = []
    for index, row in enumerate(rows):
        food, profile, relations = decompose(row)
        for cond, rel, tag in relations:
            prev = guideline[cond].get(tag)
            if prev is not None and prev != rel:
                raise ValueError(f"Inconsistent guideline {cond}/{tag}: {prev} vs {rel}")
            guideline[cond][tag] = rel
        if any(s not in STATUS_ORDER for s in profile["statuses"]):
            raise ValueError(f"Unknown status in row {index}: {profile['statuses']}")
        root_id = f"ngqa:{index}"
        is_test = bucket("ngqa", "test-by-food", food["name"], modulus=100) < test_share
        split = "test" if is_test else partition("ngqa", root_id)
        # Opening disclosure: each status revealed with prob 0.5 by a root-seeded RNG,
        # plus up to three habits. The simulated user truthfully answers questions
        # about everything else.
        rng = random.Random(f"{seed}|{root_id}")
        revealed = [s for s in profile["statuses"] if rng.random() < 0.5]
        revealed_habits = rng.sample(profile["habits"], k=min(3, len(profile["habits"])))
        roots.append({
            "root_id": root_id, "dataset": "ngqa", "split": split,
            "difficulty": row["difficulty"],  # sparse/standard/complex in the paper
            "food": food,
            "questions": {"binary": row["question_easy"], "tags": row["question_medium"], "explain": row["question_hard"]},
            "labels": {"binary": row["answer_easy"].strip(), "tags": [t.strip() for t in row["answer_medium"].split(",") if t.strip()],
                       "explain": row["answer_hard"].strip()},
            "private_profile": profile,
            "opening_disclosure": {"statuses": revealed, "habits": revealed_habits},
            "relevant_relations": [list(r) for r in relations],
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "roots.jsonl", roots)
    write_json(out_dir / "guideline.json", {"status_order": STATUS_ORDER, "relations": {k: dict(v) for k, v in guideline.items()}})
    counts = collections.Counter((r["split"], r["difficulty"]) for r in roots)
    summary = {"roots": len(roots), "by_split_difficulty": {f"{s}|{d}": n for (s, d), n in sorted(counts.items())},
               "statuses": STATUS_ORDER, "test_share_percent": test_share, "foods": len({r["food"]["name"] for r in roots})}
    write_json(out_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(REPO / "data/raw/ngqa/NGQA_benchmark.csv"))
    p.add_argument("--out", default=str(DATA / "ngqa"))
    a = p.parse_args()
    from pathlib import Path
    print(build(a.csv, Path(a.out)))
