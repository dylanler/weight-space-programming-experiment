"""Merge verifier judgments (anonymized) back to prompt ids and compute per-task scores.

Usage: python -m wsp.score <model> <split> <judgments_dir>
judgments_dir contains <task>.json files: {"judgments": [{"item_id","correct","observed"}]}
Writes results/<model>/<split>/judgments.jsonl and summary.json.
"""
import json
import os
import sys


def main(model, split, jdir):
    mapping = json.load(open(f"verify/{model}_{split}/mapping.json"))
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{model}/{split}/generations.jsonl")}
    judged = {}
    for fn in sorted(os.listdir(jdir)):
        if fn.endswith(".json"):
            for j in json.load(open(f"{jdir}/{fn}"))["judgments"]:
                judged[mapping[j["item_id"]]] = j
    summary = {}
    with open(f"results/{model}/{split}/judgments.jsonl", "w") as f:
        for rid, r in rows.items():
            j = judged.get(rid)
            correct = bool(j and j["correct"] and r["rendered"])
            f.write(json.dumps({"id": rid, "task": r["task"], "params": r["params"], "rendered": r["rendered"],
                                "finish_reason": r["finish_reason"], "judged": j is not None,
                                "correct": correct, "observed": j["observed"] if j else None}) + "\n")
            s = summary.setdefault(r["task"], {"n": 0, "rendered": 0, "correct": 0, "unjudged": 0})
            s["n"] += 1; s["rendered"] += r["rendered"]; s["correct"] += correct; s["unjudged"] += j is None
    tot = {k: sum(v[k] for v in summary.values()) for k in ("n", "rendered", "correct", "unjudged")}
    summary["TOTAL"] = tot
    json.dump(summary, open(f"results/{model}/{split}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
