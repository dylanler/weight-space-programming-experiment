"""Score the blinded paired judging (base + edited mixed per task, same judge) with an exact McNemar test.
Usage: python -m wsp.score_paired -> results/paired_summary.json"""
import json
import os
from math import comb


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    mapping = json.load(open("verify/paired_test/mapping.json"))
    rendered = {}
    for m in ("base", "final"):
        for l in open(f"results/{m}/test/generations.jsonl"):
            r = json.loads(l)
            rendered[(m, r["id"])] = (r["task"], r["rendered"])
    J = {}
    jdir = "verify/paired_test/judgments"
    for fn in os.listdir(jdir):
        for j in json.load(open(f"{jdir}/{fn}"))["judgments"]:
            m, rid = mapping[j["item_id"]]
            J[(m, rid)] = bool(j["correct"]) and rendered[(m, rid)][1]
    out = {}
    ids = sorted({rid for (_, rid) in rendered})
    for t in ["building", "apple_tree", "circle_grid", "bar_chart", "flag", "TOTAL"]:
        sel = [i for i in ids if t == "TOTAL" or rendered[("base", i)][0] == t]
        a = sum(J.get(("base", i), False) for i in sel)
        e = sum(J.get(("final", i), False) for i in sel)
        fixed = sum(J.get(("final", i), False) and not J.get(("base", i), False) for i in sel)
        broke = sum(J.get(("base", i), False) and not J.get(("final", i), False) for i in sel)
        out[t] = {"n": len(sel), "base_correct": a, "edit_correct": e, "delta": e - a,
                  "fixed": fixed, "broke": broke, "mcnemar_p": mcnemar_p(fixed, broke)}
    json.dump(out, open("results/paired_summary.json", "w"), indent=1)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
