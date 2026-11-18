"""Agreement between independent judges (separate sessions) and paired-blind judges on the same images.
Usage: python -m wsp.agreement -> results/judge_agreement.json"""
import json
import os


def load_single(m):
    return {json.loads(l)["id"]: json.loads(l)["correct"] for l in open(f"results/{m}/test/judgments.jsonl")}


def load_paired():
    mapping = json.load(open("verify/paired_test/mapping.json"))
    rendered = {}
    for m in ("base", "final"):
        for l in open(f"results/{m}/test/generations.jsonl"):
            r = json.loads(l)
            rendered[(m, r["id"])] = (r["task"], r["rendered"])
    out = {}
    for fn in os.listdir("verify/paired_test/judgments"):
        for j in json.load(open(f"verify/paired_test/judgments/{fn}"))["judgments"]:
            k = tuple(mapping[j["item_id"]])
            out[k] = bool(j["correct"]) and rendered[k][1]
    return out, rendered


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    P, rendered = load_paired()
    res = {}
    for t in ["building", "apple_tree", "circle_grid", "bar_chart", "flag", "ALL"]:
        x, y = [], []
        for m in ("base", "final"):
            S = load_single(m)
            for rid, c in S.items():
                if rendered[(m, rid)][1] and (t == "ALL" or rendered[(m, rid)][0] == t):
                    x.append(c); y.append(P[(m, rid)])
        po, k = kappa(x, y)
        res[t] = {"n_rendered_items": len(x), "agreement": round(po, 3), "cohen_kappa": round(k, 3)}
        print(t, res[t])
    json.dump(res, open("results/judge_agreement.json", "w"), indent=1)


if __name__ == "__main__":
    main()
