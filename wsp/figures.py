"""Make blog/paper figures: score bar chart and before/after example sheets.
Usage: python -m wsp.figures base [edited]"""
import json
import os
import sys

from PIL import Image, ImageDraw

TASKS = ["building", "apple_tree", "circle_grid", "bar_chart", "flag"]


def load(model):
    return {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{model}/test/judgments.jsonl")}


def sheet(models, out, per_task=4, cell=200):
    js = {m: load(m) for m in models}
    W = cell * per_task * len(models)
    H = (cell + 40) * len(TASKS)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for ti, t in enumerate(TASKS):
        ids = sorted(i for i in js[models[0]] if js[models[0]][i]["task"] == t)[:per_task]
        for mi, m in enumerate(models):
            for k, rid in enumerate(ids):
                x, y = (mi * per_task + k) * cell, ti * (cell + 40)
                p = f"results/{m}/test/png/{rid}.png"
                if os.path.exists(p):
                    im = Image.open(p).convert("RGB")
                    im.thumbnail((cell - 8, cell - 8))
                    img.paste(im, (x + 4, y + 4))
                else:
                    d.text((x + 60, y + 90), "no image", fill="gray")
                j = js[m][rid]
                d.text((x + 4, y + cell + 2), f"{m}: {'OK' if j['correct'] else 'X'}", fill="green" if j["correct"] else "red")
                d.text((x + 4, y + cell + 16), json.dumps(j["params"])[:34], fill="black")
    img.save(out)


def scores(models, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    s = {m: json.load(open(f"results/{m}/test/summary.json")) for m in models}
    x = np.arange(len(TASKS))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, m in enumerate(models):
        v = [s[m][t]["correct"] for t in TASKS]
        b = ax.bar(x + i * w, v, w, label=m)
        ax.bar_label(b, fontsize=8)
    ax.set_xticks(x + w * (len(models) - 1) / 2, TASKS)
    ax.set_ylabel("visually correct / 100")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    models = sys.argv[1:]
    os.makedirs("blog/img", exist_ok=True)
    scores(models, "blog/img/scores.png")
    sheet(models, "blog/img/examples.png")
    print("ok")
