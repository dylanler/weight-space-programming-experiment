"""Generate SVGs with a (base or edited) model on Modal, extract and render them to PNG.

Usage: python -m wsp.pipeline generate --model base --split test
"""
import argparse
import json
import os
import re

import cairosvg

SVG_RE = re.compile(r"<svg\b.*?</svg>", re.S | re.I)


def extract_svg(text: str):
    m = SVG_RE.search(text)
    return m.group(0) if m else None


def render(svg: str, out_png: str) -> bool:
    try:
        cairosvg.svg2png(bytestring=svg.encode(), write_to=out_png, output_width=512,
                         background_color="white", unsafe=False)
        return os.path.getsize(out_png) > 0
    except Exception:
        if os.path.exists(out_png):
            os.remove(out_png)
        return False


def load(split):
    with open(f"data/prompts_{split}.jsonl") as f:
        return [json.loads(l) for l in f]


def run_generate(model: str, split: str, max_tokens: int = 4096):
    """model = 'base' or a saved patch name; uses the same eager in-process engine as the editing kernel."""
    import modal
    gen = modal.Function.from_name("wsp-qwen35", "generate_patched")
    rows = load(split)
    outs = gen.remote(model, [r["prompt"] for r in rows], max_tokens=max_tokens)
    save_outputs(model, split, rows, outs)


def save_outputs(model, split, rows, outs):
    out_dir = f"results/{model}/{split}"
    os.makedirs(f"{out_dir}/png", exist_ok=True)
    os.makedirs(f"{out_dir}/svg", exist_ok=True)
    recs = []
    for r, o in zip(rows, outs):
        svg = extract_svg(o["text"])
        ok = False
        if svg:
            with open(f"{out_dir}/svg/{r['id']}.svg", "w") as f:
                f.write(svg)
            ok = render(svg, f"{out_dir}/png/{r['id']}.png")
        recs.append({**r, "raw": o["text"], "finish_reason": o["finish_reason"], "n_tokens": o["n_tokens"],
                     "has_svg": svg is not None, "rendered": ok})
    with open(f"{out_dir}/generations.jsonl", "w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    by = {}
    for x in recs:
        d = by.setdefault(x["task"], {"n": 0, "rendered": 0, "len_cap": 0, "tokens": 0})
        d["n"] += 1; d["rendered"] += x["rendered"]; d["len_cap"] += x["finish_reason"] == "length"; d["tokens"] += x["n_tokens"]
    print(json.dumps(by, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--model", default="base")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    if a.cmd == "generate":
        run_generate(a.model, a.split)
