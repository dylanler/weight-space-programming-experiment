"""Package rendered PNGs for image-only verification (no SVG code is ever included).

Items are anonymized (random ids) so the verifier cannot tell which model / prompt index produced them.
Usage: python -m wsp.verify_pack <model> <split>  -> verify/<model>_<split>/<task>.zip + mapping.json
"""
import hashlib
import json
import os
import sys
import zipfile


def main(model, split):
    src = f"results/{model}/{split}"
    out = f"verify/{model}_{split}"
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(f"{src}/generations.jsonl")]
    mapping, by_task = {}, {}
    for r in rows:
        anon = hashlib.sha256(f"{model}|{r['id']}|wsp-salt".encode()).hexdigest()[:10]
        mapping[anon] = r["id"]
        by_task.setdefault(r["task"], []).append((anon, r))
    for task, items in by_task.items():
        items.sort(key=lambda x: x[0])
        manifest = []
        with zipfile.ZipFile(f"{out}/{task}.zip", "w") as z:
            for anon, r in items:
                png = f"{src}/png/{r['id']}.png"
                has = r["rendered"] and os.path.exists(png)
                if has:
                    z.write(png, f"images/{anon}.png")
                manifest.append({"item_id": anon, "image": f"images/{anon}.png" if has else None,
                                 "requirement": r["spec"]})
            z.writestr("manifest.json", json.dumps(manifest, indent=1))
    json.dump(mapping, open(f"{out}/mapping.json", "w"), indent=1)
    print(out, {t: len(v) for t, v in by_task.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
