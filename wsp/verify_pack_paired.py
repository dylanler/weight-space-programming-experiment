"""Robustness check: one blinded pack per task mixing base and edited renders (200 items),
so a single judge scores both conditions under identical standards.
Usage: python -m wsp.verify_pack_paired base final -> verify/paired_test/<task>.zip + mapping.json"""
import hashlib
import json
import os
import sys
import zipfile


def main(a, b):
    out = "verify/paired_test"
    os.makedirs(out, exist_ok=True)
    mapping, by_task = {}, {}
    for m in (a, b):
        for l in open(f"results/{m}/test/generations.jsonl"):
            r = json.loads(l)
            anon = hashlib.sha256(f"paired|{m}|{r['id']}|wsp-salt".encode()).hexdigest()[:10]
            mapping[anon] = [m, r["id"]]
            by_task.setdefault(r["task"], []).append((anon, m, r))
    for task, items in by_task.items():
        items.sort(key=lambda x: x[0])
        manifest = []
        with zipfile.ZipFile(f"{out}/{task}.zip", "w") as z:
            for anon, m, r in items:
                png = f"results/{m}/test/png/{r['id']}.png"
                has = r["rendered"] and os.path.exists(png)
                if has:
                    z.write(png, f"images/{anon}.png")
                manifest.append({"item_id": anon, "image": f"images/{anon}.png" if has else None,
                                 "requirement": r["spec"]})
            z.writestr("manifest.json", json.dumps(manifest, indent=1))
    json.dump(mapping, open(f"{out}/mapping.json", "w"), indent=1)
    print({t: len(v) for t, v in by_task.items()})


if __name__ == "__main__":
    main(*sys.argv[1:3])
