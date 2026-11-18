"""Deploy blog/ (static) to Vercel via REST API. Requires VERCEL_TOKEN env var."""
import base64
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
SKIP = {"deploy_vercel.py"}


def main(name="weight-space-programming-qwen"):
    files = []
    for dp, _, fns in os.walk(ROOT):
        for fn in fns:
            if fn in SKIP or fn.startswith("."):
                continue
            p = os.path.join(dp, fn)
            files.append({"file": os.path.relpath(p, ROOT),
                          "data": base64.b64encode(open(p, "rb").read()).decode(), "encoding": "base64"})
    body = {"name": name, "files": files, "target": "production",
            "projectSettings": {"framework": None, "buildCommand": None, "outputDirectory": None}}
    req = urllib.request.Request("https://api.vercel.com/v13/deployments?skipAutoDetectionConfirmation=1",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {os.environ['VERCEL_TOKEN']}",
                                          "Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        print(e.code, e.read().decode()[:500]); sys.exit(1)
    print(json.dumps({"id": r.get("id"), "url": r.get("url"), "alias": r.get("alias"),
                      "readyState": r.get("readyState")}))


if __name__ == "__main__":
    main(*sys.argv[1:])
