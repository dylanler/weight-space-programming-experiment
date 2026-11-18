"""Weight-space programming kernel: a stateful A100 container running vLLM *in-process*
(eager mode) so an editor agent can read/modify the live weights with Python and
immediately sample from the edited model.

Exposed as an authenticated HTTP API (Bearer key from Modal secret `wsp-kernel-key`).
Edits are persisted as sparse patches (param name -> (index, new values)) relative to base.
"""
import base64
import contextlib
import io
import json
import os
import re
import traceback
import zipfile

import modal

from wsp.modal_app import BASE_DIR, VOL_PATH, app, image, vol

kimage = (image.apt_install("libcairo2").pip_install("fastapi[standard]", "cairosvg")
          .add_local_dir("data", "/root/data").add_local_python_source("wsp"))
PATCH_DIR = f"{VOL_PATH}/patches"
GEN_BUDGET = 40  # max number of generation batches (each <= 160 prompts) per editing run
SVG_RE = re.compile(r"<svg\b.*?</svg>", re.S | re.I)


def _render(svg):
    import cairosvg
    try:
        return cairosvg.svg2png(bytestring=svg.encode(), output_width=512, background_color="white")
    except Exception:
        return None


def build_engine():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM
    llm = LLM(model=BASE_DIR, max_model_len=5120, gpu_memory_utilization=0.6, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    return llm


def get_torch_model(llm):
    ex = llm.llm_engine.model_executor
    for path in ("driver_worker.worker.model_runner.model", "driver_worker.model_runner.model"):
        obj = ex
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate in-process model")


def chat_generate(llm, prompts, max_tokens=4096, temperature=0.0):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=1.0 if temperature == 0 else 0.8,
                        top_k=-1 if temperature == 0 else 20, seed=0)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp,
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    return [{"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
             "n_tokens": len(o.outputs[0].token_ids)} for o in outs]


def compute_patch(model, base):
    import torch
    patch = {}
    for n, p in model.named_parameters():
        b = base[n]
        diff = (p.data.detach().cpu() != b)
        if diff.any():
            idx = diff.nonzero(as_tuple=False)
            patch[n] = {"idx": idx, "val": p.data.detach().cpu()[diff].clone()}
    return patch


def apply_patch(model, patch):
    import torch
    params = dict(model.named_parameters())
    for n, d in patch.items():
        p = params[n]
        idx = d["idx"].to(p.device)
        p.data[tuple(idx.T)] = d["val"].to(p.device, p.dtype)


@app.cls(image=kimage, gpu="A100-40GB", volumes={VOL_PATH: vol}, timeout=6 * 3600, scaledown_window=900,
         max_containers=1, secrets=[modal.Secret.from_name("wsp-kernel-key")])
@modal.concurrent(max_inputs=1)
class Kernel:
    @modal.enter()
    def setup(self):
        import torch
        self.llm = build_engine()
        self.model = get_torch_model(self.llm)
        self.base = {n: p.data.detach().cpu().clone() for n, p in self.model.named_parameters()}
        self.gen_calls = 0
        self.dev = [json.loads(l) for l in open("/root/data/prompts_dev.jsonl")] if os.path.exists("/root/data/prompts_dev.jsonl") else []
        k = self

        def gen(prompts, max_tokens=4096, temperature=0.0):
            if len(prompts) > 160:
                raise ValueError("max 160 prompts per generation batch")
            if k.gen_calls >= GEN_BUDGET:
                raise RuntimeError(f"generation budget exhausted ({GEN_BUDGET} batches)")
            k.gen_calls += 1
            return chat_generate(k.llm, prompts, max_tokens, temperature)

        def reset_weights():
            for n, p in k.model.named_parameters():
                p.data.copy_(k.base[n].to(p.device))

        def changed_weights():
            patch = compute_patch(k.model, k.base)
            total = sum(p.numel() for p in k.model.parameters())
            ch = sum(int(d["val"].numel()) for d in patch.values())
            return {"changed": ch, "total": total, "fraction": ch / total, "tensors": {n: int(d["val"].numel()) for n, d in patch.items()}}

        def tokenize(text):
            return k.llm.get_tokenizer()(text)["input_ids"]

        self.ns = {"torch": torch, "llm": self.llm, "model": self.model, "base": self.base, "gen": gen,
                   "dev_prompts": self.dev, "reset_weights": reset_weights, "changed_weights": changed_weights,
                   "tokenizer": self.llm.get_tokenizer(), "tokenize": tokenize, "render_png": _render,
                   "extract_svg": lambda t: (m.group(0) if (m := SVG_RE.search(t)) else None)}

    def _auth(self, req):
        if req.headers.get("authorization", "") != f"Bearer {os.environ['WSP_KERNEL_KEY']}":
            from fastapi import HTTPException
            raise HTTPException(401)

    @modal.asgi_app()
    def api(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import Response
        web = FastAPI()

        @web.post("/exec")
        async def exec_code(req: Request):
            self._auth(req)
            body = await req.json()
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(body["code"], self.ns)
                err = None
            except Exception:
                err = traceback.format_exc()
            return {"stdout": buf.getvalue()[-20000:], "error": err, "gen_calls_used": self.gen_calls,
                    "gen_budget": GEN_BUDGET}

        @web.post("/eval_dev")
        async def eval_dev(req: Request):
            """Generate the dev set (optionally a subset of tasks) with CURRENT weights; returns a zip of
            PNG renders + generations.jsonl. Counts as one generation batch."""
            self._auth(req)
            body = await req.json()
            tasks = body.get("tasks")
            rows = [r for r in self.dev if not tasks or r["task"] in tasks][:160]
            outs = self.ns["gen"]([r["prompt"] for r in rows])
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w") as z:
                lines = []
                for r, o in zip(rows, outs):
                    svg = self.ns["extract_svg"](o["text"])
                    png = _render(svg) if svg else None
                    if png:
                        z.writestr(f"png/{r['id']}.png", png)
                    lines.append(json.dumps({**r, **o, "has_svg": svg is not None, "rendered": png is not None}))
                z.writestr("generations.jsonl", "\n".join(lines))
            return Response(zbuf.getvalue(), media_type="application/zip")

        @web.post("/save_patch")
        async def save_patch(req: Request):
            import torch
            self._auth(req)
            name = (await req.json())["name"]
            patch = compute_patch(self.model, self.base)
            os.makedirs(PATCH_DIR, exist_ok=True)
            torch.save(patch, f"{PATCH_DIR}/{name}.pt")
            vol.commit()
            return self.ns["changed_weights"]()

        @web.get("/status")
        async def status(req: Request):
            self._auth(req)
            return {"gen_calls_used": self.gen_calls, "gen_budget": GEN_BUDGET, **self.ns["changed_weights"]()}

        return web


@app.function(image=kimage, gpu="A100-40GB", volumes={VOL_PATH: vol}, timeout=3600, scaledown_window=60)
def generate_patched(patch_name: str, prompts: list[str], max_tokens: int = 4096):
    """Final evaluation: base weights + saved patch, greedy decoding (same engine config as kernel)."""
    import torch
    vol.reload()
    llm = build_engine()
    model = get_torch_model(llm)
    if patch_name not in ("", "base"):
        apply_patch(model, torch.load(f"{PATCH_DIR}/{patch_name}.pt"))
    return chat_generate(llm, prompts, max_tokens)
