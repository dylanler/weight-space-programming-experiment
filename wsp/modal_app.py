"""Modal app: efficient Qwen3.5-2B inference (vLLM on A100) + weight-space editing kernel.

GPUs scale to zero automatically (scaledown_window) and every entrypoint is run with
`modal run`, which tears the app down when the call returns.
"""
import json
import os

import modal

APP_NAME = "wsp-qwen35"
BASE_MODEL = "Qwen/Qwen3.5-2B"
VOL_PATH = "/vol"
BASE_DIR = f"{VOL_PATH}/models/base"
EDIT_ROOT = f"{VOL_PATH}/models/edits"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("wsp-qwen35-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub[hf_transfer]", "safetensors", "numpy")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{VOL_PATH}/hf"})
)


def _model_dir(name: str) -> str:
    return BASE_DIR if name in ("", "base") else f"{EDIT_ROOT}/{name}"


@app.function(image=image, volumes={VOL_PATH: vol}, timeout=1800)
def download_base():
    from huggingface_hub import snapshot_download

    if not os.path.exists(f"{BASE_DIR}/config.json"):
        snapshot_download(BASE_MODEL, local_dir=BASE_DIR)
        vol.commit()
    return sorted(os.listdir(BASE_DIR))


@app.function(image=image, gpu="A100-40GB", volumes={VOL_PATH: vol}, timeout=3600, scaledown_window=60)
def generate(model_name: str, prompts: list[str], max_tokens: int = 4096, temperature: float = 0.7,
             top_p: float = 0.8, top_k: int = 20, presence_penalty: float = 1.5, enable_thinking: bool = False,
             seed: int = 0):
    """Batch chat generation. Defaults = Qwen3.5 recommended non-thinking sampling."""
    from vllm import LLM, SamplingParams

    vol.reload()
    path = _model_dir(model_name)
    llm = LLM(model=path, tokenizer=BASE_DIR, max_model_len=max_tokens + 1024, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=seed)
    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                        presence_penalty=presence_penalty, seed=seed)
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": enable_thinking})
    return [{"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
             "n_tokens": len(o.outputs[0].token_ids)} for o in outs]


@app.local_entrypoint()
def smoke_disabled():
    print(download_base.remote())
    r = generate.remote("base", ["Create an SVG of a building with 3 floors and 4 windows per floor. Output only the SVG code."], max_tokens=2048)
    print(json.dumps(r, indent=1)[:3000])
