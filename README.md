# Weight-space programming on Qwen3.5-2B (SVG tasks)

Can an LLM agent (GPT-6.2 Sol in the Devin harness) improve `Qwen/Qwen3.5-2B` on visual SVG-generation tasks by
*directly editing its weights* (no gradient training), following
[Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)?

- `wsp/tasks.py` – 5 SVG task families, 100 test prompts each (+30 dev prompts each for the editor)
- `wsp/modal_app.py` – vLLM inference on a Modal A100 (scale-to-zero)
- `wsp/kernel.py` – authenticated weight-editing kernel (in-process vLLM, live weights, sparse patch saving)
- `wsp/pipeline.py` – generation, SVG extraction, PNG rendering
- `results/` – generations, renders, verifier judgments, before/after diffs
- `blog/`, `paper/` – write-ups
