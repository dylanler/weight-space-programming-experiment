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

## Results (500 held-out prompts, image-only GPT-6.2 Sol judges)

| Task | Base correct | Edited correct |
|---|---|---|
| building | 4 | 2 |
| apple_tree | 11 | 33 |
| circle_grid | 38 | 58 |
| bar_chart | 35 | 54 |
| flag | 0 | 0 |
| **Total** | **88/500** | **147/500** |

Paired blinded re-judging: 78 → 120. The edit changes 0.073% of the weights (5 MLP `down_proj` matrices, a rank-1 projection that removes a "repetition-loop" direction).
Blog: https://weight-space-programming-qwen-dylan-team.vercel.app · Paper: `paper/main.pdf`
