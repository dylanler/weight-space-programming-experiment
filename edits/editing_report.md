# Weight-space programming on Qwen3.5-2B: SVG drawing tasks

**Result.** The final edit, saved as patch `final`, fixes generation loops. On the 150 dev prompts, rendered images went from 124/150 to **150/150**. Visually correct images went from **27/150 to 46/150**. The edit changes **1,370,750 weights (0.0728% of 1,881,825,088)**. All of them sit in five MLP `down_proj` matrices (layers 15, 17, 19, 21, 23). No hooks stay installed. There is no training, and prompts and decoding are unchanged.

## 1. Hypotheses
- H1 (from the blog post): repetition loops come from a residual-stream "keep repeating" direction written by mid-to-late MLPs. If that direction is removed from the output columns of the MLP neurons that write it most, the model should be able to close the SVG.
- H2: the decision to loop is made at *element boundaries*. These are the positions where the model picks between opening another element (` <`) and closing (` </`, `</svg>`). A direction measured only at those positions should be more specific, and do less collateral damage, than one averaged over all tokens.
- H3: after one round of edits, the loops that remain (mostly "arithmetic-progression" rectangle loops in `building`) use a partly different direction. Repeating the measurement on the edited model should find it.

## 2. Analysis (activations and weights, no generation)
- I loaded an HF copy of the base weights on the same GPU and ran teacher-forced forward passes over the dev generations. Forward hooks were used only for analysis and were removed in `finally` blocks (for example `a7.py`). The final model has no hooks.
- Labelling: loop onset is the first repeated normalized SVG line in a `length`-capped generation. Tokens after onset are labelled loop (L); tokens of finished generations are labelled normal (N).
- The per-layer mean residual difference L−N is small in early layers and grows steadily with depth, with the largest differences in layers 15–23. I therefore targeted late odd layers (15–23) and treated early layers (11, 13) as an ablation (E3).
- Delimiter direction `dirsD[l]` = unit(mean residual of L − mean residual of N), measured only at positions where the next token opens or closes an SVG element (H2).
- Neuron selection: for each layer, I ranked MLP neurons by how strongly their `down_proj` column writes along u (|u·W_down[:,j]| weighted by activation magnitude) and took the top k (128, 64 or 32).
- Round-2 direction `dirsD2`: I recomputed the same delimiter statistic on the **E2-edited** model using E2's own dev outputs (123 loop-context tokens, mostly building loops). Cosine with round 1 is 0.64 at layer 12 and falls to 0.19 at layer 23. The remaining loops therefore use a largely new late-layer direction (H3).
- Teacher-forced checks for each edit. "Loop log p(close)" is the log-probability of closing inside already-looping text; lower means the loop context is disrupted more. The other columns are NLL at loop onset, NLL on normal text, and greedy agreement on normal text:

| edit | loop log p(close) | onset NLL | normal NLL | normal greedy agree |
|---|---|---|---|---|
| base | −12.86 | 0.042 | 0.058 | 0.989 |
| E1 | −13.66 | 0.039 | 0.060 | 0.986 |
| E2 | −14.13 | 0.040 | 0.060 | 0.987 |
| E3 | −14.04 | 0.040 | 0.060 | 0.987 |
| E4 | −14.87 | 0.042 | 0.061 | 0.987 |
| E5 | −14.77 | 0.041 | 0.061 | 0.987 |

  Normal-text NLL rises by only about 4%. This screen was used to keep edits harmless before spending a generation batch on them.

## 3. Edit operation
For each selected layer l and neuron set idx: `W_down[:, idx] -= u uᵀ W_down[:, idx]`, with u a unit vector in R^2048. This is a rank-1 projection that removes u from what those neurons can write into the residual stream. It is applied identically to the vLLM tensor (and to the HF analysis copy).

## 4. Edits attempted (dev set, 30 prompts per task)
Each cell is **rendered / visually correct**. Judging rules, applied the same way to every run after looking at each PNG (zoomed contact sheets):
- building: both window count and floor count exact.
- apple_tree: exactly N red apples on a tree. A bare green square is wrong, and merged or overlapping blobs are wrong.
- circle_grid: all R×C circles visible. Clipping more than half of a circle is wrong.
- bar_chart: exactly N bars, k-th red and tallest, others blue. A cut-off bar is wrong.
- flag: S stripes and M distinct star marks. Dot-shaped stars are accepted.

| edit | description | #weights | building | apple | circle | bar | flag | total |
|---|---|---|---|---|---|---|---|---|
| base | – | 0 | 21 / 4 | 14 / 0 | 30 / 7 | 30 / 13 | 29 / 3 | **124 / 27** |
| E1 | dirsD, L15,17,19,21, top-64 | 503,134 | 21 / 3 | 30 / 12 | 30 / 5 | 30 / 9 | 29 / 2 | 140 / 31 |
| E2 | dirsD, L15,17,19,21,23, top-128 | 1,253,339 | 25 / 3 | 30 / 20 | 30 / 5 | 30 / 14 | 29 / 2 | 144 / 44 |
| E3 | dirsD, L11,13,…,23, top-128 | 1,752,892 | 24 / 2 | 30 / 12 | 30 / 6 | 30 / 13 | 29 / 2 | 143 / 35 |
| E4 | E2 + dirsD2, same layers, top-64 | 1,622,668 | 30 / 3 | 28 / 11 | 30 / 8 | 30 / 13 | 28 / 4 | 146 / 39 |
| **E5 (final)** | E2 + dirsD2, same layers, top-32 | **1,370,750** | **30 / 2** | **30 / 10** | **30 / 10** | **30 / 19** | **30 / 5** | **150 / 46** |

Teacher-forced check on normal text for the final edit: NLL 0.0582 → 0.0606 and greedy agreement 0.989 → 0.987. All 150 outputs are still valid SVG that renders.

## 5. What worked and what didn't
- **Worked:** removing the delimiter-position loop direction from the top MLP writers in odd layers 15–23. One round (E2) removed every apple-tree loop (16 → 0). A second, re-measured round (E5) removed the remaining building and flag loops (9 → 0 and 1 → 0). Bar-chart and circle-grid correctness rose as a side effect: outputs got more compact and less self-repeating.
- **Partly worked:** making the edit larger or wider (E3 adds layers 11 and 13; E4 uses top-64 in round 2). Loops were reduced about as much, but correctness was lower. E4 brought 2 apple and 2 flag loops back. Smaller round-2 edits were better.
- **Didn't work / not pursued:** fixing the counts themselves: building window/floor arithmetic and apple counting remain the main errors. Counts move around between edits (apple correct ranged 10–20), which suggests the correctness counts per edit carry about ±4 of noise per task. The loop-removal gain (+26 rendered) is the robust effect.
- **Caveat:** with only 30 dev prompts per task and greedy decoding, correctness differences of a few images between edits are not significant. E5 was chosen for its 150/150 completion plus the highest total.

## 6. Final edit
- Patch `final` = E5: 5 tensors, `language_model.model.layers.{15,17,19,21,23}.mlp.down_proj.weight`. For each layer: 128 columns (round 1) plus 32 columns (round 2), with possible overlap.
- Changed weights: 1,370,750 / 1,881,825,088 = 0.0728%. Per tensor: 280,546 / 281,112 / 266,732 / 266,888 / 275,472.
- `final_edit.py` embeds the 10 direction vectors and the neuron indices, starts with `reset_weights()`, and reproduces the patch exactly. I checked this with a SHA-256 of the edited tensors against the E5 state: MATCH True.

## 7. Budget and time
- Generation batches: 6 `/eval_dev` calls by my count (base, E1–E5), and no `gen()` calls. The kernel counter reports `gen_calls_used: 5` of 40.
- Wall-clock time: about 42 minutes by the VM clock, from session start (01:36 UTC) to the final `/status` (02:18 UTC).

## 8. Final /status
```
{"gen_calls_used":5,"gen_budget":40,"changed":1370750,"total":1881825088,"fraction":0.0007284152011475362,"tensors":{"language_model.model.layers.15.mlp.down_proj.weight":280546,"language_model.model.layers.17.mlp.down_proj.weight":281112,"language_model.model.layers.19.mlp.down_proj.weight":266732,"language_model.model.layers.21.mlp.down_proj.weight":266888,"language_model.model.layers.23.mlp.down_proj.weight":275472}}
```
