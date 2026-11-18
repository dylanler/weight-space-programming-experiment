"""Write paper/results_table.tex and results/comparison.json from summaries. Usage: python -m wsp.tables base final"""
import json
import sys

TASKS = ["building", "apple_tree", "circle_grid", "bar_chart", "flag"]
NAMES = {"building": "Building (windows, floors)", "apple_tree": "Apple tree (apples)",
         "circle_grid": "Circle grid (rows, cols)", "bar_chart": "Bar chart (n, tallest)", "flag": "Flag (stripes, stars)"}


def main(a, b):
    sa = json.load(open(f"results/{a}/test/summary.json"))
    sb = json.load(open(f"results/{b}/test/summary.json"))
    ja = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{a}/test/judgments.jsonl")}
    jb = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{b}/test/judgments.jsonl")}
    comp = {}
    rows = []
    for t in TASKS + ["TOTAL"]:
        ids = [i for i in ja if t == "TOTAL" or ja[i]["task"] == t]
        fixed = sum(1 for i in ids if jb[i]["correct"] and not ja[i]["correct"])
        broke = sum(1 for i in ids if ja[i]["correct"] and not jb[i]["correct"])
        comp[t] = {"base_rendered": sa[t]["rendered"], "edit_rendered": sb[t]["rendered"],
                   "base_correct": sa[t]["correct"], "edit_correct": sb[t]["correct"],
                   "delta": sb[t]["correct"] - sa[t]["correct"], "fixed": fixed, "broke": broke, "n": sa[t]["n"]}
        c = comp[t]
        name = "\\textbf{Total}" if t == "TOTAL" else NAMES[t]
        rows.append(f"{name} & {c['base_rendered']} & {c['edit_rendered']} & {c['base_correct']}/{c['n']} & "
                    f"{c['edit_correct']}/{c['n']} & {c['delta']:+d} & {fixed} & {broke} \\\\")
    json.dump(comp, open("results/comparison.json", "w"), indent=1)
    tex = ("\\begin{tabular}{lrrrrrrr}\\toprule\n"
           "Task & \\multicolumn{2}{c}{Rendered} & \\multicolumn{2}{c}{Visually correct} & $\\Delta$ & Fixed & Broken\\\\\n"
           "& base & edit & base & edit & & & \\\\\\midrule\n" + "\n".join(rows[:-1]) + "\n\\midrule\n" + rows[-1] +
           "\n\\bottomrule\\end{tabular}\n")
    open("paper/results_table.tex", "w").write(tex)
    print(json.dumps(comp, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
