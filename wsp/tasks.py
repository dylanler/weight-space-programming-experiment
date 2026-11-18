"""Five SVG task families with randomized, visually verifiable parameters."""
import json
import random

SUFFIX = " Output only the complete SVG code, starting with <svg and ending with </svg>."


def building(rng):
    floors = rng.randint(1, 8)
    windows = rng.randint(1, 24)
    return ({"floors": floors, "windows": windows},
            f"Create an SVG image of a building with exactly {windows} windows in total and exactly {floors} floors.",
            f"A building with exactly {floors} floor(s) and exactly {windows} window(s) in total.")


def apple_tree(rng):
    apples = rng.randint(1, 15)
    return ({"apples": apples},
            f"Create an SVG image of a tree with exactly {apples} red apples on it.",
            f"A tree with exactly {apples} red apple(s) visible on it.")


def circle_grid(rng):
    rows, cols = rng.randint(1, 6), rng.randint(1, 6)
    return ({"rows": rows, "cols": cols},
            f"Create an SVG image of a grid of circles with exactly {rows} rows and {cols} columns.",
            f"A grid of circles with exactly {rows} row(s) and exactly {cols} column(s) ({rows*cols} circles total).")


def bar_chart(rng):
    n = rng.randint(3, 9)
    k = rng.randint(1, n)
    ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")
    return ({"bars": n, "tallest": k},
            f"Create an SVG image of a bar chart with exactly {n} vertical bars, where the {ordinal} bar from the left is the tallest and colored red, and all other bars are blue.",
            f"A bar chart with exactly {n} vertical bars; the {ordinal} bar from the left is the tallest bar and is red; all other bars are blue.")


def flag(rng):
    stripes = rng.randint(2, 9)
    stars = rng.randint(1, 9)
    return ({"stripes": stripes, "stars": stars},
            f"Create an SVG image of a flag with exactly {stripes} horizontal stripes and exactly {stars} stars.",
            f"A flag with exactly {stripes} horizontal stripe(s) and exactly {stars} star(s).")


TASKS = {"building": building, "apple_tree": apple_tree, "circle_grid": circle_grid,
         "bar_chart": bar_chart, "flag": flag}


def make_split(split: str, n: int):
    seed_base = {"test": 1000, "dev": 2000}[split]
    out = []
    for t_i, (name, fn) in enumerate(TASKS.items()):
        rng = random.Random(seed_base + t_i)
        for i in range(n):
            params, prompt, spec = fn(rng)
            out.append({"id": f"{split}-{name}-{i:03d}", "task": name, "params": params,
                        "prompt": prompt + SUFFIX, "spec": spec})
    return out


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    for split, n in [("test", 100), ("dev", 30)]:
        with open(f"data/prompts_{split}.jsonl", "w") as f:
            for r in make_split(split, n):
                f.write(json.dumps(r) + "\n")
    print("written")
