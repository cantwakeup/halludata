import csv, argparse
from pathlib import Path
from collections import defaultdict

def f(x):
    return float(x) if x not in ("", None) else 0.0

ap = argparse.ArgumentParser()
ap.add_argument("--summary", required=True)
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()

rows = list(csv.DictReader(open(args.summary, encoding="utf-8")))
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

base = {(r["Dataset"], r["Setting"]): r for r in rows if r["Method"] == "Regular"}
cats = [r for r in rows if r["Method"] == "CatExpert" and int(float(r["N"])) == 3000]

groups = defaultdict(list)
for r in cats:
    groups[(r["Dataset"], r["Setting"])].append(r)

# 固定顺序：MSCOCO 三个 + GQA 三个
order = [
    ("MSCOCO", "random"),
    ("MSCOCO", "popular"),
    ("MSCOCO", "adversarial"),
    ("GQA", "random"),
    ("GQA", "popular"),
    ("GQA", "adversarial"),
]
order = [k for k in order if k in groups]

import matplotlib.pyplot as plt

metrics = ["Accuracy", "F1 Score", "Precision", "Recall", "Yes Rate", "FP", "FN"]

for metric in metrics:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    axes = axes.ravel()

    for ax, key in zip(axes, order):
        dataset, setting = key
        g = sorted(groups[key], key=lambda r: f(r["Alpha"]))
        b = base[key]

        xs = [f(r["Alpha"]) for r in g]
        ys = [f(r[metric]) for r in g]
        baseline = f(b[metric])

        ax.plot(xs, ys, marker="o", color="#2563eb", linewidth=2.2, label="CatExpert")
        ax.axhline(
            baseline,
            color="#dc2626",
            linestyle="--",
            linewidth=2.4,
            label=f"Regular={baseline:.2f}",
        )

        if metric == "F1 Score":
            best = max(g, key=lambda r: f(r["F1 Score"]))
            ax.scatter([f(best["Alpha"])], [f(best["F1 Score"])], s=85, color="#16a34a", zorder=5)
            ax.annotate(
                f"α={best['Alpha']}",
                (f(best["Alpha"]), f(best["F1 Score"])),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=9,
            )

        ax.set_title(f"{dataset} {setting}")
        ax.set_xlabel("alpha")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes[len(order):]:
        ax.axis("off")

    fig.suptitle(f"POPE CatExpert Alpha Sweep vs Regular Baseline: {metric}", fontsize=17)
    fig.tight_layout()
    fig.savefig(out / f"grid_alpha_curve_{metric.lower().replace(' ', '_')}.png", dpi=180)
    plt.close(fig)

print("wrote", out)
