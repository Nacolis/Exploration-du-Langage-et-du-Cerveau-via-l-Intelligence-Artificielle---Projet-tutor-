import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = {"semantic": "#2166ac", "jabberwocky": "#d6604d"}
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_hunchback():
    d = load_json(DATA_DIR / "id_hunchback.json")
    result = {}
    for cond in ["semantic", "jabberwocky"]:
        layers = [e["layer"] for e in d[cond] if e["mean"] is not None]
        ids = [e["mean"] for e in d[cond] if e["mean"] is not None]
        ci_low = [e["ci_low"] for e in d[cond] if e["mean"] is not None]
        ci_high = [e["ci_high"] for e in d[cond] if e["mean"] is not None]
        result[cond] = {
            "layers": np.array(layers),
            "ids": np.array(ids),
            "ci_low": np.array(ci_low),
            "ci_high": np.array(ci_high),
        }
    return result


def load_hunchback_mean():
    d = load_json(DATA_DIR / "id_hunchback_mean.json")
    result = {}
    for cond in ["semantic", "jabberwocky"]:
        layers = [e["layer"] for e in d[cond] if e["mean"] is not None]
        ids = [e["mean"] for e in d[cond] if e["mean"] is not None]
        ci_low = [e["ci_low"] for e in d[cond] if e["mean"] is not None]
        ci_high = [e["ci_high"] for e in d[cond] if e["mean"] is not None]
        result[cond] = {
            "layers": np.array(layers),
            "ids": np.array(ids),
            "ci_low": np.array(ci_low),
            "ci_high": np.array(ci_high),
        }
    return result


def load_ramping():
    return load_json(DATA_DIR / "id_ramping.json")


def load_token_counts():
    return load_json(DATA_DIR / "token_counts.json")


def fig_hunchback(hb: dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    layer_labels = ["Emb"] + [f"L{i}" for i in range(1, 12)] + ["L12"]

    for cond, style in [("semantic", "-o"), ("jabberwocky", "--s")]:
        d = hb[cond]
        ax.plot(
            d["layers"],
            d["ids"],
            style,
            color=COLORS[cond],
            linewidth=2.0,
            markersize=6,
            label=cond.capitalize(),
            zorder=3,
        )
        ax.fill_between(
            d["layers"],
            d["ci_low"],
            d["ci_high"],
            color=COLORS[cond],
            alpha=0.15,
            zorder=2,
        )

    ax.set_xticks(range(13))
    ax.set_xticklabels(layer_labels, fontsize=10)
    ax.set_xlabel("GPT-2 Layer", fontsize=12)
    ax.set_ylabel("Intrinsic Dimensionality (TwoNN)", fontsize=12)
    ax.set_title(
        "Hunchback Profile: ID across GPT-2 Small layers\n"
        "(last-token representation, N=500 sentences, 95% bootstrap CI)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5, label="Embedding")
    ax.axvline(12, color="gray", linestyle=":", alpha=0.5)

    ax.annotate(
        "Embedding\nlayer",
        xy=(0, ax.get_ylim()[0]),
        xytext=(0.3, ax.get_ylim()[0] * 0.98),
        fontsize=9,
        color="gray",
    )
    ax.annotate(
        "Final layer",
        xy=(12, ax.get_ylim()[0]),
        xytext=(11.0, ax.get_ylim()[0] * 0.98),
        fontsize=9,
        color="gray",
    )

    fig.tight_layout()
    out = FIG_DIR / "fig1_hunchback.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def fig_hunchback_mean(hb: dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    layer_labels = ["Emb"] + [f"L{i}" for i in range(1, 12)] + ["L12"]

    for cond, style in [("semantic", "-o"), ("jabberwocky", "--s")]:
        d = hb[cond]
        ax.plot(
            d["layers"],
            d["ids"],
            style,
            color=COLORS[cond],
            linewidth=2.0,
            markersize=6,
            label=cond.capitalize(),
            zorder=3,
        )
        ax.fill_between(
            d["layers"],
            d["ci_low"],
            d["ci_high"],
            color=COLORS[cond],
            alpha=0.15,
            zorder=2,
        )

    ax.set_xticks(range(13))
    ax.set_xticklabels(layer_labels, fontsize=10)
    ax.set_xlabel("GPT-2 Layer", fontsize=12)
    ax.set_ylabel("Intrinsic Dimensionality (TwoNN)", fontsize=12)
    ax.set_title(
        "Figure 1.2 — Hunchback Profile (Mean Token Representation):\n"
        "ID across GPT-2 Small layers (mean of all tokens, N=500 sentences, 95% bootstrap CI)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5, label="Embedding")
    ax.axvline(12, color="gray", linestyle=":", alpha=0.5)

    ax.annotate(
        "Embedding\nlayer",
        xy=(0, ax.get_ylim()[0]),
        xytext=(0.3, ax.get_ylim()[0] * 0.98),
        fontsize=9,
        color="gray",
    )
    ax.annotate(
        "Final layer",
        xy=(12, ax.get_ylim()[0]),
        xytext=(11.0, ax.get_ylim()[0] * 0.98),
        fontsize=9,
        color="gray",
    )

    fig.tight_layout()
    out = FIG_DIR / "fig1_2_hunchback.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def fig_ramping(ramping: dict):
    SHOW_LAYERS = [3, 6, 9, 12]
    fig, axes = plt.subplots(1, len(SHOW_LAYERS), figsize=(14, 3.8), sharey=False)

    bin_labels = [f"{int(b * 10)}%" for b in [0.1 * i for i in range(10)]]
    stats_by_layer = {s["layer"]: s for s in ramping.get("statistics", [])}

    for ax, layer in zip(axes, SHOW_LAYERS):
        layer_data = ramping["by_layer"].get(
            str(layer), ramping["by_layer"].get(layer, {})
        )

        for cond in ["semantic", "jabberwocky"]:
            entries = layer_data.get(cond, [])
            bins = [e["bin"] for e in entries if e.get("mean") is not None]
            ids = [e["mean"] for e in entries if e.get("mean") is not None]
            ci_low = [e["ci_low"] for e in entries if e.get("mean") is not None]
            ci_high = [e["ci_high"] for e in entries if e.get("mean") is not None]

            if not bins:
                continue

            style = "-o" if cond == "semantic" else "--s"
            ax.plot(
                bins,
                ids,
                style,
                color=COLORS[cond],
                linewidth=1.8,
                markersize=5,
                label=cond.capitalize(),
            )
            ax.fill_between(bins, ci_low, ci_high, color=COLORS[cond], alpha=0.15)

        ax.set_title(
            f"{'Embedding' if layer == 0 else f'Layer {layer}'}",
            fontsize=11,
        )
        ax.set_xlabel("Word position\n(decile)", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("ID (TwoNN)", fontsize=10)

        stat = stats_by_layer.get(layer)
        if stat and stat.get("significant"):
            ax.set_title(
                ax.get_title() + " *",
                fontsize=11,
                color="crimson",
            )

        ax.set_xticks(range(0, 10, 2))
        ax.set_xticklabels([f"{i * 10}%" for i in range(0, 10, 2)], fontsize=8)

    handles = [
        mpatches.Patch(color=COLORS["semantic"], label="Semantic"),
        mpatches.Patch(color=COLORS["jabberwocky"], label="Jabberwocky"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10)
    fig.suptitle(
        "Ramping Profile: ID evolution across word position\n"
        "(*layer shows significant Semantic vs Jabberwocky difference, Bonferroni-corrected)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig2_ramping.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def fig_bpe(token_counts: list):
    ratios = [t["bpe_ratio"] for t in token_counts if "bpe_ratio" in t]
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(ratios, bins=30, color="#5aae61", edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="ratio = 1.0")
    ax.axvline(
        np.mean(ratios),
        color="#d73027",
        linestyle="-",
        linewidth=2,
        label=f"mean = {np.mean(ratios):.2f}",
    )

    ax.set_xlabel("Token count ratio (Jabberwocky / Semantic)", fontsize=11)
    ax.set_ylabel("Number of sentences", fontsize=11)
    ax.set_title(
        "BPE Fragmentation: Jabberwocky vs Semantic token counts\n"
        "(ratio > 1 means pseudo-words split into more sub-tokens)",
        fontsize=12,
    )
    ax.legend(fontsize=10)

    fig.tight_layout()
    out = FIG_DIR / "fig3_bpe_fragmentation.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def fig_difference(hb: dict):
    sem = hb["semantic"]
    jab = hb["jabberwocky"]

    common = np.intersect1d(sem["layers"], jab["layers"])
    sem_ids = {l: id_ for l, id_ in zip(sem["layers"], sem["ids"])}
    jab_ids = {l: id_ for l, id_ in zip(jab["layers"], jab["ids"])}

    diff = np.array([sem_ids[l] - jab_ids[l] for l in common])

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2166ac" if d > 0 else "#d6604d" for d in diff]
    ax.bar(common, diff, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1.0)

    layer_labels = ["Emb"] + [f"L{i}" for i in range(1, 13)]
    ax.set_xticks(common)
    ax.set_xticklabels([layer_labels[l] for l in common], fontsize=10)
    ax.set_xlabel("GPT-2 Layer", fontsize=12)
    ax.set_ylabel("ΔID (Semantic − Jabberwocky)", fontsize=12)
    ax.set_title(
        "Difference in ID: Semantic minus Jabberwocky per layer\n"
        "(blue = semantic richer; red = jabberwocky richer)",
        fontsize=12,
    )

    fig.tight_layout()
    out = FIG_DIR / "fig4_id_difference.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    hb = load_hunchback()
    hb_mean = load_hunchback_mean()
    ramping = load_ramping()
    token_counts = load_token_counts()

    fig_hunchback(hb)
    fig_hunchback_mean(hb_mean)
    fig_ramping(ramping)
    fig_bpe(token_counts)
    fig_difference(hb)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
