import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import mannwhitneyu, linregress
from skdim.id import TwoNN

SEED = 42
rng = np.random.default_rng(SEED)

DATA_DIR = Path("data")
N_BOOTSTRAP = 1000
BOOTSTRAP_FRAC = 0.8
MIN_POINTS_FOR_ID = 50
N_LAYERS = 13
N_WORD_BINS = 10

ALPHA = 0.05
N_TESTS = N_LAYERS * N_WORD_BINS



def clean_matrix(matrix: np.ndarray) -> np.ndarray | None:
    mask = np.isfinite(matrix).all(axis=1)
    clean = matrix[mask]
    if clean.shape[0] < MIN_POINTS_FOR_ID:
        return None
    return clean



def compute_id_twonn(matrix: np.ndarray) -> float | None:
    mat = clean_matrix(matrix)
    if mat is None:
        return None

    std = mat.std(axis=0)
    active = std > 1e-10
    if active.sum() < 2:
        return None

    X = mat[:, active]
    X = (X - X.mean(axis=0)) / std[active]

    try:
        return float(TwoNN().fit(X).dimension_)
    except Exception as e:
        print(f"  [WARNING] TwoNN failed: {e}")
        return None



def bootstrap_id(matrix: np.ndarray, n_boot: int = N_BOOTSTRAP) -> dict:
    id_point = compute_id_twonn(matrix)
    if id_point is None:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_boot_valid": 0}

    n = matrix.shape[0]
    n_sample = max(MIN_POINTS_FOR_ID, int(n * BOOTSTRAP_FRAC))

    boot_ids = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n_sample, replace=False)
        val = compute_id_twonn(matrix[idx])
        if val is not None:
            boot_ids.append(val)

    if len(boot_ids) < 10:
        return {
            "mean": id_point,
            "ci_low": id_point,
            "ci_high": id_point,
            "n_boot_valid": len(boot_ids),
        }

    boot_arr = np.array(boot_ids)
    return {
        "mean": id_point,
        "ci_low": float(np.percentile(boot_arr, 2.5)),
        "ci_high": float(np.percentile(boot_arr, 97.5)),
        "n_boot_valid": len(boot_ids),
    }



def hunchback_analysis(layerwise_path: Path) -> dict:
    print("\n── Hunchback Analysis ───────────────────────────────────────────")
    data = np.load(layerwise_path)
    X = data["data"]
    condition = data["condition"]

    sem_idx = np.where(condition == 0)[0]
    jab_idx = np.where(condition == 1)[0]

    results = {"semantic": [], "jabberwocky": []}

    for layer in tqdm(range(N_LAYERS), desc="Hunchback layers"):
        for cond_name, idx in [("semantic", sem_idx), ("jabberwocky", jab_idx)]:
            matrix = X[idx, layer, :]
            stats = bootstrap_id(matrix)
            stats["layer"] = layer
            results[cond_name].append(stats)

            if stats["mean"] is not None:
                print(f"  Layer {layer:2d} | {cond_name:12s} | "
                      f"ID={stats['mean']:.2f} "
                      f"[{stats['ci_low']:.2f}, {stats['ci_high']:.2f}] "
                      f"(boots={stats['n_boot_valid']})")
            else:
                print(f"  Layer {layer:2d} | {cond_name:12s} | SKIP")

    return results



def ramping_analysis(incremental_path: Path) -> dict:
    print("\n── Ramping Analysis ─────────────────────────────────────────────")
    data = np.load(incremental_path)

    store = {"semantic": {}, "jabberwocky": {}}
    for key in data.files:
        parts = key.split("_")
        cond = parts[0]
        layer = int(parts[1][1:])
        bin_idx = int(parts[2][1:])
        store[cond].setdefault(layer, {})[bin_idx] = data[key]

    results = {"by_layer": {}, "statistics": []}

    for layer in tqdm(range(N_LAYERS), desc="Ramping layers"):
        layer_res = {"semantic": [], "jabberwocky": []}

        for bin_idx in range(N_WORD_BINS):
            for cond in ["semantic", "jabberwocky"]:
                mat = store[cond].get(layer, {}).get(bin_idx)
                if mat is None or mat.shape[0] < MIN_POINTS_FOR_ID:
                    layer_res[cond].append(
                        {"bin": bin_idx, "mean": None, "ci_low": None,
                         "ci_high": None, "n": 0}
                    )
                    continue
                stats = bootstrap_id(mat, n_boot=200)
                stats["bin"] = bin_idx
                stats["n"] = int(mat.shape[0])
                layer_res[cond].append(stats)

        results["by_layer"][str(layer)] = layer_res

        sem_ids = [e["mean"] for e in layer_res["semantic"] if e["mean"] is not None]
        jab_ids = [e["mean"] for e in layer_res["jabberwocky"] if e["mean"] is not None]

        if len(sem_ids) >= 3 and len(jab_ids) >= 3:
            stat, p = mannwhitneyu(sem_ids, jab_ids, alternative="two-sided")
            n1, n2 = len(sem_ids), len(jab_ids)
            r = 1 - (2 * stat) / (n1 * n2)
            results["statistics"].append({
                "layer": layer,
                "mwu_stat": float(stat),
                "p_raw": float(p),
                "p_bonferroni": min(float(p) * N_TESTS, 1.0),
                "effect_r": float(r),
                "significant": float(p) * N_TESTS < ALPHA,
            })

    return results



def test_ramping_slope(results: dict) -> dict:
    slope_results = {}
    for layer in range(N_LAYERS):
        layer_res = results["by_layer"].get(str(layer), {})
        slope_results[str(layer)] = {}
        for cond in ["semantic", "jabberwocky"]:
            bins, ids = [], []
            for entry in layer_res.get(cond, []):
                if entry["mean"] is not None:
                    bins.append(entry["bin"])
                    ids.append(entry["mean"])
            if len(bins) >= 3:
                slope, _, r, p, _ = linregress(bins, ids)
                slope_results[str(layer)][cond] = {
                    "slope": round(float(slope), 4),
                    "r2": round(float(r**2), 4),
                    "p": round(float(p), 4),
                    "positive_ramping": bool(slope > 0 and p < ALPHA),
                }
            else:
                slope_results[str(layer)][cond] = None
    return slope_results


def main():
    print("── NaN/Inf diagnostic ───────────────────────────────────────────")
    lw = np.load(DATA_DIR / "hidden_states_layerwise.npz")
    X = lw["data"]
    nan_rows = (~np.isfinite(X)).any(axis=(1, 2)).sum()
    print(f"  Layerwise array shape: {X.shape}")
    print(f"  Rows with any NaN/Inf: {nan_rows} / {X.shape[0]}")

    layerwise_path = DATA_DIR / "hidden_states_layerwise.npz"
    layerwise_mean_path = DATA_DIR / "hidden_states_layerwise_mean.npz"
    incremental_path = DATA_DIR / "hidden_states_incremental.npz"

    hunchback = hunchback_analysis(layerwise_path)
    with open(DATA_DIR / "id_hunchback.json", "w") as f:
        json.dump(hunchback, f, indent=2)
    print(f"\nSaved → {DATA_DIR}/id_hunchback.json")

    print("\nRunning Hunchback Analysis with mean-token representation...")
    hunchback_mean = hunchback_analysis(layerwise_mean_path)
    with open(DATA_DIR / "id_hunchback_mean.json", "w") as f:
        json.dump(hunchback_mean, f, indent=2)
    print(f"\nSaved → {DATA_DIR}/id_hunchback_mean.json")

    ramping = ramping_analysis(incremental_path)
    slope_tests = test_ramping_slope(ramping)
    ramping["slope_tests"] = slope_tests
    with open(DATA_DIR / "id_ramping.json", "w") as f:
        json.dump(ramping, f, indent=2)
    print(f"Saved → {DATA_DIR}/id_ramping.json")

    print("\n── Hunchback peak layer ─────────────────────────────────────────")
    for cond in ["semantic", "jabberwocky"]:
        entries = [(e["layer"], e["mean"]) for e in hunchback[cond] if e["mean"]]
        if entries:
            peak = max(entries, key=lambda x: x[1])
            print(f"  {cond:12s}: peak ID={peak[1]:.2f} at layer {peak[0]}")

    print("\n── Significant positive ramping slopes ──────────────────────────")
    found = False
    for layer_str, layer_data in slope_tests.items():
        for cond, result in layer_data.items():
            if result and result["positive_ramping"]:
                print(f"  Layer {layer_str:>2s} | {cond:12s}: "
                      f"slope={result['slope']:.4f}, R²={result['r2']:.3f}, "
                      f"p={result['p']:.4f}")
                found = True
    if not found:
        print("  None found at α=0.05 (may still see trends in visualisation).")


if __name__ == "__main__":
    main()