import numpy as np
from skdim.id import TwoNN

SEED = 42
rng = np.random.default_rng(SEED)

AMBIENT_DIM = 768
N_TRIALS = 10
N_POINTS = 500



def embed_in_768d(X_low: np.ndarray) -> np.ndarray:
    n, d_low = X_low.shape
    A = rng.standard_normal((AMBIENT_DIM, d_low))
    Q, _ = np.linalg.qr(A)
    basis = Q[:, :d_low]
    return X_low @ basis.T


# Calcul de la dimension intrinsèque  TwoNN
def run_twonn(X: np.ndarray) -> float:
    std = X.std(axis=0)
    active = std > 1e-10
    if active.sum() < 2:
        raise ValueError(f"Only {active.sum()} active dims — need >= 2.")
    X_active = X[:, active]
    X_std = (X_active - X_active.mean(axis=0)) / X_active.std(axis=0)
    return float(TwoNN().fit(X_std).dimension_)


# Test de TwoNN
def test_manifold(name: str, true_id: int, generator) -> tuple:
    estimates = []
    for _ in range(N_TRIALS):
        X_low = generator(N_POINTS)
        if X_low.ndim == 1:
            X_low = X_low.reshape(-1, 1)
        if X_low.shape[1] < 2:
            X_low = np.hstack([X_low, np.zeros((N_POINTS, 1))])
        X_high = embed_in_768d(X_low)
        id_est = run_twonn(X_high)
        estimates.append(id_est)

    mean_est = float(np.mean(estimates))
    std_est = float(np.std(estimates))
    error = abs(mean_est - true_id)
    status = "✓" if error < 1.5 else "✗  <-- check!"
    print(f"  {name:35s}  true={true_id:3d}  "
          f"est={mean_est:6.2f} ± {std_est:.2f}  {status}")
    return mean_est, std_est


def test_min_sample_size():
    print("\n── Minimum sample size stability (true ID = 5) ──────────────────")

    def gen_5d(n):
        return rng.standard_normal((n, 5))

    for n in [20, 30, 50, 75, 100, 200, 500]:
        estimates = []
        for _ in range(N_TRIALS):
            X = embed_in_768d(gen_5d(n))
            try:
                estimates.append(run_twonn(X))
            except Exception:
                estimates.append(np.nan)
        mean_est = float(np.nanmean(estimates))
        std_est = float(np.nanstd(estimates))
        stable = "stable" if std_est < 1.0 else "unstable"
        print(f"  N={n:4d}:  est={mean_est:6.2f} ± {std_est:.2f}  ({stable})")


def main():
    print("=" * 60)
    print("TwoNN Calibration Tests")
    print(f"Ambient dim: {AMBIENT_DIM}D,  N points per test: {N_POINTS}")
    print("=" * 60)

    print("\n── Synthetic manifold recovery ──────────────────────────────────")

    test_manifold("Line (1D)", true_id=1,
                  generator=lambda n: rng.standard_normal(n))

    test_manifold("Plane (2D)", true_id=2,
                  generator=lambda n: rng.standard_normal((n, 2)))

    def sphere_surface(n):
        X = rng.standard_normal((n, 4))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    test_manifold("3-Sphere surface (3D)", true_id=3,
                  generator=sphere_surface)

    test_manifold("10D hyperplane", true_id=10,
                  generator=lambda n: rng.standard_normal((n, 10)))

    test_manifold("20D hyperplane", true_id=20,
                  generator=lambda n: rng.standard_normal((n, 20)))

    print("\n  Full 768D Gaussian (no manifold, expect ID >> 20):")
    try:
        X = rng.standard_normal((N_POINTS, AMBIENT_DIM))
        print(f"    est = {run_twonn(X):.1f}")
    except Exception as e:
        print(f"    Failed: {e}")

    test_min_sample_size()


if __name__ == "__main__":
    main()