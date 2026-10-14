import numpy as np

K_NEIGHBORS = 10    # neighbours used in Part 4
TOP_N = 10          # how many recommendations to show


# =====================================================================
# SHARED HELPERS (copied from the main file so this module is standalone)
# =====================================================================

def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def get_title(movies, i):
    """Movie titles are stored in a MATLAB cell array -> unwrap nested arrays."""
    t = movies[i]
    while isinstance(t, (np.ndarray, list)) and len(t) > 0:
        t = t[0]
    return str(t)


def recommend_from_users(users_movies, user_rows, exclude_idx, top_n=TOP_N, min_raters=2):
    """
    Given a group of similar users (row numbers in users_movies), recommend movies
    that they rated highly, excluding movies already seen (the 20 popular ones).
    Score = average rating among neighbours who rated the movie.
    """
    block = users_movies[user_rows].astype(float)            # neighbours x movies
    rated = block > 0
    cnt = rated.sum(axis=0)
    avg = np.where(cnt > 0, block.sum(axis=0) / np.maximum(cnt, 1), 0)
    avg[cnt < min(min_raters, len(user_rows))] = 0
    avg[exclude_idx] = 0
    # sort by average rating, then by number of raters
    order = np.lexsort((-cnt, -avg))
    return [int(j) for j in order[:top_n] if avg[j] > 0]


# =====================================================================
# PART 4  (Person 4)  Eigenvalues -> Diagonalisation -> PCA -> Final output
# =====================================================================

def part4_eigen_pca(ctx):
    banner("PART 4: Eigenvalues, diagonalisation of covariance matrix, PCA")
    movies, um = ctx["movies"], ctx["users_movies"]
    R, trial, idx, valid_rows = ctx["R"], ctx["trial_user"], ctx["index_small"], ctx["valid_rows"]

    mu = R.mean(axis=0)
    X = R - mu                                     # centred data (users x 20)
    C = (X.T @ X) / (X.shape[0] - 1)               # 20 x 20 SYMMETRIC covariance matrix
    print("Covariance matrix is symmetric?", np.allclose(C, C.T), "| shape", C.shape)

    # --- Eigen decomposition (eigh -> real eigenvalues, orthogonal eigenvectors) ---
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, Qe = vals[order], vecs[:, order]
    explained = vals / vals.sum()
    cum = np.cumsum(explained)
    print("\nTop 5 eigenvalues         :", vals[:5])
    print("Explained variance ratio  :", explained[:5])
    print("Cumulative (top 5)        :", cum[:5])

    # --- Diagonalisation:  C = Q Lambda Q^T ---
    recon = Qe @ np.diag(vals) @ Qe.T
    print("\nDiagonalisation check  C = Q Λ Qᵀ  ->  max error:", np.abs(C - recon).max())
    print("Eigenvectors orthogonal (QᵀQ = I)?", np.allclose(Qe.T @ Qe, np.eye(Qe.shape[0])))

    # --- Choose number of components (noise reduction) ---
    k = int(np.searchsorted(cum, 0.80) + 1)
    k = max(2, min(k, len(vals)))
    print(f"\nComponents needed for >=80% variance: {k} of {len(vals)}")
    Qk = Qe[:, :k]

    # Interpretation: which movies load most on PC1 / PC2
    for pc in range(min(2, k)):
        load = Qe[:, pc]
        top = np.argsort(-np.abs(load))[:3]
        print(f"PC{pc + 1} strongest movies:",
              [(get_title(movies, idx[t]), round(float(load[t]), 2)) for t in top])

    # --- Noise-reduced (low-rank) model of the users ---
    Z = X @ Qk                                     # users in k-D taste space
    X_hat = Z @ Qk.T
    err = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
    print(f"Rank-{k} approximation relative error: {err:.3f} "
          f"(compressed 20 -> {k} numbers per user)")

    # --- Trial user in PCA space; nearest users there ---
    zt = (trial - mu) @ Qk
    trial_smooth = np.clip(mu + zt @ Qk.T, 1, 5)
    print("\nTrial user's coordinates in PC space:", zt)
    print("Original ratings      :", trial.astype(int))
    print("Denoised (rank-k) fit :", trial_smooth)

    d_pca = np.linalg.norm(Z - zt, axis=1)
    pca_order = np.argsort(d_pca)
    kk = min(K_NEIGHBORS, R.shape[0])
    rec_pca = recommend_from_users(um, valid_rows[pca_order[:kk]], np.asarray(idx))
    print("\nRecommendations from PCA-space neighbours:")
    for j in rec_pca:
        print("  ", get_title(movies, j))

    ctx.update(rec_pca=rec_pca, vals=vals, explained=explained, Z=Z, zt=zt, k_pc=k)


def make_plots(ctx):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping plots)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].plot(range(1, len(ctx["vals"]) + 1), ctx["vals"], "o-")
    ax[0].set(title="Scree plot (eigenvalues)", xlabel="Component", ylabel="Eigenvalue")

    Z = ctx["Z"]
    ax[1].scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.4, label="users")
    ax[1].scatter(*ctx["zt"][:2], c="red", s=90, marker="*", label="trial user")
    ax[1].set(title="Users in PC1-PC2 taste space", xlabel="PC1", ylabel="PC2")
    ax[1].legend()

    ax[2].hist(ctx["pearson"], bins=40, color="tab:green")
    ax[2].set(title="Pearson correlation: trial user vs others", xlabel="correlation")

    plt.tight_layout()
    plt.savefig("lab07_plots.png", dpi=130)
    print("Plots saved to lab07_plots.png")


def final_summary(ctx):
    banner("FINAL APPLICATION OUTPUT: Recommendations by each method")
    movies = ctx["movies"]
    methods = {
        "Euclidean norm (Part 2)": ctx["rec_eu"],
        "Pearson / angle (Part 2)": ctx["rec_pe"],
        "Least squares projection (Part 3)": ctx["rec_ls"],
        "PCA / eigenvectors (Part 4)": ctx["rec_pca"],
    }
    votes = {}
    for name, recs in methods.items():
        print(f"\n{name}:")
        for j in recs:
            print("   -", get_title(movies, j))
            votes[j] = votes.get(j, 0) + 1

    print("\nCONSENSUS (movies recommended by several methods):")
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])[:TOP_N]
    for j, v in ranked:
        print(f"   {v}/4 methods  ->  {get_title(movies, j)}")
