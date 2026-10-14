import numpy as np

K_NEIGHBORS = 10    
TOP_N = 10          


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




def gram_schmidt(A, tol=1e-10):
    """Modified Gram-Schmidt on the COLUMNS of A. Returns Q (orthonormal) and kept columns."""
    Q, kept = [], []
    for j in range(A.shape[1]):
        v = A[:, j].astype(float).copy()
        for q in Q:
            v = v - (q @ v) * q
        nv = np.linalg.norm(v)
        if nv > tol:
            Q.append(v / nv)
            kept.append(j)
    return np.array(Q).T, kept


def part3_gram_schmidt_ls(ctx):
    banner("PART 3: Gram-Schmidt, projection and least-squares prediction")
    movies, um = ctx["movies"], ctx["users_movies"]
    R, trial, idx, valid_rows = ctx["R"], ctx["trial_user"], ctx["index_small"], ctx["valid_rows"]

    k = min(K_NEIGHBORS, R.shape[0])
    nb = ctx["pe_order"][:k]                      # k most correlated users
    A = R[nb].T                                   # 20 x k  (columns = neighbours)
    b = trial                                     # 20-vector to approximate

    # --- Orthogonalisation ---
    Q, kept = gram_schmidt(A)
    print(f"Neighbour matrix A: {A.shape}; independent neighbours kept: {len(kept)}")
    print("Check Q^T Q = I  ->  max deviation:", np.abs(Q.T @ Q - np.eye(Q.shape[1])).max())

    # --- Projection of trial user onto Col(A) ---
    p = Q @ (Q.T @ b)
    resid = b - p
    print("\nProjection p of trial user onto span(neighbours):")
    print("  trial  :", b.astype(int))
    print("  proj p :", p)
    print(f"  ||b - p|| = {np.linalg.norm(resid):.3f}  (distance to the neighbour subspace)")
    print("  Check A^T (b - p) = 0 ->", np.abs(A.T @ resid).max())

    # --- Least squares: min ||A x - b||  (normal equations vs lstsq) ---
    x_ne = np.linalg.solve(A.T @ A, A.T @ b)
    x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    print("\nLeast-squares weights x (how much each neighbour 'contributes'):")
    print("  normal equations :", x_ne)
    print("  np.linalg.lstsq  :", x_ls)
    print("  agree?", np.allclose(x_ne, x_ls, atol=1e-6))

    # --- Hold-out validation: hide one of the 20 movies, predict it ---
    err_ls, err_mean, err_nn = [], [], []
    for j in range(R.shape[1]):
        keep = [i for i in range(R.shape[1]) if i != j]
        xj, *_ = np.linalg.lstsq(A[keep, :], b[keep], rcond=None)
        pred = float(np.clip(A[j, :] @ xj, 1, 5))
        err_ls.append(pred - b[j])
        err_mean.append(b[keep].mean() - b[j])
        err_nn.append(A[j, 0] - b[j])      # copy rating of the most similar user
    rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
    print("\nLeave-one-movie-out RMSE (lower is better):")
    print(f"  Least squares (projection) : {rmse(err_ls):.3f}")
    print(f"  Trial user's own average   : {rmse(err_mean):.3f}")
    print(f"  Copy nearest neighbour     : {rmse(err_nn):.3f}")

    # --- Predict ratings for ALL other movies, then recommend ---
    N = um[valid_rows[nb]].astype(float)          # k x all-movies
    rated = N > 0
    num = (x_ls[:, None] * N * rated).sum(axis=0)
    den = (np.abs(x_ls)[:, None] * rated).sum(axis=0)
    support = rated.sum(axis=0)
    pred_all = np.where(den > 0, num / np.where(den > 0, den, 1), 0)
    pred_all = np.clip(pred_all, 0, 5)
    pred_all[support < min(3, k)] = 0             # need a few raters to trust it
    pred_all[np.asarray(idx)] = 0                 # skip the 20 seen movies
    order = np.argsort(-pred_all)
    rec_ls = [int(j) for j in order[:TOP_N] if pred_all[j] > 0]
    print("\nTop predicted movies (least-squares weighted neighbours):")
    for j in rec_ls:
        print(f"  {get_title(movies, j):45s} predicted rating = {pred_all[j]:.2f}")

    ctx.update(rec_ls=rec_ls, Q=Q, lsq_weights=x_ls)