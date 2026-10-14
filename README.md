# Linear-Algebra-Project
Movie Title Recommendation System Using Principles Of Linear Algebra
# Movie Recommender using Linear Algebra

**UE25MA242A: Mathematics for AI and Data (MFAD), Mini Project**
PES University, Department of Computer Science and Engineering

A movie recommendation system built from scratch using core linear algebra concepts. It takes a user-by-movie ratings matrix, follows the workflow from the course guidelines (matrix representation, RREF, rank and basis, Gram-Schmidt, projection, least squares, eigenvalues, diagonalisation), and ends with movie recommendations for a "trial user".

## Team

| Part | Member | Topic |
|------|--------|-------|
| Part 1 | _Name_ | Matrix representation, RREF, rank/nullity, basis |
| Part 2 | _Name_ | Norms, angles, Pearson correlation, nearest users |
| Part 3 | _Name_ | Gram-Schmidt, projection, least squares |
| Part 4 | _Name_ | Eigenvalues, diagonalisation, PCA, final output |

## Problem Statement

Given the ratings of many users for many movies, and the ratings a new (trial) user has given to the 20 most popular movies, recommend other movies this user is likely to enjoy.

## Dataset

The data is stored in `users_movies.mat`, which holds:

| Variable | Meaning |
|----------|---------|
| `users_movies` | Ratings matrix (users x movies). Values are 1 to 5, and `0` means "not rated" |
| `movies` | Movie titles |
| `index_small` | Column indices of the 20 most popular movies |
| `trial_user` | The trial user's ratings for those 20 movies |

`create_mat.py` generates this file with synthetic data: 1500 users, 200 movies, and 3 hidden "taste" features. The first 20 movies are popular and rated far more often than the rest. If `users_movies.mat` is missing, `main.py` builds the same kind of synthetic data automatically, so the project always runs.

## Linear Algebra Workflow

| Stage | Concept | Where | What it does here |
|-------|---------|-------|-------------------|
| Matrix representation | Data as a matrix | Part 1 | Ratings are stored as a users x movies matrix, with `0` meaning unrated |
| Matrix simplification | Gaussian elimination / RREF | Part 1 | A hand-written Gauss-Jordan routine reduces a sample of users to RREF and finds the pivot columns |
| Structure of the space | Rank, nullity, basis | Part 1 | Computes the rank of the matrix of users who rated all 20 popular movies, then its nullity and whether the movies form a basis |
| Remove redundancy | Linear independence | Part 1 | Identifies which movies are independent and which are redundant |
| Similarity | Norms, angles, correlation | Part 2 | Compares the trial user to every other user using Euclidean, Manhattan and Chebyshev distances, cosine angle, and Pearson correlation |
| Orthogonalisation | Gram-Schmidt | Part 3 | Builds an orthonormal basis for the span of the most similar users |
| Projection | Orthogonal projection | Part 3 | Projects the trial user onto the subspace spanned by similar users |
| Prediction | Least squares | Part 3 | Finds the weights that best combine similar users to reproduce the trial user, then predicts ratings for unseen movies |
| Pattern discovery | Eigenvalues and eigenvectors | Part 4 | Finds the main taste directions in the covariance matrix |
| System simplification | Diagonalisation (symmetric matrix) | Part 4 | Verifies `C = QΛQᵀ` for the covariance matrix |
| Final output | PCA, noise reduction, predictions | Part 4 | Compresses users to a few principal components, finds neighbours in that space, and combines all methods into a final recommendation list |

## Repository Structure

```
.
├── main.py             # Data loading, Parts 1 and 2, and main()
├── gramschmidt.py      # Part 3: Gram-Schmidt, projection, least squares
├── eigen_pca.py        # Part 4: eigenvalues, PCA, plots, final summary
├── create_mat.py       # Generates users_movies.mat (synthetic data)
├── users_movies.mat    # Dataset (created by create_mat.py)
├── lab07_plots.png     # Output plots (created when the program runs)
└── README.md
```

> The main file should not be named `math.py`. Python already has a built-in `math` module, and a local file with that name can shadow it and break imports.

## Setup and Usage

**Requirements:** Python 3.8+, `numpy`, `scipy`, and optionally `matplotlib` (for the plots).

```bash
python -m pip install numpy scipy matplotlib
```

**1. Generate the dataset** (optional, since synthetic data is used if the file is missing):

```bash
python create_mat.py
```

**2. Run the project:**

```bash
python main.py
```

All four modules must be in the same folder. `main.py` imports Parts 3 and 4 from `gramschmidt.py` and `eigen_pca.py`.

## What the Program Prints

1. **Part 1:** matrix dimensions and sparsity, the 20 popular movies, the number of users who rated all 20, the RREF and pivot columns of a sample, and the rank, nullity and basis conclusion.
2. **Part 2:** the closest users to the trial user under each distance and similarity measure, and recommendations from the closest user and from the K nearest users.
3. **Part 3:** a check that the Gram-Schmidt output is orthonormal, the projection of the trial user and its residual, least-squares weights (normal equations vs `np.linalg.lstsq`), a leave-one-movie-out error comparison, and the top predicted movies.
4. **Part 4:** covariance matrix checks, eigenvalues and explained variance, the diagonalisation check, the number of components needed for 80% variance, the strongest movies per principal component, the low-rank approximation error, and recommendations from PCA-space neighbours.
5. **Final output:** recommendations from every method, plus a consensus list of movies recommended by several methods.

The program also saves `lab07_plots.png` with three plots: a scree plot of eigenvalues, users in PC1-PC2 space with the trial user marked, and a histogram of Pearson correlations between the trial user and other users.

## Viva Quick Reference (Concept → Purpose → Outcome)

| Concept | Purpose | Outcome |
|---------|---------|---------|
| Matrix representation | Turn ratings into something we can compute with | A users x movies matrix |
| RREF | Simplify the matrix and expose its structure | Pivot columns and independent movies |
| Rank, nullity, basis | Measure how much independent information the data holds | Whether the 20 movies are all needed |
| Norms and angles | Measure how close two users' tastes are | The nearest users to the trial user |
| Gram-Schmidt | Get a clean orthogonal basis for similar users | An orthonormal basis `Q` |
| Projection | Find the best approximation of the trial user inside that subspace | The projected rating vector and its residual |
| Least squares | Find the best weights for combining similar users | Weights used to predict unseen movies |
| Eigenvalues and eigenvectors | Find the main directions in which tastes vary | Principal components and explained variance |
| Diagonalisation | Simplify the covariance matrix into `QΛQᵀ` | A compact, noise-reduced taste model |

## Files in This Repo

| File | Description |
|------|-------------|
| `main.py` | Entry point. Loads data, runs Parts 1 and 2, and calls the other modules |
| `gramschmidt.py` | Part 3 functions |
| `eigen_pca.py` | Part 4 functions, plotting, and the final summary |
| `create_mat.py` | Synthetic dataset generator |
| `UE25MA242A_MFAD_-Mini-Project_Guidelines.pdf` | Course guidelines for the mini project (if included in the repo) |
