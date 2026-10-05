import textwrap
import numpy as np
from data_loader import BookVectorDataset
from math_engine import (
    compute_cosine_similarity,
    compute_angle_degrees,
    compute_euclidean_distance,
    compute_vector_norm
)
from visualizer import plot_vector_angles_and_bars

def run_recommendation_pipeline(query_title, dataset, top_k=5, show_window=True, save_path=None):
    target_idx = dataset.find_book_index(query_title)

    if target_idx is None:
        print(f"[Error] Book '{query_title}' not found in the dataset. Please try another title.\n")
        return False

    target_meta = dataset.get_book_metadata(target_idx)
    target_vec = dataset.get_book_vector(target_idx)
    target_norm = compute_vector_norm(target_vec)

    print("=" * 86)
    print("STAGE 1: FEATURE MATRIX EXTRACTION, RANK & TARGET VECTORIZATION")
    print("=" * 86)

    # --- Extract & Print Feature Matrix + Dimensions + Rank ---
    feature_matrix = dataset.feature_matrix
    num_books, num_features = feature_matrix.shape
    matrix_rank = int(np.linalg.matrix_rank(feature_matrix))
    independence_status = (
        "Full Column Rank (All 20 features are linearly independent)"
        if matrix_rank == num_features
        else f"Rank Deficient ({num_features - matrix_rank} redundant features exist)"
    )

    print(f"Dataset Matrix : {num_books} Books x {num_features} Features (Shape: {feature_matrix.shape})")
    print(f"Matrix Rank    : {matrix_rank} -> {independence_status}")
    print(f"\nExtracted Feature Matrix X (in R^{{{num_books}x{num_features}}}) [Showing first 3 & last 2 rows]:")
    np.set_printoptions(precision=2, suppress=True, threshold=100, edgeitems=3, linewidth=120)
    print(feature_matrix)
    print("-" * 86)

    print(f"Selected Book  : {target_meta['Title']} by {target_meta['Author']} ({target_meta['Year']})")
    print(f"Primary Genre  : {target_meta['Category']}")
    print(f"Vector Space   : R^{len(target_vec)} (12 Genres + 8 Literary Attributes)")
    print(f"Feature Vector : {target_vec.round(2).tolist()}")
    print(f"L2 Norm ||A||  : {target_norm:.4f}")

    # Stage 2: Compute pairwise linear algebra metrics across all other novels
    results = []
    for idx in range(len(dataset.df)):
        if idx == target_idx:
            continue
        cand_meta = dataset.get_book_metadata(idx)
        cand_vec = dataset.get_book_vector(idx)

        cos_sim, dot_prod, norm_a, norm_b = compute_cosine_similarity(target_vec, cand_vec)
        angle_deg = compute_angle_degrees(cos_sim)
        euclid_dist = compute_euclidean_distance(target_vec, cand_vec)

        results.append({
            **cand_meta,
            "dot_product": dot_prod,
            "norm_b": norm_b,
            "cos_sim": cos_sim,
            "angle_deg": angle_deg,
            "euclid_dist": euclid_dist
        })

    # Sort descending by Cosine Similarity
    results.sort(key=lambda x: x["cos_sim"], reverse=True)

    print("\n" + "=" * 86)
    print(f"STAGE 2: INTERMEDIATE LINEAR ALGEBRA CALCULATIONS (TOP {top_k} MATCHES)")
    print("=" * 86)
    title_col_width = 34
    header = f"{'Rank':<5} {'Book Title':<{title_col_width}} {'A . B':<8} {'||B||':<8} {'cos(θ)':<9} {'Angle θ':<9} {'||A-B||'}"
    print(header)
    print("-" * 86)
    for rank, r in enumerate(results[:top_k], start=1):
        wrapped_lines = textwrap.wrap(r['Title'], width=title_col_width - 2)
        angle_str = f"{r['angle_deg']:.2f}°"
        
        print(f"{rank:<5} {wrapped_lines[0]:<{title_col_width}} {r['dot_product']:<8.4f} {r['norm_b']:<8.4f} {r['cos_sim']:<9.4f} {angle_str:<9} {r['euclid_dist']:.4f}")
        
        for extra_line in wrapped_lines[1:]:
            print(f"{'':<5} {extra_line:<{title_col_width}}")

    print("\n" + "=" * 86)
    print(f"STAGE 3: FINAL RANKED RECOMMENDATIONS FOR '{target_meta['Title'].upper()}'")
    print("=" * 86)
    for rank, r in enumerate(results[:top_k], start=1):
        print(f"{rank}. {r['Title']} by {r['Author']} ({r['Year']}) [{r['Category']}] -> Match: {r['cos_sim']*100:.2f}% (θ = {r['angle_deg']:.2f}°)")

    print("\n[Stage 4] Opening interactive Geometric Vector & Similarity Window...")
    plot_vector_angles_and_bars(target_meta, target_vec, results, dataset.feature_columns, show_window=show_window, save_path=save_path)
    return True

if __name__ == "__main__":
    dataset = BookVectorDataset("novels_vector_dataset_300.csv")
    while True:
        user_query = input("Enter a novel title: ").strip()
        if not user_query:
            print("You didn't type anything! Please enter a valid book title.\n")
            continue
        if run_recommendation_pipeline(user_query, dataset, top_k=5, show_window=True, save_path=None):
            break
