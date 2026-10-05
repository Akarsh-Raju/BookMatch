import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.patches import Arc
import textwrap

"""
Member 3 Module: Polished Geometric & Visual Analytics of Book Vectors.
Displays an interactive Matplotlib dashboard window directly using plt.show().
"""

def plot_vector_angles_and_bars(target_meta, target_vec, ranked_results, feature_names, show_window=True, save_path=None):
    top_match = ranked_results[0]
    least_match = ranked_results[-1]

    # Clean, modern typography and styling
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'
    plt.rcParams['axes.linewidth'] = 1.1

    fig = plt.figure(figsize=(16.5, 7.0), facecolor='#F8FAFC')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.58, left=0.055, right=0.965, top=0.81, bottom=0.12)

    # Main Dashboard Header Banner
    fig.suptitle(
        f"Book-Match Linear Algebra Dashboard — Target Novel: '{target_meta['Title']}'",
        fontsize=15.5, fontweight='bold', color='#0F172A', y=0.95
    )
    fig.text(
        0.5, 0.895,
        f"Author: {target_meta['Author']} ({target_meta['Year']})   |   Primary Category: {target_meta['Category'].replace('_', ' ')}   |   Vector Space: R^20",
        ha='center', fontsize=10.5, color='#475569'
    )

    # =========================================================================
    # PANEL 1: 2D Geometric Plane Projection of Vector Angles (θ)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#FFFFFF')

    norm_target = float(np.sqrt(np.sum(target_vec ** 2)))
    norm_top = float(top_match["norm_b"])
    norm_least = float(least_match["norm_b"])

    theta1_rad = np.radians(top_match["angle_deg"])
    x1, y1 = norm_top * np.cos(theta1_rad), norm_top * np.sin(theta1_rad)

    theta2_rad = np.radians(least_match["angle_deg"])
    x2, y2 = norm_least * np.cos(theta2_rad), norm_least * np.sin(theta2_rad)

    max_lim = max(norm_target, norm_top, norm_least) * 1.30

    ax1.grid(True, linestyle='--', alpha=0.5, color='#CBD5E1', zorder=0)
    ax1.axhline(0, color='#94A3B8', linewidth=1.2, zorder=1)
    ax1.axvline(0, color='#94A3B8', linewidth=1.2, zorder=1)

    # Draw geometric angle arcs to highlight theta
    arc_radius_1 = min(norm_target, norm_top) * 0.42
    arc1 = Arc((0, 0), arc_radius_1 * 2, arc_radius_1 * 2, angle=0, theta1=0, theta2=top_match["angle_deg"],
               color='#10B981', linewidth=2.0, linestyle='-', zorder=2)
    ax1.add_patch(arc1)

    arc_radius_2 = min(norm_target, norm_least) * 0.26
    arc2 = Arc((0, 0), arc_radius_2 * 2, arc_radius_2 * 2, angle=0, theta1=0, theta2=least_match["angle_deg"],
               color='#EF4444', linewidth=1.6, linestyle=':', zorder=2)
    ax1.add_patch(arc2)

    # Plot the 3 Vectors using Quiver
    t_title = textwrap.shorten(target_meta['Title'], width=26, placeholder="...")
    m_title = textwrap.shorten(top_match['Title'], width=26, placeholder="...")
    l_title = textwrap.shorten(least_match['Title'], width=26, placeholder="...")

    ax1.quiver(0, 0, norm_target, 0, angles='xy', scale_units='xy', scale=1,
               color='#2563EB', width=0.014, headwidth=4.2, headlength=5, zorder=4,
               label=f"Target: {t_title} (0.0°, ||A||={norm_target:.2f})")

    ax1.quiver(0, 0, x1, y1, angles='xy', scale_units='xy', scale=1,
               color='#10B981', width=0.014, headwidth=4.2, headlength=5, zorder=5,
               label=f"Top Match: {m_title} (θ = {top_match['angle_deg']:.1f}°, cos θ = {top_match['cos_sim']:.4f})")

    ax1.quiver(0, 0, x2, y2, angles='xy', scale_units='xy', scale=1,
               color='#EF4444', width=0.014, headwidth=4.2, headlength=5, zorder=3,
               label=f"Least Similar: {l_title} (θ = {least_match['angle_deg']:.1f}°, cos θ = {least_match['cos_sim']:.4f})")

    # Clean callout badges at the tip of the vectors
    ax1.annotate(
        f"θ = {top_match['angle_deg']:.1f}°",
        xy=(x1, y1), xytext=(x1 + 0.12, y1 + 0.12),
        fontsize=9, fontweight='bold', color='#047857',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='#ECFDF5', edgecolor='#6EE7B7', lw=0.9)
    )
    ax1.annotate(
        f"θ = {least_match['angle_deg']:.1f}°",
        xy=(x2, y2), xytext=(x2 + 0.10, y2 + 0.08),
        fontsize=9, fontweight='bold', color='#B91C1C',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='#FEF2F2', edgecolor='#FCA5A5', lw=0.9)
    )

    ax1.set_xlim(-0.25, max_lim)
    ax1.set_ylim(-0.25, max_lim * 0.92)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title("1. Geometric Vector Space Projection (Angle θ & Norm)", fontsize=12, fontweight='bold', color='#0F172A', pad=14)
    ax1.set_xlabel("Projection Along Target Vector A  (||B|| cos θ)", fontsize=10, fontweight='bold', color='#334155', labelpad=8)
    ax1.set_ylabel("Orthogonal Component  (||B|| sin θ)", fontsize=10, fontweight='bold', color='#334155', labelpad=8)
    ax1.tick_params(colors='#475569', labelsize=9.5)

    legend = ax1.legend(loc="upper left", frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=8.8)
    for text in legend.get_texts():
        text.set_color('#0F172A')

    # =========================================================================
    # DISTINCT VERTICAL DIVIDER LINE BETWEEN THE TWO CHARTS
    # =========================================================================
    divider_line = lines.Line2D(
        [0.412, 0.412], [0.07, 0.85],
        transform=fig.transFigure, color='#CBD5E1', linewidth=2.0, linestyle='-'
    )
    fig.lines.append(divider_line)

    # =========================================================================
    # PANEL 2: Top 5 Recommendations Ranked by Cosine Similarity
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#FFFFFF')

    top_5 = ranked_results[:5][::-1]  # Reverse so Rank #1 is at the top
    y_labels = []
    for idx_rev, r in enumerate(top_5):
        rank_num = 5 - idx_rev
        wrapped_title = textwrap.fill(r['Title'], width=26)
        cat_clean = r['Category'].replace('_', ' ')
        y_labels.append(f"#{rank_num}  {wrapped_title}\n[{cat_clean}]")

    scores = [r["cos_sim"] for r in top_5]
    angles = [r["angle_deg"] for r in top_5]

    bar_colors = ['#93C5FD', '#60A5FA', '#3B82F6', '#2563EB', '#1D4ED8']
    bars = ax2.barh(y_labels, scores, color=bar_colors, edgecolor='#1E3A8A', linewidth=0.9, height=0.55, zorder=3)

    # Full boxed frame ending cleanly at 1.42 so badges sit well inside the right border
    ax2.set_xlim(0, 1.42)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.grid(axis='x', linestyle='--', alpha=0.45, color='#CBD5E1', zorder=0)

    ax2.set_title("2. Top 5 Recommended Novels (Cosine Similarity & Angle)", fontsize=12, fontweight='bold', color='#0F172A', pad=14)
    ax2.set_xlabel("Cosine Similarity Score:   cos(θ) = (A · B) / (||A|| ||B||)", fontsize=10, fontweight='bold', color='#334155', labelpad=8)
    ax2.tick_params(axis='y', colors='#0F172A', labelsize=9.2)
    ax2.tick_params(axis='x', colors='#475569', labelsize=9.5)

    # Polished pill badges placed cleanly to the right of each bar with plenty of margin before the right border
    for bar, score, ang in zip(bars, scores, angles):
        ax2.text(
            bar.get_width() + 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"cos θ = {score:.4f}  ({ang:.1f}°)",
            va='center', ha='left', fontsize=9.2, fontweight='bold', color='#0F172A',
            bbox=dict(boxstyle='round,pad=0.28', facecolor='#F1F5F9', edgecolor='#CBD5E1', lw=0.9),
            zorder=4
        )

    if save_path:
        plt.savefig(save_path, dpi=220, facecolor=fig.get_facecolor())

    if show_window:
        plt.show()
    else:
        plt.close()
