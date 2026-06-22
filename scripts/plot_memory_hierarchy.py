#!/usr/bin/env python3
"""
plot_memory_hierarchy.py — Phase B: Memory Bandwidth + Latency
Reads: results/memory_bandwidth.csv
        results/memory_latency.csv
        results/hw_profile.csv   (for cache boundary annotations)
Saves: results/plot_memory_bandwidth.png
        results/plot_memory_latency.png

Bandwidth plot:
  - Log-scale x-axis (buffer size in KB)
  - Shaded regions for L1 / L2 / L3 / DRAM based on detected cache sizes
  - Four lines: read, write, copy, triad

Latency plot:
  - Log-scale x-axis, linear y-axis (ns)
  - Annotated inflection points for each cache level
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np

BW_CSV   = "results/memory_bandwidth.csv"
LAT_CSV  = "results/memory_latency.csv"
HW_CSV   = "results/hw_profile.csv"

BW_PLOT  = "results/plot_memory_bandwidth.png"
LAT_PLOT = "results/plot_memory_latency.png"

# ── Style ──────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
BG_COLOR    = "#0d0d0d"
PANEL_COLOR = "#141414"
GRID_COLOR  = "#2a2a2a"

# Cache region colors (muted, semi-transparent fills)
REGION_COLORS = {
    "L1":   ("#4CAF50", 0.10),  # green
    "L2":   ("#2196F3", 0.10),  # blue
    "L3":   ("#FF9800", 0.10),  # orange
    "DRAM": ("#9C27B0", 0.07),  # purple
}

# Kernel line colors
LINE_COLORS = {
    "read_gbps":  "#4FC3F7",
    "write_gbps": "#EF5350",
    "copy_gbps":  "#66BB6A",
    "triad_gbps": "#FFA726",
}
LINE_LABELS = {
    "read_gbps":  "Read",
    "write_gbps": "Write",
    "copy_gbps":  "Copy (2× BW)",
    "triad_gbps": "Triad (3× BW)",
}

def load_hw_profile():
    """Return dict of key→value from hw_profile.csv, or empty dict."""
    if not os.path.exists(HW_CSV):
        return {}
    df = pd.read_csv(HW_CSV)
    return dict(zip(df["key"].str.strip(), df["value"].str.strip()))

def get_cache_boundaries_kb(hw):
    """Return {label: size_kb} for detected cache levels."""
    bounds = {}
    try:
        l1 = int(hw.get("l1d_size_kb", 0))
        if l1 > 0: bounds["L1"] = l1
        l2 = int(hw.get("l2_size_kb", 0))
        if l2 > 0: bounds["L2"] = l2
        l3 = int(hw.get("l3_size_kb", 0))
        if l3 > 0: bounds["L3"] = l3
    except (ValueError, KeyError):
        pass
    return bounds

# ─────────────────────────────────────────────────────────────────────────
# Bandwidth plot
# ─────────────────────────────────────────────────────────────────────────
def plot_bandwidth(hw):
    if not os.path.exists(BW_CSV):
        print(f"SKIP: {BW_CSV} not found.")
        return

    df = pd.read_csv(BW_CSV)
    x  = df["buffer_size_kb"].values

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    # ── Shaded cache regions ──────────────────────────────────────────
    bounds = get_cache_boundaries_kb(hw)
    sorted_bounds = sorted(bounds.items(), key=lambda kv: kv[1])

    x_min, x_max = x.min(), x.max()
    prev_boundary = x_min

    region_order = ["L1", "L2", "L3", "DRAM"]
    region_patches = []

    for label in region_order:
        if label == "DRAM":
            lo, hi = prev_boundary, x_max * 1.5
        else:
            if label not in bounds:
                continue
            lo, hi = prev_boundary, bounds[label]

        color, alpha = REGION_COLORS[label]
        ax.axvspan(lo, hi, alpha=alpha, color=color, zorder=0)
        midpoint = (lo * hi) ** 0.5   # geometric midpoint on log scale
        ax.text(midpoint, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 1,
                label, color=color, fontsize=9, ha="center", va="top",
                alpha=0.8)
        region_patches.append(mpatches.Patch(color=color, alpha=0.4,
                                              label=label))
        if label != "DRAM":
            prev_boundary = bounds[label]
            # Add vertical boundary line
            ax.axvline(x=bounds[label], color=color, linewidth=0.8,
                       linestyle="--", alpha=0.5, zorder=1)

    # ── Bandwidth lines ────────────────────────────────────────────────
    for col in ["read_gbps", "write_gbps", "copy_gbps", "triad_gbps"]:
        if col in df.columns:
            ax.semilogx(x, df[col].values,
                        color=LINE_COLORS[col], linewidth=2.0,
                        marker="o", markersize=5,
                        label=LINE_LABELS[col], zorder=3)

    # ── Axis labels ────────────────────────────────────────────────────
    ax.set_xlabel("Buffer Size (KB)", fontsize=12, color="white")
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=12, color="white")
    ax.tick_params(colors="white")
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.6)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )

    ax.set_title(
        "Phase B1 — Memory Bandwidth vs Buffer Size\n"
        "(shaded regions = detected cache levels)",
        fontsize=13, color="white", pad=12
    )

    # ── Legend ────────────────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + region_patches,
              labels + [p.get_label() for p in region_patches],
              facecolor=PANEL_COLOR, edgecolor="#444", fontsize=9,
              loc="upper right")

    plt.tight_layout()
    plt.savefig(BW_PLOT, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved: {BW_PLOT}")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────
# Latency plot
# ─────────────────────────────────────────────────────────────────────────
def plot_latency(hw):
    if not os.path.exists(LAT_CSV):
        print(f"SKIP: {LAT_CSV} not found.")
        return

    df = pd.read_csv(LAT_CSV)
    x  = df["buffer_size_kb"].values
    y  = df["latency_ns"].values

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    # ── Shaded cache regions ──────────────────────────────────────────
    bounds = get_cache_boundaries_kb(hw)
    region_order = ["L1", "L2", "L3", "DRAM"]
    prev = x.min()
    region_patches = []
    for label in region_order:
        if label == "DRAM":
            lo, hi = prev, x.max() * 1.5
        else:
            if label not in bounds: continue
            lo, hi = prev, bounds[label]
        color, alpha = REGION_COLORS[label]
        ax.axvspan(lo, hi, alpha=alpha, color=color, zorder=0)
        region_patches.append(mpatches.Patch(color=color, alpha=0.4,
                                              label=label))
        if label != "DRAM":
            ax.axvline(x=bounds[label], color=color, linewidth=0.8,
                       linestyle="--", alpha=0.5)
            prev = bounds[label]

    # ── Latency curve ─────────────────────────────────────────────────
    ax.semilogx(x, y, color="#CE93D8", linewidth=2.5,
                marker="D", markersize=7, label="Access latency", zorder=3)
    ax.fill_between(x, y, alpha=0.12, color="#CE93D8")

    # ── Annotate each cache-level inflection ──────────────────────────
    for label, kb in sorted(bounds.items(), key=lambda kv: kv[1]):
        # Find the closest data point just above this boundary
        candidates = df[df["buffer_size_kb"] >= kb]
        if candidates.empty: continue
        row = candidates.iloc[0]
        color = REGION_COLORS[label][0]
        ax.annotate(
            f"{label} limit\n{kb:,} KB",
            xy=(row["buffer_size_kb"], row["latency_ns"]),
            xytext=(row["buffer_size_kb"] * 1.8, row["latency_ns"] * 0.7),
            color=color, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_COLOR,
                      ec=color, alpha=0.85)
        )

    # ── Axis labels ────────────────────────────────────────────────────
    ax.set_xlabel("Buffer Size (KB, log scale)", fontsize=12, color="white")
    ax.set_ylabel("Access Latency (ns)", fontsize=12, color="white")
    ax.tick_params(colors="white")
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.6)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )

    ax.set_title(
        "Phase B2 — Memory Latency via Pointer Chasing\n"
        "(random access, hardware prefetch defeated)",
        fontsize=13, color="white", pad=12
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + region_patches,
              labels + [p.get_label() for p in region_patches],
              facecolor=PANEL_COLOR, edgecolor="#444", fontsize=9)

    plt.tight_layout()
    plt.savefig(LAT_PLOT, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved: {LAT_PLOT}")
    plt.close()

def main():
    hw = load_hw_profile()
    plot_bandwidth(hw)
    plot_latency(hw)

if __name__ == "__main__":
    main()
