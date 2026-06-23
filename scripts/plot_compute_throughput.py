#!/usr/bin/env python3
"""
plot_compute_throughput.py — Phase C: Compute Throughput + ILP Sweep
Reads: results/compute_throughput.csv
Saves: results/plot_ilp_sweep.png
        results/plot_compute_summary.png

ILP sweep plot:
  Shows throughput (GFLOP/s) vs dependency chain depth for scalar f64.
  Annotates the plateau (ILP ceiling = optimal unroll factor).

Summary bar chart:
  Compares peak throughput across all kernel variants (scalar f64/f32, SIMD).
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CSV_PATH     = "results/compute_throughput.csv"
ILP_PLOT     = "results/plot_ilp_sweep.png"
SUMMARY_PLOT = "results/plot_compute_summary.png"

plt.style.use("dark_background")
BG_COLOR    = "#0d0d0d"
PANEL_COLOR = "#141414"
GRID_COLOR  = "#2a2a2a"

KERNEL_COLORS = {
    "scalar_fma/f64": "#4FC3F7",
    "scalar_fma/f32": "#81C784",
    "simd_fma/f32":   "#FFB74D",
    "simd_mul_add/f32": "#CE93D8",
}

def load():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run 'make run ARG=C' first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    expected = {"kernel", "precision", "chain_depth", "simd_lanes",
                "gflops_per_core"}
    if not expected.issubset(df.columns):
        print(f"ERROR: unexpected columns: {list(df.columns)}")
        sys.exit(1)
    df["kernel_key"] = df["kernel"] + "/" + df["precision"]
    return df

# ─────────────────────────────────────────────────────────────────────────
# ILP sweep plot (scalar f64)
# ─────────────────────────────────────────────────────────────────────────
def plot_ilp_sweep(df):
    f64 = df[(df["kernel"] == "scalar_fma") & (df["precision"] == "f64")].copy()
    if f64.empty:
        print("SKIP: no f64 scalar ILP data found.")
        return

    f64 = f64.sort_values("chain_depth")
    depths = f64["chain_depth"].values
    gflops = f64["gflops_per_core"].values

    peak_idx   = np.argmax(gflops)
    plateau_gf = gflops[peak_idx]
    plateau_d  = depths[peak_idx]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    # ── ILP curve ──────────────────────────────────────────────────────
    ax.plot(depths, gflops, color="#4FC3F7", linewidth=2.5,
            marker="o", markersize=9, label="Measured throughput", zorder=3)
    ax.fill_between(depths, gflops, alpha=0.15, color="#4FC3F7")

    # ── Plateau line ──────────────────────────────────────────────────
    ax.axhline(y=plateau_gf, color="#EF5350", linewidth=1.2,
               linestyle="--", alpha=0.7, label=f"Peak = {plateau_gf:.3f} GFLOP/s")

    # ── Saturation annotation ─────────────────────────────────────────
    ax.annotate(
        f"ILP ceiling reached\nat chain depth = {plateau_d}\n"
        f"→ unroll by ≥ {plateau_d}× in your kernel",
        xy=(plateau_d, plateau_gf),
        xytext=(plateau_d + 1, plateau_gf * 0.75),
        color="#EF5350", fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#EF5350", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", fc=PANEL_COLOR,
                  ec="#EF5350", alpha=0.9)
    )

    # ── Latency-bound annotation at chain=1 ───────────────────────────
    ax.annotate(
        "Chain depth 1:\nlatency-bound\n(FP units stall)",
        xy=(depths[0], gflops[0]),
        xytext=(depths[0] + 0.5, gflops[0] * 1.15),
        color="#FFB74D", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#FFB74D", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_COLOR,
                  ec="#FFB74D", alpha=0.9)
    )

    ax.set_xlabel("Dependency Chain Depth (independent accumulators)",
                  fontsize=12, color="white")
    ax.set_ylabel("Throughput (GFLOP/s, single core)", fontsize=12, color="white")
    ax.tick_params(colors="white")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, color=GRID_COLOR, linewidth=0.7)

    ax.set_title(
        "Phase C — ILP Sweep: How Many Sequential Ops Can You Chain?\n"
        "(scalar f64 FMA, single core — plateau = FP execution unit ceiling)",
        fontsize=13, color="white", pad=14
    )

    ax.legend(facecolor=PANEL_COLOR, edgecolor="#444", fontsize=10)

    # ── Explanation box ───────────────────────────────────────────────
    explanation = (
        "Reading this plot:\n"
        "  Chain 1 = each FMA depends on previous result\n"
        f"    → throughput limited by FMA LATENCY\n"
        f"  Chain ≥ {plateau_d} = independent chains fill the pipeline\n"
        f"    → throughput approaches PEAK\n"
        f"  Recommendation: unroll loops by ≥ {plateau_d}× in CV kernels"
    )
    ax.text(0.98, 0.05, explanation,
            transform=ax.transAxes, fontsize=8.5, color="#AAAAAA",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc=PANEL_COLOR,
                      ec="#444", alpha=0.9))

    plt.tight_layout()
    plt.savefig(ILP_PLOT, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved: {ILP_PLOT}")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────
# Summary bar chart (peak per kernel type)
# ─────────────────────────────────────────────────────────────────────────
def plot_summary(df):
    # For each kernel_key, take the maximum throughput (= peak with best depth)
    peaks = df.groupby("kernel_key")["gflops_per_core"].max().reset_index()
    peaks = peaks.sort_values("gflops_per_core")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    colors = [KERNEL_COLORS.get(k, "#888888") for k in peaks["kernel_key"]]
    bars = ax.barh(peaks["kernel_key"], peaks["gflops_per_core"],
                   color=colors, edgecolor="#222", height=0.55)

    # Value labels
    for bar, val in zip(bars, peaks["gflops_per_core"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left",
                color="white", fontsize=10)

    ax.set_xlabel("Peak Throughput (GFLOP/s, single core)", fontsize=12, color="white")
    ax.set_title(
        "Phase C — Peak Compute Throughput by Kernel Type",
        fontsize=13, color="white", pad=12
    )
    ax.tick_params(colors="white")
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.7)
    ax.set_xlim(0, peaks["gflops_per_core"].max() * 1.25)

    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved: {SUMMARY_PLOT}")
    plt.close()

def main():
    df = load()
    plot_ilp_sweep(df)
    plot_summary(df)

if __name__ == "__main__":
    main()
