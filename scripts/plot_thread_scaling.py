#!/usr/bin/env python3
"""
plot_thread_scaling.py — Phase A: CPU Thread Scaling
Reads: results/thread_scaling.csv
Saves: results/plot_thread_scaling.png

Dual-axis plot:
  Left  axis  — Throughput (GFLOP/s) and ideal linear scaling
  Right axis  — Efficiency (%)
Annotates the saturation point (first thread count where efficiency < 90%).
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CSV_PATH  = "results/thread_scaling.csv"
PLOT_PATH = "results/plot_thread_scaling.png"

# ── Style ──────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
ACCENT_BLUE   = "#4FC3F7"
ACCENT_ORANGE = "#FFB74D"
ACCENT_RED    = "#EF5350"
GRID_COLOR    = "#2a2a2a"
BG_COLOR      = "#0d0d0d"
PANEL_COLOR   = "#141414"

def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run 'make run ARG=A' first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    required = {"threads", "throughput_gflops", "efficiency_pct", "speedup"}
    if not required.issubset(df.columns):
        print(f"ERROR: Expected columns {required}, got {list(df.columns)}")
        sys.exit(1)
    return df

def find_saturation(df, threshold=90.0):
    """Return first thread count where efficiency drops below threshold."""
    below = df[df["efficiency_pct"] < threshold]
    if below.empty:
        return None
    return int(below.iloc[0]["threads"])

def main():
    df = load_data()
    sat = find_saturation(df)

    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    ax2 = ax1.twinx()
    fig.patch.set_facecolor(BG_COLOR)
    ax1.set_facecolor(PANEL_COLOR)

    threads = df["threads"].values
    gflops  = df["throughput_gflops"].values
    eff     = df["efficiency_pct"].values

    # ── Ideal linear scaling ───────────────────────────────────────────
    ideal = gflops[0] * threads
    ax1.plot(threads, ideal, color="#555555", linewidth=1.5,
             linestyle="--", label="Ideal linear scaling", zorder=1)

    # ── Actual throughput ──────────────────────────────────────────────
    ax1.plot(threads, gflops, color=ACCENT_BLUE, linewidth=2.5,
             marker="o", markersize=8, label="Measured throughput",
             zorder=3)
    ax1.fill_between(threads, gflops, alpha=0.15, color=ACCENT_BLUE)

    # ── Efficiency ────────────────────────────────────────────────────
    ax2.plot(threads, eff, color=ACCENT_ORANGE, linewidth=2.0,
             marker="s", markersize=7, linestyle=":", label="Efficiency (%)",
             zorder=2)
    ax2.axhline(y=90, color=ACCENT_ORANGE, linewidth=0.8,
                linestyle="--", alpha=0.6, label="90% threshold")

    # ── Saturation annotation ─────────────────────────────────────────
    if sat is not None:
        sat_row = df[df["threads"] == sat].iloc[0]
        ax1.axvline(x=sat, color=ACCENT_RED, linewidth=1.5,
                    linestyle="-.", alpha=0.9)
        ax1.annotate(
            f"Saturation\npoint = {sat} threads",
            xy=(sat, sat_row["throughput_gflops"]),
            xytext=(sat + 0.4, sat_row["throughput_gflops"] * 0.6),
            color=ACCENT_RED,
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_COLOR,
                      ec=ACCENT_RED, alpha=0.9)
        )

    # ── Labels & grid ─────────────────────────────────────────────────
    ax1.set_xlabel("Thread Count", fontsize=12, color="white")
    ax1.set_ylabel("Throughput (GFLOP/s)", fontsize=12, color=ACCENT_BLUE)
    ax2.set_ylabel("Efficiency (%)", fontsize=12, color=ACCENT_ORANGE)
    ax1.tick_params(colors="white")
    ax2.tick_params(colors=ACCENT_ORANGE)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylim(0, 115)
    ax1.grid(True, color=GRID_COLOR, linewidth=0.7)

    # ── Title ─────────────────────────────────────────────────────────
    ax1.set_title(
        "Phase A — CPU Thread Scaling\n"
        "(compute-bound FMA kernel, double precision, 8-chain ILP)",
        fontsize=13, color="white", pad=14
    )

    # ── Legend (combined from both axes) ──────────────────────────────
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               facecolor=PANEL_COLOR, edgecolor="#444", fontsize=9,
               loc="upper left")

    # ── Data table inset ──────────────────────────────────────────────
    col_labels = ["Threads", "GFLOP/s", "Speedup", "Efficiency"]
    table_data = [[
        f"{int(r.threads)}",
        f"{r.throughput_gflops:.3f}",
        f"{r.speedup:.2f}×",
        f"{r.efficiency_pct:.1f}%"
    ] for _, r in df.iterrows()]
    table = ax1.table(
        cellText=table_data, colLabels=col_labels,
        cellLoc="center", loc="lower right",
        bbox=[0.60, 0.02, 0.38, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(PANEL_COLOR if row > 0 else "#1e1e1e")
        cell.set_edgecolor("#333")
        cell.set_text_props(color="white")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR)
    print(f"Saved: {PLOT_PATH}")
    plt.close()

if __name__ == "__main__":
    main()
