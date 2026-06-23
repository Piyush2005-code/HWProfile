#!/usr/bin/env python3
"""
plot_sync_overhead.py — Phase D: Synchronization Overhead
Reads: results/sync_overhead.csv
Saves: results/plot_sync_overview.png
        results/plot_sync_contention.png

Overview plot:
  Horizontal bar chart of all primitive × thread combinations.
  Shows the full spectrum from atomic (nanoseconds) to condvar (microseconds).

Contention scaling plot:
  For mutex and atomic, shows how latency grows with thread count.
  Helps the developer decide between mutex and atomic for their queue.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CSV_PATH      = "results/sync_overhead.csv"
OVERVIEW_PLOT = "results/plot_sync_overview.png"
SCALING_PLOT  = "results/plot_sync_contention.png"

plt.style.use("dark_background")
BG_COLOR    = "#0d0d0d"
PANEL_COLOR = "#141414"
GRID_COLOR  = "#2a2a2a"

PRIM_COLORS = {
    "mutex_uncontended":  "#4CAF50",
    "mutex_contended":    "#EF5350",
    "atomic_fetch_add":   "#4FC3F7",
    "pthread_barrier":    "#FFB74D",
    "condvar_handoff":    "#CE93D8",
}

PRIM_LABELS = {
    "mutex_uncontended":  "Mutex (uncontended)",
    "mutex_contended":    "Mutex (contended)",
    "atomic_fetch_add":   "Atomic fetch_add",
    "pthread_barrier":    "pthread barrier",
    "condvar_handoff":    "Condvar hand-off (pipeline stage)",
}

def load():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run 'make run ARG=D' first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    expected = {"primitive", "threads", "latency_ns"}
    if not expected.issubset(df.columns):
        print(f"ERROR: unexpected columns: {list(df.columns)}")
        sys.exit(1)
    return df

# ─────────────────────────────────────────────────────────────────────────
# Overview bar chart
# ─────────────────────────────────────────────────────────────────────────
def plot_overview(df):
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    # Build label for each row
    rows = []
    for _, r in df.iterrows():
        prim = r["primitive"]
        t    = int(r["threads"])
        ns   = r["latency_ns"]
        if t > 1:
            label = f"{PRIM_LABELS.get(prim, prim)} [{t}T]"
        else:
            label = PRIM_LABELS.get(prim, prim)
        color = PRIM_COLORS.get(prim, "#888888")
        rows.append((label, ns, color))

    # Sort by latency ascending
    rows.sort(key=lambda x: x[1])
    labels  = [r[0] for r in rows]
    values  = [r[1] for r in rows]
    colors  = [r[2] for r in rows]

    bars = ax.barh(labels, values, color=colors, edgecolor="#222", height=0.6)

    # Value labels
    for bar, val in zip(bars, values):
        unit = "ns" if val < 1000 else "µs"
        disp = val if val < 1000 else val / 1000
        ax.text(val * 1.02, bar.get_y() + bar.get_height()/2,
                f"{disp:.1f} {unit}", va="center", ha="left",
                color="white", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Latency (ns, log scale)", fontsize=12, color="white")
    ax.tick_params(colors="white", axis="both")
    ax.grid(True, axis="x", which="both", color=GRID_COLOR, linewidth=0.6)
    ax.set_xlim(1, max(values) * 4)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0f} ns" if v < 1000
                             else f"{v/1000:.1f} µs")
    )

    ax.set_title(
        "Phase D — Synchronization Primitive Latency Overview\n"
        "(all primitives × thread counts — log scale)",
        fontsize=13, color="white", pad=12
    )

    # ── Guideline annotation ───────────────────────────────────────────
    ax.text(0.98, 0.02,
            "CV/ML pipeline budget:\n"
            "  Condvar hand-off ≈ latency of one\n"
            "  frame transfer between pipeline stages.\n"
            "  Keep stage count low if this dominates.",
            transform=ax.transAxes, fontsize=8.5, color="#AAAAAA",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc=PANEL_COLOR,
                      ec="#444", alpha=0.9))

    plt.tight_layout()
    plt.savefig(OVERVIEW_PLOT, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved: {OVERVIEW_PLOT}")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────
# Contention scaling: mutex & atomic vs thread count
# ─────────────────────────────────────────────────────────────────────────
def plot_contention_scaling(df):
    prims_to_plot = ["mutex_uncontended", "mutex_contended", "atomic_fetch_add"]
    available = [p for p in prims_to_plot if p in df["primitive"].values]
    if not available:
        print("SKIP: no contention data found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    for prim in available:
        sub = df[df["primitive"] == prim].sort_values("threads")
        if sub.empty: continue
        color = PRIM_COLORS.get(prim, "#888")
        label = PRIM_LABELS.get(prim, prim)
        ax.plot(sub["threads"], sub["latency_ns"],
                color=color, linewidth=2.2, marker="o", markersize=8,
                label=label)
        ax.fill_between(sub["threads"], sub["latency_ns"],
                        alpha=0.10, color=color)

    ax.set_xlabel("Thread Count (competing for same resource)", fontsize=12, color="white")
    ax.set_ylabel("Latency per operation (ns)", fontsize=12, color="white")
    ax.tick_params(colors="white")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, color=GRID_COLOR, linewidth=0.7)

    ax.set_title(
        "Phase D — Lock Contention: Latency vs Number of Competing Threads\n"
        "(choose atomic over mutex when threads > 2 and operation is simple)",
        fontsize=12, color="white", pad=12
    )
    ax.legend(facecolor=PANEL_COLOR, edgecolor="#444", fontsize=10)

    plt.tight_layout()
    plt.savefig(SCALING_PLOT, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved: {SCALING_PLOT}")
    plt.close()

def main():
    df = load()
    plot_overview(df)
    plot_contention_scaling(df)

if __name__ == "__main__":
    main()
