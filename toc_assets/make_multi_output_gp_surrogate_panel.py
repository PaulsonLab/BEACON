#!/usr/bin/env python3
"""Generate a local-only multi-output GP surrogate panel for a TOC figure."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "beacon_toc_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
XDG_CACHE_HOME = Path(tempfile.gettempdir()) / "beacon_toc_cache"
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FIGSIZE = (2.2, 4.0)
DPI = 600
SEED = 20260529
SHARED_X_NEXT = 0.64

OUTPUT_DIR = Path(__file__).resolve().parent
BASE_NAME = "multi_output_gp_surrogate_panel"


def gp_like_sample(x: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Draw a smooth, low-amplitude function for a plausible posterior sample."""
    centers = np.linspace(0.04, 0.96, 10)
    weights = rng.normal(0.0, 1.0, size=centers.size)
    width = 0.16
    basis = np.exp(-0.5 * ((x[:, None] - centers[None, :]) / width) ** 2)
    sample = basis @ weights
    sample -= sample.mean()
    sample /= max(np.max(np.abs(sample)), 1e-9)
    return scale * sample


def panel_data(
    x: np.ndarray, rng: np.random.Generator
) -> list[dict[str, np.ndarray | str | tuple[float, float, float, float]]]:
    mean_a = (
        0.22
        + 0.27 * np.sin(2.05 * np.pi * (x - 0.06))
        + 0.11 * np.cos(4.9 * np.pi * x + 0.25)
        + 0.20 * x
    )
    unc_a = 0.14 + 0.13 * (np.abs(x - 0.48) ** 1.25) + 0.04 * np.sin(2.2 * np.pi * x + 0.4) ** 2

    mean_b = (
        -0.04
        + 0.33 * np.cos(1.72 * np.pi * (x + 0.12))
        - 0.16 * np.sin(5.45 * np.pi * x - 0.15)
        + 0.11 * (x - 0.5)
    )
    unc_b = 0.13 + 0.11 * (np.abs(x - 0.54) ** 1.18) + 0.045 * np.cos(2.7 * np.pi * x - 0.3) ** 2

    x_obs_a = np.array([0.08, 0.18, 0.34, 0.49, 0.67, 0.82])
    x_obs_b = np.array([0.11, 0.27, 0.43, 0.58, 0.73, 0.88])

    y_obs_a = np.interp(x_obs_a, x, mean_a) + rng.normal(0.0, 0.065, size=x_obs_a.size)
    y_obs_b = np.interp(x_obs_b, x, mean_b) + rng.normal(0.0, 0.065, size=x_obs_b.size)

    return [
        {
            "mean": mean_a,
            "unc": unc_a,
            "sample": mean_a + gp_like_sample(x, rng, 0.10),
            "x_obs": x_obs_a,
            "y_obs": y_obs_a,
            "x_next": SHARED_X_NEXT,
            "y_next": np.interp(SHARED_X_NEXT, x, mean_a),
            "curve": "#007EA7",
            "band": "#35BFE0",
            "sample_color": "#259BC2",
            "ylim": (-0.35, 0.95),
        },
        {
            "mean": mean_b,
            "unc": unc_b,
            "sample": mean_b + gp_like_sample(x, rng, 0.095),
            "x_obs": x_obs_b,
            "y_obs": y_obs_b,
            "x_next": SHARED_X_NEXT,
            "y_next": np.interp(SHARED_X_NEXT, x, mean_b),
            "curve": "#009C8D",
            "band": "#48D0C0",
            "sample_color": "#1EA99A",
            "ylim": (-0.68, 0.72),
        },
    ]


def style_axis(ax: plt.Axes, dark: bool) -> None:
    spine_color = "#B7C0CA" if not dark else "#5B6673"
    tick_color = "#A9B0BA" if not dark else "#758293"
    grid_color = "#DDE3EA" if not dark else "#25303B"

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=tick_color, labelbottom=False, length=0, width=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=grid_color, linewidth=0.45, alpha=0.7)
    ax.grid(axis="y", color=grid_color, linewidth=0.35, alpha=0.35)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(spine_color)
        ax.spines[side].set_linewidth(0.7)


def draw_panel(dark: bool = False) -> plt.Figure:
    rng = np.random.default_rng(SEED)
    x = np.linspace(0.0, 1.0, 520)
    data = panel_data(x, rng)

    if dark:
        face = "#070A0F"
        obs_face = "#D9E0E8"
        obs_edge = "#101820"
        gold = "#F0C04A"
        gold_edge = "#FFF0B0"
        alpha_band = 0.28
    else:
        face = "none"
        obs_face = "#424B57"
        obs_edge = "#FFFFFF"
        gold = "#D79A00"
        gold_edge = "#7A5700"
        alpha_band = 0.28

    fig, axes = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        dpi=DPI,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.02},
    )
    fig.patch.set_alpha(0.0 if not dark else 1.0)
    fig.patch.set_facecolor(face)

    for ax, values in zip(axes, data, strict=True):
        mean = values["mean"]
        unc = values["unc"]
        sample = values["sample"]
        curve = str(values["curve"])
        band = str(values["band"])

        ax.set_facecolor(face)
        ax.fill_between(x, mean - 1.75 * unc, mean + 1.75 * unc, color=band, alpha=alpha_band, linewidth=0)
        ax.plot(x, sample, color=str(values["sample_color"]), linewidth=1.0, alpha=0.42, solid_capstyle="round")
        ax.plot(x, mean, color=curve, linewidth=2.45, solid_capstyle="round")
        ax.scatter(
            values["x_obs"],
            values["y_obs"],
            s=18,
            c=obs_face,
            edgecolors=obs_edge,
            linewidths=0.55,
            zorder=5,
        )
        ax.scatter(
            [values["x_next"]],
            [values["y_next"]],
            s=34,
            c=gold,
            edgecolors=gold_edge,
            linewidths=0.75,
            zorder=6,
        )
        ax.set_ylim(*values["ylim"])
        style_axis(ax, dark=dark)

    fig.subplots_adjust(left=0.095, right=0.99, top=0.975, bottom=0.035)
    return fig


def save_outputs() -> None:
    transparent_fig = draw_panel(dark=False)
    transparent_fig.savefig(
        OUTPUT_DIR / f"{BASE_NAME}.png",
        dpi=DPI,
        transparent=True,
        metadata={"Creator": "BEACON local TOC asset script"},
    )
    transparent_fig.savefig(
        OUTPUT_DIR / f"{BASE_NAME}.svg",
        transparent=True,
        metadata={"Creator": "BEACON local TOC asset script"},
    )
    transparent_fig.savefig(
        OUTPUT_DIR / f"{BASE_NAME}.pdf",
        transparent=True,
        metadata={"Creator": "BEACON local TOC asset script"},
    )
    plt.close(transparent_fig)

    dark_fig = draw_panel(dark=True)
    dark_fig.savefig(
        OUTPUT_DIR / f"{BASE_NAME}_dark.png",
        dpi=DPI,
        facecolor=dark_fig.get_facecolor(),
        edgecolor="none",
        metadata={"Creator": "BEACON local TOC asset script"},
    )
    plt.close(dark_fig)


if __name__ == "__main__":
    save_outputs()
    print(f"Wrote {BASE_NAME} assets to {OUTPUT_DIR}")
    print(f"Figure size: {FIGSIZE[0]:.1f} x {FIGSIZE[1]:.1f} in at {DPI} dpi")
