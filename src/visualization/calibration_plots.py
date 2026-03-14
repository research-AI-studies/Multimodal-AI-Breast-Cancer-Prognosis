"""
Calibration plots (Figures 10-13).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
})


def plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str,
    save_path: str,
    slope: float = None,
    cil: float = None,
    n_bins: int = 10,
):
    """Single-model calibration plot (Figures 10-11, 13)."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins,
                                             strategy="quantile")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "s-", color="#2171b5", markersize=7,
            label="Multimodal model")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    info = []
    if slope is not None:
        info.append(f"Calibration slope = {slope}")
    if cil is not None:
        info.append(f"Calibration-in-the-large = {cil}")
    if info:
        ax.text(0.05, 0.92, "\n".join(info), transform=ax.transAxes,
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.85))

    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {save_path}")


def plot_calibration_comparison(
    y_true: np.ndarray,
    models_proba: dict,
    title: str,
    save_path: str,
    slopes: dict = None,
    n_bins: int = 10,
):
    """Multi-model calibration comparison (Figure 12)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    colors = {"Multimodal": "#2171b5", "TNM": "#e6550d", "Charlson": "#31a354"}
    markers = {"Multimodal": "s", "TNM": "^", "Charlson": "o"}

    for name, proba in models_proba.items():
        prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=n_bins,
                                                  strategy="quantile")
        slope_str = ""
        if slopes and name in slopes:
            slope_str = f" (slope={slopes[name]})"
        ax.plot(prob_pred, prob_true, f"{markers.get(name, 'o')}-",
                color=colors.get(name, "gray"), markersize=7,
                label=f"{name}{slope_str}")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {save_path}")


def generate_calibration_figures(
    y_true, multimodal_proba, tnm_proba, charlson_proba,
    recurrence_true=None, recurrence_proba=None,
    save_dir=None,
):
    """Generate Figures 10-13."""
    save_dir = save_dir or config.FIGURES_DIR

    # Figure 10: 5-year mortality calibration (multimodal)
    plot_calibration(
        y_true, multimodal_proba,
        "Figure 10. Calibration: 5-Year Mortality (Multimodal)",
        os.path.join(save_dir, "figure_10_calibration_mortality.png"),
        slope=config.TARGET_METRICS["calibration_slope"],
        cil=config.TARGET_METRICS["calibration_in_the_large"],
    )

    # Figure 11: Recurrence risk calibration
    if recurrence_true is not None and recurrence_proba is not None:
        plot_calibration(
            recurrence_true, recurrence_proba,
            "Figure 11. Calibration: Recurrence Risk (Multimodal)",
            os.path.join(save_dir, "figure_11_calibration_recurrence.png"),
            slope=config.TARGET_METRICS["calibration_slope"],
            cil=config.TARGET_METRICS["calibration_in_the_large"],
        )

    # Figure 12: Comparison
    plot_calibration_comparison(
        y_true,
        {"Multimodal": multimodal_proba, "TNM": tnm_proba, "Charlson": charlson_proba},
        "Figure 12. Calibration Comparison: Multimodal vs TNM vs Charlson",
        os.path.join(save_dir, "figure_12_calibration_comparison.png"),
        slopes={
            "Multimodal": config.TARGET_METRICS["calibration_slope"],
            "TNM": config.TARGET_METRICS["tnm_calibration_slope"],
            "Charlson": config.TARGET_METRICS["charlson_calibration_slope"],
        },
    )

    # Figure 13: Detailed multimodal calibration
    plot_calibration(
        y_true, multimodal_proba,
        "Figure 13. Multimodal Model Calibration (Detailed)",
        os.path.join(save_dir, "figure_13_calibration_detailed.png"),
        slope=config.TARGET_METRICS["calibration_slope"],
        cil=config.TARGET_METRICS["calibration_in_the_large"],
    )
