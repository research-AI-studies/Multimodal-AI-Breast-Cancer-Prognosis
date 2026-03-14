"""
Kaplan-Meier survival curve figures (Figures 3-9).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
})


def _plot_two_km(kmf1, kmf2, title, xlabel, ylabel, p_value, save_path,
                 annotation=None):
    """Generic two-curve KM plot with log-rank p."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    kmf1.plot_survival_function(ax=ax, ci_show=True)
    kmf2.plot_survival_function(ax=ax, ci_show=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=True, fontsize=10)

    p_text = f"Log-rank p = {p_value:.4f}" if p_value >= 0.001 else f"Log-rank p < 0.001"
    ax.text(0.98, 0.95, p_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                   edgecolor="gray", alpha=0.8))
    if annotation:
        ax.text(0.98, 0.85, annotation, transform=ax.transAxes, ha="right",
                va="top", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {save_path}")


def plot_km_comorbidity(df, comorb_col, fig_num, label_pos, label_neg,
                        title_suffix, save_dir=None):
    """Plot KM for a binary comorbidity (Figures 3-6)."""
    save_dir = save_dir or config.FIGURES_DIR
    pos = df[comorb_col] == 1
    neg = df[comorb_col] == 0

    kmf_pos = KaplanMeierFitter()
    kmf_pos.fit(df.loc[pos, "survival_time"], df.loc[pos, "event"], label=label_pos)
    kmf_neg = KaplanMeierFitter()
    kmf_neg.fit(df.loc[neg, "survival_time"], df.loc[neg, "event"], label=label_neg)

    lr = logrank_test(df.loc[pos, "survival_time"], df.loc[neg, "survival_time"],
                      df.loc[pos, "event"], df.loc[neg, "event"])

    path = os.path.join(save_dir, f"figure_{fig_num}_km_{comorb_col}.png")
    _plot_two_km(kmf_neg, kmf_pos,
                 f"Figure {fig_num}. KM Survival by {title_suffix}",
                 "Time (months)", "Survival Probability",
                 lr.p_value, path)
    return lr.p_value


def plot_km_threshold(df, col, thresh_low, thresh_high, fig_num,
                      label_low, label_high, title_suffix, save_dir=None):
    """Plot KM for PRO thresholds (Figures 7-8)."""
    save_dir = save_dir or config.FIGURES_DIR
    low = df[col] < thresh_low
    high = df[col] >= thresh_high

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(df.loc[low, "survival_time"], df.loc[low, "event"], label=label_low)
    kmf_high = KaplanMeierFitter()
    kmf_high.fit(df.loc[high, "survival_time"], df.loc[high, "event"], label=label_high)

    if low.sum() > 0 and high.sum() > 0:
        lr = logrank_test(df.loc[low, "survival_time"], df.loc[high, "survival_time"],
                          df.loc[low, "event"], df.loc[high, "event"])
        p = lr.p_value
    else:
        p = 1.0

    path = os.path.join(save_dir, f"figure_{fig_num}_km_{col}.png")
    _plot_two_km(kmf_low, kmf_high,
                 f"Figure {fig_num}. KM Survival by {title_suffix}",
                 "Time (months)", "Survival Probability",
                 p, path)
    return p


def plot_km_risk_groups(df, fig_num=9, save_dir=None):
    """Plot KM by risk score groups — top 20% vs rest (Figure 9)."""
    save_dir = save_dir or config.FIGURES_DIR
    high = df["risk_group"] == "high"
    low = df["risk_group"] == "low"

    kmf_h = KaplanMeierFitter()
    kmf_h.fit(df.loc[high, "survival_time"], df.loc[high, "event"],
              label="High Risk (top 20%)")
    kmf_l = KaplanMeierFitter()
    kmf_l.fit(df.loc[low, "survival_time"], df.loc[low, "event"],
              label="Low Risk (remaining 80%)")

    lr = logrank_test(df.loc[high, "survival_time"], df.loc[low, "survival_time"],
                      df.loc[high, "event"], df.loc[low, "event"])

    path = os.path.join(save_dir, f"figure_{fig_num}_km_risk_groups.png")
    _plot_two_km(kmf_l, kmf_h,
                 f"Figure {fig_num}. KM Survival by Multimodal Risk Score",
                 "Time (months)", "Survival Probability",
                 lr.p_value, path)
    return lr.p_value


def generate_all_km_figures(df, save_dir=None):
    """Generate all KM figures (3-9)."""
    p_values = {}
    p_values["fig3_uti"] = plot_km_comorbidity(
        df, "comorb_uti", 3, "UTI (comorb_uti=1)", "No UTI (comorb_uti=0)",
        "UTI Status", save_dir)
    p_values["fig4_diabetes"] = plot_km_comorbidity(
        df, "comorb_diabetes", 4, "Diabetes (comorb_diabetes=1)",
        "No Diabetes (comorb_diabetes=0)", "Diabetes Status", save_dir)
    p_values["fig5_gi"] = plot_km_comorbidity(
        df, "comorb_gastrointestinal", 5,
        "GI (comorb_gastrointestinal=1)", "No GI (comorb_gastrointestinal=0)",
        "Gastrointestinal Status", save_dir)
    p_values["fig6_depression"] = plot_km_comorbidity(
        df, "comorb_depression", 6, "Depression (comorb_depression=1)",
        "No Depression (comorb_depression=0)", "Depression Status", save_dir)

    p_values["fig7_ql"] = plot_km_threshold(
        df, "ql", 50, 70, 7, "ql < 50 (poor)", "ql >= 70 (good)",
        "Global Health Status (ql)", save_dir)
    p_values["fig8_fa"] = plot_km_threshold(
        df, "fa", 70, 70, 8, "fa >= 70 (high fatigue)", "fa < 70 (low fatigue)",
        "Fatigue (fa)", save_dir)

    if "risk_group" in df.columns:
        p_values["fig9_risk"] = plot_km_risk_groups(df, save_dir=save_dir)

    return p_values
