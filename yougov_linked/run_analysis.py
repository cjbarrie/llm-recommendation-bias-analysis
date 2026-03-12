#!/usr/bin/env python3
"""
Comprehensive Analysis Pipeline — YouGov Linked Pipeline

Adapts run_comprehensive_analysis.py for the YouGov linked dataset.
Differences from main pipeline:
- Single dataset (yougov) — model dimension replaces dataset dimension
- Reads from yougov_linked/outputs/experiments/{model_slug}/post_level_data.csv
- Outputs to yougov_linked/analysis_outputs/

Generates same 4 visualization types:
1. Feature distributions
2. Bias magnitude heatmaps (by model, by prompt, fully aggregated)
3. Directional bias plots (by model, by prompt, fully aggregated)
4. Feature importance (Random Forest + SHAP)

Usage:
    python run_analysis.py
    python run_analysis.py --output-dir analysis_outputs/
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import shap

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# =============================================================================
# CONFIGURATION — mirrors run_comprehensive_analysis.py
# =============================================================================

FEATURES = {
    "author":       ["author_gender", "author_political_leaning", "author_is_minority"],
    "text_metrics": ["text_length", "avg_word_length"],
    "sentiment":    ["sentiment_polarity", "sentiment_subjectivity"],
    "style":        ["has_emoji", "has_hashtag", "has_mention", "has_url"],
    "content":      ["polarization_score", "controversy_level", "primary_topic"],
    "toxicity":     ["toxicity", "severe_toxicity"],
}

ALL_FEATURES = [f for group in FEATURES.values() for f in group]

FEATURE_TYPES = {
    "author_gender":             "categorical",
    "author_political_leaning":  "categorical",
    "author_is_minority":        "categorical",
    "text_length":               "numerical",
    "avg_word_length":           "numerical",
    "sentiment_polarity":        "numerical",
    "sentiment_subjectivity":    "numerical",
    "has_emoji":                 "binary",
    "has_hashtag":               "binary",
    "has_mention":               "binary",
    "has_url":                   "binary",
    "polarization_score":        "numerical",
    "controversy_level":         "categorical",
    "primary_topic":             "categorical",
    "toxicity":                  "numerical",
    "severe_toxicity":           "numerical",
}

CATEGORY_ORDERS = {
    "author_political_leaning": ["left", "center-left", "center", "center-right", "right", "unknown"],
    "author_gender":            ["male", "female", "non-binary", "unknown"],
    "author_is_minority":       ["no", "yes", "unknown"],
    "controversy_level":        ["low", "medium", "high"],
    "has_emoji":   [0, 1],
    "has_hashtag": [0, 1],
    "has_mention": [0, 1],
    "has_url":     [0, 1],
}

PROMPT_STYLES = ["general", "popular", "engaging", "informative", "controversial", "neutral"]

MODEL_COLORS = {
    "openai_gpt-4o-mini":         "#2F2F2F",
    "anthropic_claude-sonnet-4-5": "#4A90E2",
    "google_gemini-2.0-flash-001": "#FF6B35",
}
MODEL_LABELS = {
    "openai_gpt-4o-mini":         "GPT-4o-mini",
    "anthropic_claude-sonnet-4-5": "Claude Sonnet 4.5",
    "google_gemini-2.0-flash-001": "Gemini 2.0 Flash",
}

DIVERGING_CMAP = "RdYlBu_r"
SEQ_CMAP = "YlOrRd"


# =============================================================================
# DATA LOADING
# =============================================================================

def get_experiment_dirs(experiments_base: Path) -> dict[str, Path]:
    """Return {model_slug: path} for all completed experiments."""
    result = {}
    if not experiments_base.exists():
        return result
    for d in sorted(experiments_base.iterdir()):
        csv = d / "post_level_data.csv"
        if d.is_dir() and csv.exists():
            result[d.name] = d
    return result


def load_experiment_data(exp_dir: Path) -> Optional[pd.DataFrame]:
    """Load post_level_data.csv for one model."""
    csv = exp_dir / "post_level_data.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv, low_memory=False)
    return df


def load_pool_data(exp_dir: Path) -> Optional[pd.DataFrame]:
    """Load unique pool posts (selected=0) for a model experiment."""
    df = load_experiment_data(exp_dir)
    if df is None:
        return None
    pool = df[df["selected"] == 0].drop_duplicates(subset="original_index").copy()
    return pool


# =============================================================================
# STATISTICAL UTILITIES (copied from run_comprehensive_analysis.py)
# =============================================================================

def compute_cohens_d(group1, group2) -> float:
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compute_cramers_v(pool_vals, rec_vals) -> float:
    try:
        pool_vals = pool_vals.reset_index(drop=True)
        rec_vals  = rec_vals.reset_index(drop=True)
        combined  = pd.concat([pool_vals, rec_vals], ignore_index=True)
        labels    = pd.Series(["pool"] * len(pool_vals) + ["rec"] * len(rec_vals))
        contingency = pd.crosstab(combined, labels)
        if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
            return 0.0
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))
    except Exception:
        return 0.0


def compute_bias_metric(pool_vals, rec_vals, feature_type) -> tuple[float, float, str]:
    """Returns (bias_value, p_value, metric_name)."""
    pool_vals = pool_vals.dropna()
    rec_vals  = rec_vals.dropna()
    if len(pool_vals) < 10 or len(rec_vals) < 10:
        return 0.0, 1.0, "insufficient_data"

    if feature_type == "numerical":
        if pool_vals.std() == 0 and rec_vals.std() == 0:
            return 0.0, 1.0, "Cohen's d (no variance)"
        bias = compute_cohens_d(rec_vals, pool_vals)
        try:
            _, p_val = stats.ttest_ind(rec_vals, pool_vals, equal_var=False)
            if np.isnan(p_val):
                p_val = 1.0
        except Exception:
            p_val = 1.0
        return bias, p_val, "Cohen's d"

    elif feature_type in ("categorical", "binary"):
        if len(pool_vals.unique()) <= 1 and len(rec_vals.unique()) <= 1:
            return 0.0, 1.0, "Cramér's V (no variance)"
        try:
            bias = compute_cramers_v(pool_vals, rec_vals)
            combined = pd.concat(
                [pool_vals.reset_index(drop=True), rec_vals.reset_index(drop=True)],
                ignore_index=True,
            )
            labels = pd.Series(["pool"] * len(pool_vals) + ["rec"] * len(rec_vals))
            contingency = pd.crosstab(combined, labels)
            _, p_val, _, _ = chi2_contingency(contingency)
            if np.isnan(p_val):
                p_val = 1.0
            return bias, p_val, "Cramér's V"
        except Exception:
            return 0.0, 1.0, "Cramér's V (error)"

    return 0.0, 1.0, "unknown"


def standardize_categories(series: pd.Series, feature_name: str) -> pd.Series:
    """Standardise category labels and apply known ordering."""
    if feature_name in CATEGORY_ORDERS:
        series = series.astype(str).str.lower()
        valid = [str(c).lower() for c in CATEGORY_ORDERS[feature_name]]
        series = pd.Categorical(series, categories=valid, ordered=True)
    return series


# =============================================================================
# 1. FEATURE DISTRIBUTIONS
# =============================================================================

def generate_feature_distributions(
    pool_data: dict[str, pd.DataFrame],
    output_dir: Path,
):
    """Plot feature distributions for each model's pool."""
    dist_dir = output_dir / "1_distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)

    model_slugs = list(pool_data.keys())
    n_models = len(model_slugs)
    if n_models == 0:
        print("  No pool data available for distributions.")
        return

    for feature in ALL_FEATURES:
        ftype = FEATURE_TYPES.get(feature, "numerical")
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)

        for col_idx, slug in enumerate(model_slugs):
            ax = axes[0][col_idx]
            df = pool_data[slug]
            if feature not in df.columns:
                ax.set_visible(False)
                continue

            values = df[feature].dropna()
            label = MODEL_LABELS.get(slug, slug)
            color = MODEL_COLORS.get(slug, "#666666")

            if ftype == "numerical":
                ax.hist(values, bins=30, color=color, alpha=0.7, edgecolor="white")
                ax.axvline(values.mean(), color="red", linestyle="--", linewidth=1, label=f"mean={values.mean():.2f}")
                ax.legend(fontsize=7)
            else:
                vc = standardize_categories(values, feature).value_counts()
                vc.plot(kind="bar", ax=ax, color=color, alpha=0.8)
                ax.tick_params(axis="x", rotation=45)

            ax.set_title(f"{label}\n(n={len(values):,})", fontsize=9)
            ax.set_xlabel(feature, fontsize=8)

        fig.suptitle(f"Distribution: {feature}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(dist_dir / f"{feature}_distribution.png", bbox_inches="tight")
        plt.close()

    print(f"  Saved {len(ALL_FEATURES)} distribution plots → {dist_dir}")


# =============================================================================
# 2. BIAS HEATMAPS
# =============================================================================

def compute_all_bias(
    experiment_dirs: dict[str, Path],
) -> pd.DataFrame:
    """
    Compute bias metrics for all model × prompt combinations.

    Returns DataFrame with columns:
    [model, prompt_style, feature, bias_value, p_value, metric]
    """
    records = []

    for slug, exp_dir in experiment_dirs.items():
        df = load_experiment_data(exp_dir)
        if df is None:
            continue

        for style in PROMPT_STYLES:
            style_df = df[df["prompt_style"] == style]
            if len(style_df) == 0:
                continue

            pool_df = style_df[style_df["selected"] == 0]
            rec_df  = style_df[style_df["selected"] == 1]

            for feature in ALL_FEATURES:
                if feature not in style_df.columns:
                    continue
                ftype = FEATURE_TYPES.get(feature, "numerical")
                bias, p_val, metric = compute_bias_metric(
                    pool_df[feature], rec_df[feature], ftype
                )
                records.append({
                    "model":        slug,
                    "prompt_style": style,
                    "feature":      feature,
                    "bias_value":   bias,
                    "p_value":      p_val,
                    "metric":       metric,
                    "significant":  p_val < 0.05,
                })

    return pd.DataFrame(records)


def significance_marker(sig_fraction: float) -> str:
    if sig_fraction > 0.75:
        return "***"
    elif sig_fraction > 0.60:
        return "**"
    elif sig_fraction > 0.50:
        return "*"
    return ""


def plot_heatmap(
    matrix: pd.DataFrame,
    sig_matrix: Optional[pd.DataFrame],
    title: str,
    output_path: Path,
    vmax: float = 0.3,
):
    """Plot a bias heatmap with optional significance markers."""
    fig, ax = plt.subplots(figsize=(max(10, matrix.shape[1] * 0.9), max(6, matrix.shape[0] * 0.5)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=SEQ_CMAP,
        vmin=0,
        vmax=vmax,
        annot=False,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Bias magnitude"},
    )

    # Overlay significance markers
    if sig_matrix is not None:
        for row_i, row_label in enumerate(matrix.index):
            for col_j, col_label in enumerate(matrix.columns):
                val = matrix.loc[row_label, col_label]
                if pd.isna(val):
                    continue
                marker = significance_marker(sig_matrix.loc[row_label, col_label])
                if marker:
                    ax.text(
                        col_j + 0.5, row_i + 0.5, marker,
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color="black",
                    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_bias_heatmaps(bias_df: pd.DataFrame, output_dir: Path):
    """Generate bias heatmaps aggregated in different ways."""
    heatmap_dir = output_dir / "2_bias_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    if bias_df.empty:
        print("  No bias data available.")
        return

    # --- Fully aggregated ---
    agg = bias_df.groupby("feature")["bias_value"].mean().reindex(ALL_FEATURES)
    sig = bias_df.groupby("feature")["significant"].mean().reindex(ALL_FEATURES)
    mat = pd.DataFrame({"bias": agg, "sig": sig})
    mat_plot = mat[["bias"]].T.rename(index={"bias": "All models & prompts"})
    sig_plot = mat[["sig"]].T.rename(index={"sig": "All models & prompts"})
    plot_heatmap(mat_plot, sig_plot, "Bias Magnitude — Fully Aggregated",
                 heatmap_dir / "fully_aggregated.png")

    # --- By model ---
    for slug in bias_df["model"].unique():
        sub = bias_df[bias_df["model"] == slug]
        agg = sub.groupby("feature")["bias_value"].mean().reindex(ALL_FEATURES).fillna(0)
        sig = sub.groupby("feature")["significant"].mean().reindex(ALL_FEATURES).fillna(0)
        label = MODEL_LABELS.get(slug, slug)
        mat = pd.DataFrame({"bias": agg}).T.rename(index={"bias": label})
        sig_mat = pd.DataFrame({"sig": sig}).T.rename(index={"sig": label})
        plot_heatmap(mat, sig_mat, f"Bias Magnitude — {label}",
                     heatmap_dir / f"model_{slug}.png")

    # --- Aggregated by model (all prompts) ---
    pivot = bias_df.groupby(["model", "feature"])["bias_value"].mean().unstack(fill_value=0)
    sig_pivot = bias_df.groupby(["model", "feature"])["significant"].mean().unstack(fill_value=0)
    pivot = pivot.reindex(columns=ALL_FEATURES, fill_value=0)
    sig_pivot = sig_pivot.reindex(columns=ALL_FEATURES, fill_value=0)
    pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]
    sig_pivot.index = pivot.index
    plot_heatmap(pivot, sig_pivot, "Bias Magnitude — By Model",
                 heatmap_dir / "aggregated_by_model.png", vmax=0.4)

    # --- Aggregated by prompt style ---
    pivot = bias_df.groupby(["prompt_style", "feature"])["bias_value"].mean().unstack(fill_value=0)
    sig_pivot = bias_df.groupby(["prompt_style", "feature"])["significant"].mean().unstack(fill_value=0)
    pivot = pivot.reindex(columns=ALL_FEATURES, fill_value=0)
    sig_pivot = sig_pivot.reindex(columns=ALL_FEATURES, fill_value=0)
    plot_heatmap(pivot, sig_pivot, "Bias Magnitude — By Prompt Style",
                 heatmap_dir / "aggregated_by_prompt.png", vmax=0.4)

    # --- Disaggregated by prompt style (model × feature per prompt) ---
    for style in PROMPT_STYLES:
        sub = bias_df[bias_df["prompt_style"] == style]
        if sub.empty:
            continue
        pivot = sub.groupby(["model", "feature"])["bias_value"].mean().unstack(fill_value=0)
        sig_pivot = sub.groupby(["model", "feature"])["significant"].mean().unstack(fill_value=0)
        pivot = pivot.reindex(columns=ALL_FEATURES, fill_value=0)
        sig_pivot = sig_pivot.reindex(columns=ALL_FEATURES, fill_value=0)
        pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]
        sig_pivot.index = pivot.index
        plot_heatmap(pivot, sig_pivot, f"Bias Magnitude — Prompt: {style.capitalize()}",
                     heatmap_dir / f"disaggregated_prompt_{style}.png", vmax=0.4)

    # =========================================================================
    # CATEGORY-AGGREGATED HEATMAPS
    # Normalize bias_value 0-1 within each feature, then average within category
    # =========================================================================

    # Build feature → category reverse mapping
    feature_to_category = {}
    for cat, feats in FEATURES.items():
        for f in feats:
            feature_to_category[f] = cat

    # Normalize bias within each feature across all conditions
    bias_norm = bias_df.copy()
    bias_norm["bias_normalized"] = 0.0
    for feature in ALL_FEATURES:
        mask = bias_norm["feature"] == feature
        vals = bias_norm.loc[mask, "bias_value"].values
        if len(vals) == 0:
            continue
        vmin, vmax_f = vals.min(), vals.max()
        if vmax_f > vmin:
            bias_norm.loc[mask, "bias_normalized"] = (vals - vmin) / (vmax_f - vmin)
        else:
            bias_norm.loc[mask, "bias_normalized"] = 0.0

    bias_norm["category"] = bias_norm["feature"].map(feature_to_category)

    def _make_cat_annot(pivot, pivot_sig):
        """Build annotation array from normalized pivot + significance pivot."""
        annot = np.empty(pivot.shape, dtype=object)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                sig = pivot_sig.iloc[i, j] if not pd.isna(pivot_sig.iloc[i, j]) else 0.0
                if pd.isna(val):
                    annot[i, j] = ""
                elif sig > 0.75:
                    annot[i, j] = "{:.3f}***".format(val)
                elif sig > 0.60:
                    annot[i, j] = "{:.3f}**".format(val)
                elif sig > 0.50:
                    annot[i, j] = "{:.3f}*".format(val)
                else:
                    annot[i, j] = "{:.3f}".format(val)
        return annot

    def _plot_cat_heatmap(pivot, pivot_sig, title, output_path):
        """Plot a category-level heatmap with significance annotations."""
        if pivot.empty:
            return
        annot = _make_cat_annot(pivot, pivot_sig)
        figw = max(6, pivot.shape[1] * 1.5)
        figh = max(4, pivot.shape[0] * 0.7)
        fig, ax = plt.subplots(figsize=(figw, figh))
        sns.heatmap(
            pivot, annot=annot, fmt="", cmap="Reds",
            vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Normalized Bias"},
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Feature Category")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    # --- fully_aggregated_by_category ---
    agg_cat_full = bias_norm.groupby("category").agg(
        bias_normalized=("bias_normalized", "mean"),
        significant=("significant", "mean"),
    ).reset_index()
    pivot_full_cat = agg_cat_full.set_index("category")[["bias_normalized"]].rename(
        columns={"bias_normalized": "All"}
    )
    sig_full_cat = agg_cat_full.set_index("category")[["significant"]].rename(
        columns={"significant": "All"}
    )
    _plot_cat_heatmap(
        pivot_full_cat, sig_full_cat,
        "Bias by Feature Category — Fully Aggregated\n"
        "(* p<0.05 >50%, ** >60%, *** >75%)",
        heatmap_dir / "fully_aggregated_by_category.png",
    )

    # --- aggregated_by_model_by_category ---
    agg_cat_model = bias_norm.groupby(["category", "model"]).agg(
        bias_normalized=("bias_normalized", "mean"),
        significant=("significant", "mean"),
    ).reset_index()
    pivot_cat_model = agg_cat_model.pivot_table(
        values="bias_normalized", index="category", columns="model", aggfunc="mean"
    )
    sig_cat_model = agg_cat_model.pivot_table(
        values="significant", index="category", columns="model", aggfunc="mean"
    )
    if not pivot_cat_model.empty:
        pivot_cat_model.columns = [MODEL_LABELS.get(c, c) for c in pivot_cat_model.columns]
        sig_cat_model.columns = pivot_cat_model.columns
    _plot_cat_heatmap(
        pivot_cat_model, sig_cat_model,
        "Bias by Feature Category × Model\n"
        "(Aggregated across Prompts; * p<0.05 >50%, ** >60%, *** >75%)",
        heatmap_dir / "aggregated_by_model_by_category.png",
    )

    # --- aggregated_by_prompt_by_category ---
    agg_cat_prompt = bias_norm.groupby(["category", "prompt_style"]).agg(
        bias_normalized=("bias_normalized", "mean"),
        significant=("significant", "mean"),
    ).reset_index()
    pivot_cat_prompt = agg_cat_prompt.pivot_table(
        values="bias_normalized", index="category", columns="prompt_style", aggfunc="mean"
    )
    sig_cat_prompt = agg_cat_prompt.pivot_table(
        values="significant", index="category", columns="prompt_style", aggfunc="mean"
    )
    _plot_cat_heatmap(
        pivot_cat_prompt, sig_cat_prompt,
        "Bias by Feature Category × Prompt Style\n"
        "(Aggregated across Models; * p<0.05 >50%, ** >60%, *** >75%)",
        heatmap_dir / "aggregated_by_prompt_by_category.png",
    )

    # --- disaggregated_prompt_{style}_by_category ---
    for style in PROMPT_STYLES:
        sub_norm = bias_norm[bias_norm["prompt_style"] == style]
        if sub_norm.empty:
            continue
        agg_sub_cat = sub_norm.groupby(["category", "model"]).agg(
            bias_normalized=("bias_normalized", "mean"),
            significant=("significant", "mean"),
        ).reset_index()
        piv = agg_sub_cat.pivot_table(
            values="bias_normalized", index="category", columns="model", aggfunc="mean"
        )
        sig_piv = agg_sub_cat.pivot_table(
            values="significant", index="category", columns="model", aggfunc="mean"
        )
        if not piv.empty:
            piv.columns = [MODEL_LABELS.get(c, c) for c in piv.columns]
            sig_piv.columns = piv.columns
        _plot_cat_heatmap(
            piv, sig_piv,
            "Bias by Feature Category — Prompt: {}\n".format(style.capitalize()) +
            "(* p<0.05 >50%, ** >60%, *** >75%)",
            heatmap_dir / "disaggregated_prompt_{}_by_category.png".format(style),
        )

    # =========================================================================
    # SIGNIFICANCE RATE PLOTS
    # =========================================================================

    # --- significance_by_model ---
    models = sorted(bias_df["model"].unique())
    sig_rates_model = []
    for slug in models:
        sub = bias_df[bias_df["model"] == slug]
        rate = sub["significant"].mean() * 100.0
        sig_rates_model.append((MODEL_LABELS.get(slug, slug), rate))
    sig_rates_model.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(8, max(3, len(sig_rates_model) * 0.7)))
    labels_m = [r[0] for r in sig_rates_model]
    values_m = [r[1] for r in sig_rates_model]
    ax.barh(labels_m, values_m, color="#4A90E2", alpha=0.85, edgecolor="white")
    ax.set_xlabel("% Conditions Significant (p < 0.05)", fontsize=9)
    ax.set_title(
        "Significance Rate by Model\n(% of prompt_style × feature conditions)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.35)
    for i, v in enumerate(values_m):
        ax.text(v + 0.5, i, "{:.1f}%".format(v), va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(heatmap_dir / "significance_by_model.png", bbox_inches="tight")
    plt.close()

    # --- significance_by_prompt ---
    prompts = PROMPT_STYLES
    sig_rates_prompt = []
    for style in prompts:
        sub = bias_df[bias_df["prompt_style"] == style]
        rate = sub["significant"].mean() * 100.0
        sig_rates_prompt.append((style, rate))
    sig_rates_prompt.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(8, max(3, len(sig_rates_prompt) * 0.7)))
    labels_p = [r[0] for r in sig_rates_prompt]
    values_p = [r[1] for r in sig_rates_prompt]
    ax.barh(labels_p, values_p, color="#FF6B35", alpha=0.85, edgecolor="white")
    ax.set_xlabel("% Conditions Significant (p < 0.05)", fontsize=9)
    ax.set_title(
        "Significance Rate by Prompt Style\n(% of model × feature conditions)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.35)
    for i, v in enumerate(values_p):
        ax.text(v + 0.5, i, "{:.1f}%".format(v), va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(heatmap_dir / "significance_by_prompt.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # FULLY AGGREGATED BAR PLOT (mean bias_value per feature, coloured by category)
    # =========================================================================

    CATEGORY_COLORS = {
        "author":       "#4C72B0",
        "text_metrics": "#55A868",
        "sentiment":    "#C44E52",
        "style":        "#8172B2",
        "content":      "#CCB974",
        "toxicity":     "#64B5CD",
    }

    agg_feat = (
        bias_df.groupby("feature")["bias_value"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    agg_feat = agg_feat.sort_values("mean", ascending=True)
    agg_feat["category"] = agg_feat["feature"].map(feature_to_category)

    colors_bar = [CATEGORY_COLORS.get(c, "#999999") for c in agg_feat["category"]]

    fig, ax = plt.subplots(figsize=(9, max(6, len(agg_feat) * 0.45)))
    y_pos = np.arange(len(agg_feat))
    ax.barh(
        y_pos,
        agg_feat["mean"].values,
        xerr=agg_feat["std"].fillna(0).values,
        color=colors_bar,
        alpha=0.85,
        edgecolor="white",
        error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "black"},
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg_feat["feature"].values, fontsize=8)
    ax.set_xlabel("Mean Bias (Cramér's V / Cohen's d)", fontsize=9)
    ax.set_title(
        "Mean Bias per Feature — Fully Aggregated\n(error bars = std across conditions)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.35)

    # Legend for categories
    legend_handles = []
    for cat, col in CATEGORY_COLORS.items():
        if cat in agg_feat["category"].values:
            legend_handles.append(mpatches.Patch(color=col, label=cat.replace("_", " ").title()))
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(heatmap_dir / "fully_aggregated_bar_plot.png", bbox_inches="tight")
    plt.close()

    print(f"  Saved bias heatmaps → {heatmap_dir}")


# =============================================================================
# 3. DIRECTIONAL BIAS
# =============================================================================

def compute_directional_bias(experiment_dirs: dict[str, Path]) -> pd.DataFrame:
    """
    Compute directional bias (proportion_rec - proportion_pool) for
    categorical/binary features and (mean_rec - mean_pool) for numerical.
    """
    records = []

    for slug, exp_dir in experiment_dirs.items():
        df = load_experiment_data(exp_dir)
        if df is None:
            continue

        for style in PROMPT_STYLES:
            style_df = df[df["prompt_style"] == style]
            if len(style_df) == 0:
                continue
            pool_df = style_df[style_df["selected"] == 0]
            rec_df  = style_df[style_df["selected"] == 1]

            for feature in ALL_FEATURES:
                if feature not in df.columns:
                    continue
                ftype = FEATURE_TYPES.get(feature, "numerical")

                if ftype in ("categorical", "binary"):
                    pool_dist = pool_df[feature].value_counts(normalize=True)
                    rec_dist  = rec_df[feature].value_counts(normalize=True)
                    all_vals  = set(pool_dist.index) | set(rec_dist.index)
                    for val in all_vals:
                        diff = rec_dist.get(val, 0) - pool_dist.get(val, 0)
                        records.append({
                            "model": slug, "prompt_style": style,
                            "feature": feature, "category": str(val),
                            "direction": diff,
                            "pool_proportion": pool_dist.get(val, 0),
                            "rec_proportion":  rec_dist.get(val, 0),
                        })
                else:
                    diff = rec_df[feature].mean() - pool_df[feature].mean()
                    records.append({
                        "model": slug, "prompt_style": style,
                        "feature": feature, "category": feature,
                        "direction": diff,
                        "pool_proportion": pool_df[feature].mean(),
                        "rec_proportion":  rec_df[feature].mean(),
                    })

    return pd.DataFrame(records)


def plot_directional_bars(
    dir_df: pd.DataFrame,
    feature: str,
    groupby: str,
    title: str,
    output_path: Path,
):
    """Bar chart showing directional bias by category value, coloured by group."""
    ftype = FEATURE_TYPES.get(feature, "numerical")
    sub = dir_df[dir_df["feature"] == feature]
    if sub.empty:
        return

    groups = sorted(sub[groupby].unique())
    categories = sorted(sub["category"].unique())
    n_groups = len(groups)
    n_cats = len(categories)

    fig, ax = plt.subplots(figsize=(max(8, n_cats * 1.5), 5))
    bar_width = 0.8 / n_groups
    x = np.arange(n_cats)

    palette = plt.cm.get_cmap("tab10", n_groups)

    for gi, group in enumerate(groups):
        grp_sub = sub[sub[groupby] == group]
        values = []
        for cat in categories:
            cat_row = grp_sub[grp_sub["category"] == cat]
            values.append(cat_row["direction"].mean() if len(cat_row) > 0 else 0.0)

        offset = (gi - n_groups / 2 + 0.5) * bar_width
        if groupby == "model":
            color = MODEL_COLORS.get(group, palette(gi))
            label = MODEL_LABELS.get(group, group)
        else:
            color = palette(gi)
            label = group

        ax.bar(x + offset, values, width=bar_width, label=label, color=color, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Rec proportion − Pool proportion" if ftype in ("categorical", "binary")
                  else "Mean(rec) − Mean(pool)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_directional_bias_plots(dir_df: pd.DataFrame, output_dir: Path):
    """Generate directional bias plots grouped by model and by prompt."""
    dir_bias_dir = output_dir / "3_directional_bias"
    dir_bias_dir.mkdir(parents=True, exist_ok=True)

    if dir_df.empty:
        print("  No directional bias data available.")
        return

    for feature in ALL_FEATURES:
        # By model (aggregated over all prompts)
        agg = dir_df.groupby(["model", "feature", "category"])["direction"].mean().reset_index()
        plot_directional_bars(
            agg, feature, "model",
            f"{feature} — Directional Bias by Model",
            dir_bias_dir / f"{feature}_by_model.png",
        )
        # By prompt (aggregated over all models)
        agg2 = dir_df.groupby(["prompt_style", "feature", "category"])["direction"].mean().reset_index()
        plot_directional_bars(
            agg2, feature, "prompt_style",
            f"{feature} — Directional Bias by Prompt",
            dir_bias_dir / f"{feature}_by_prompt.png",
        )
        # Fully aggregated
        agg3 = dir_df.groupby(["feature", "category"])["direction"].mean().reset_index()
        agg3["_all"] = "All models & prompts"
        plot_directional_bars(
            agg3, feature, "_all",
            f"{feature} — Directional Bias (Fully Aggregated)",
            dir_bias_dir / f"{feature}_fully_aggregated.png",
        )

    print(f"  Saved directional bias plots → {dir_bias_dir}")


# =============================================================================
# 4. FEATURE IMPORTANCE (Random Forest + SHAP)
# =============================================================================

FEATURE_IMPORTANCE_CACHE = None


def prepare_ml_features(df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Encode features for Random Forest."""
    X_parts = []
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            continue
        ftype = FEATURE_TYPES.get(feature, "numerical")
        if ftype == "numerical":
            col = pd.to_numeric(df[feature], errors="coerce").fillna(0)
            X_parts.append(col.rename(feature))
        else:
            dummies = pd.get_dummies(
                df[feature].astype(str).str.lower(), prefix=feature
            )
            X_parts.append(dummies)

    if not X_parts:
        return None

    X = pd.concat(X_parts, axis=1)
    y = df["selected"].astype(int)
    return X, y


def compute_feature_importance(
    experiment_dirs: dict[str, Path],
    cache_path: Path,
) -> pd.DataFrame:
    """Train RF + compute SHAP for each model×prompt condition. Returns importance DF."""
    if cache_path.exists():
        print(f"  Loading cached feature importance from {cache_path}")
        return pd.read_csv(cache_path)

    records = []
    for slug, exp_dir in experiment_dirs.items():
        df = load_experiment_data(exp_dir)
        if df is None:
            continue

        for style in PROMPT_STYLES:
            style_df = df[df["prompt_style"] == style].copy()
            if len(style_df) < 50:
                continue

            result = prepare_ml_features(style_df)
            if result is None:
                continue
            X, y = result

            if y.sum() < 5 or (len(y) - y.sum()) < 5:
                continue

            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_scaled, y)

                auroc = roc_auc_score(y, rf.predict_proba(X_scaled)[:, 1])

                explainer = shap.TreeExplainer(rf)
                shap_vals  = explainer.shap_values(X_scaled)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                mean_shap = np.abs(shap_vals).mean(axis=0)

                # Map SHAP values back to original features
                feature_shap: dict[str, float] = {}
                for col_i, col_name in enumerate(X.columns):
                    # find original feature name from encoded column
                    for feat in ALL_FEATURES:
                        if col_name == feat or col_name.startswith(feat + "_"):
                            feature_shap[feat] = feature_shap.get(feat, 0) + float(np.mean(mean_shap[col_i]))
                            break

                for feat, importance in feature_shap.items():
                    records.append({
                        "model":        slug,
                        "prompt_style": style,
                        "feature":      feat,
                        "shap_importance": importance,
                        "auroc":        auroc,
                    })

                print(f"    {MODEL_LABELS.get(slug, slug)} / {style}: AUROC={auroc:.3f}")

            except Exception as e:
                print(f"    Warning: RF failed for {slug}/{style}: {e}")

    importance_df = pd.DataFrame(records)
    if not importance_df.empty:
        importance_df.to_csv(cache_path, index=False)
        print(f"  Cached feature importance to {cache_path}")
    return importance_df


def plot_importance_heatmap(
    importance_df: pd.DataFrame,
    groupby: str,
    title: str,
    output_path: Path,
):
    """Plot a feature importance heatmap."""
    if importance_df.empty:
        return

    pivot = (
        importance_df.groupby([groupby, "feature"])["shap_importance"]
        .mean()
        .unstack(fill_value=0)
    )
    pivot = pivot.reindex(columns=ALL_FEATURES, fill_value=0)

    if groupby == "model":
        pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.9), max(4, pivot.shape[0] * 0.6)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=SEQ_CMAP,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean |SHAP|"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_feature_importance_plots(importance_df: pd.DataFrame, output_dir: Path):
    """Generate SHAP importance heatmaps."""
    imp_dir = output_dir / "4_feature_importance"
    imp_dir.mkdir(parents=True, exist_ok=True)

    if importance_df.empty:
        print("  No feature importance data.")
        return

    # Fully aggregated
    agg = importance_df.copy()
    agg["_all"] = "All models & prompts"
    plot_importance_heatmap(agg, "_all", "Feature Importance — Fully Aggregated",
                            imp_dir / "fully_aggregated.png")

    # By model
    plot_importance_heatmap(importance_df, "model",
                            "Feature Importance — By Model",
                            imp_dir / "aggregated_by_model.png")

    # By prompt
    plot_importance_heatmap(importance_df, "prompt_style",
                            "Feature Importance — By Prompt Style",
                            imp_dir / "aggregated_by_prompt.png")

    # --- Disaggregated by prompt style: features × models per prompt ---
    # Normalize SHAP within each feature across all conditions first
    imp_norm = importance_df.copy()
    imp_norm["shap_normalized"] = 0.0
    for feature in ALL_FEATURES:
        mask = imp_norm["feature"] == feature
        vals = imp_norm.loc[mask, "shap_importance"].values
        if len(vals) == 0:
            continue
        vmin_i, vmax_i = float(vals.min()), float(vals.max())
        if vmax_i > vmin_i:
            imp_norm.loc[mask, "shap_normalized"] = (vals - vmin_i) / (vmax_i - vmin_i)
        else:
            imp_norm.loc[mask, "shap_normalized"] = 0.0

    for style in PROMPT_STYLES:
        sub_imp = imp_norm[imp_norm["prompt_style"] == style]
        if sub_imp.empty:
            continue

        pivot_norm = sub_imp.groupby(["feature", "model"])["shap_normalized"].mean().unstack(fill_value=0)
        pivot_raw  = sub_imp.groupby(["feature", "model"])["shap_importance"].mean().unstack(fill_value=0)

        pivot_norm = pivot_norm.reindex(index=ALL_FEATURES, fill_value=0)
        pivot_raw  = pivot_raw.reindex(index=ALL_FEATURES, fill_value=0)

        if not pivot_norm.empty:
            pivot_norm.columns = [MODEL_LABELS.get(c, c) for c in pivot_norm.columns]
            pivot_raw.columns  = pivot_norm.columns

        # Build annotation array with raw SHAP values
        annot_imp = np.empty(pivot_norm.shape, dtype=object)
        for i in range(pivot_norm.shape[0]):
            for j in range(pivot_norm.shape[1]):
                val = pivot_raw.iloc[i, j]
                annot_imp[i, j] = "{:.3f}".format(val) if not pd.isna(val) else ""

        figw = max(6, pivot_norm.shape[1] * 2.0)
        figh = max(6, pivot_norm.shape[0] * 0.5)
        fig, ax = plt.subplots(figsize=(figw, figh))
        sns.heatmap(
            pivot_norm, annot=annot_imp, fmt="", cmap=SEQ_CMAP,
            vmin=0, vmax=1, ax=ax,
            cbar_kws={"label": "Normalized SHAP Importance"},
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(
            "Feature Importance (SHAP) — Prompt: {}\n".format(style.capitalize()) +
            "(Normalized within feature; annotations show raw SHAP values)",
            fontsize=11, fontweight="bold", pad=10,
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Feature")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(imp_dir / "disaggregated_prompt_{}.png".format(style), bbox_inches="tight")
        plt.close()

    # Mean AUROC
    auroc_mean = importance_df.groupby("model")["auroc"].mean()
    print(f"  Mean AUROC by model:")
    for slug, auroc in auroc_mean.items():
        print(f"    {MODEL_LABELS.get(slug, slug)}: {auroc:.3f}")

    print(f"  Saved feature importance plots → {imp_dir}")


# =============================================================================
# IMPORTANCE VS BIAS COMPARISON
# =============================================================================

def plot_importance_vs_bias(
    bias_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    output_dir: Path,
):
    """
    Scatter plot: x = mean bias_value per feature (across all conditions),
    y = mean shap_importance per feature (across all conditions).
    Each point is a feature, labelled, coloured by feature category.
    """
    if bias_df.empty or importance_df.empty:
        print("  Skipping importance vs bias plot (insufficient data).")
        return

    CATEGORY_COLORS = {
        "author":       "#4C72B0",
        "text_metrics": "#55A868",
        "sentiment":    "#C44E52",
        "style":        "#8172B2",
        "content":      "#CCB974",
        "toxicity":     "#64B5CD",
    }

    feature_to_category = {}
    for cat, feats in FEATURES.items():
        for f in feats:
            feature_to_category[f] = cat

    mean_bias = bias_df.groupby("feature")["bias_value"].mean().reset_index()
    mean_bias.columns = ["feature", "mean_bias"]

    mean_imp = importance_df.groupby("feature")["shap_importance"].mean().reset_index()
    mean_imp.columns = ["feature", "mean_shap"]

    merged = pd.merge(mean_bias, mean_imp, on="feature", how="inner")
    if merged.empty:
        print("  Skipping importance vs bias plot (no matching features).")
        return

    merged["category"] = merged["feature"].map(feature_to_category).fillna("other")
    merged["color"] = merged["category"].map(
        lambda c: CATEGORY_COLORS.get(c, "#999999")
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    for cat, grp in merged.groupby("category"):
        color = CATEGORY_COLORS.get(cat, "#999999")
        ax.scatter(
            grp["mean_bias"], grp["mean_shap"],
            color=color, s=80, alpha=0.85, edgecolors="white", linewidths=0.5,
            label=cat.replace("_", " ").title(), zorder=3,
        )

    # Label each point
    for _, row in merged.iterrows():
        ax.annotate(
            row["feature"].replace("_", " "),
            xy=(row["mean_bias"], row["mean_shap"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7,
            color="#333333",
        )

    ax.set_xlabel("Mean Bias (Cramér's V / Cohen's d)", fontsize=10)
    ax.set_ylabel("Mean SHAP Importance", fontsize=10)
    ax.set_title(
        "Feature Importance vs Bias Magnitude\n(each point = one feature, aggregated across all conditions)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(title="Category", fontsize=8, title_fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "importance_vs_bias_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved importance vs bias comparison → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive bias analysis on YouGov linked experiments"
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "experiments",
        help="Directory containing per-model experiment subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "analysis_outputs",
        help="Output directory for visualizations and CSVs",
    )
    args = parser.parse_args()

    viz_dir = args.output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("YOUGOV LINKED COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print(f"Experiments dir: {args.experiments_dir}")
    print(f"Output dir:      {args.output_dir}")
    print()

    # ------------------------------------------------------------------
    # Discover experiment directories
    # ------------------------------------------------------------------
    experiment_dirs = get_experiment_dirs(args.experiments_dir)
    if not experiment_dirs:
        print(f"ERROR: No completed experiments found in {args.experiments_dir}")
        print("Run run_experiment.py (or run_all_experiments.py) first.")
        return

    print(f"Found {len(experiment_dirs)} completed experiments:")
    for slug in experiment_dirs:
        print(f"  {MODEL_LABELS.get(slug, slug)}")
    print()

    # Load pool data for distributions
    pool_data = {}
    for slug, exp_dir in experiment_dirs.items():
        pool = load_pool_data(exp_dir)
        if pool is not None:
            pool_data[slug] = pool

    # ------------------------------------------------------------------
    # 1. Feature distributions
    # ------------------------------------------------------------------
    print("1. Generating feature distributions...")
    generate_feature_distributions(pool_data, viz_dir)
    print()

    # ------------------------------------------------------------------
    # 2. Bias heatmaps
    # ------------------------------------------------------------------
    print("2. Computing bias metrics (Cramér's V / Cohen's d)...")
    bias_df = compute_all_bias(experiment_dirs)
    if not bias_df.empty:
        bias_csv = args.output_dir / "pool_vs_recommended_summary.csv"
        bias_df.to_csv(bias_csv, index=False)
        print(f"  Saved bias summary → {bias_csv}")
        generate_bias_heatmaps(bias_df, viz_dir)
    print()

    # ------------------------------------------------------------------
    # 3. Directional bias
    # ------------------------------------------------------------------
    print("3. Computing directional bias...")
    dir_df = compute_directional_bias(experiment_dirs)
    if not dir_df.empty:
        dir_csv = args.output_dir / "directional_bias_data.csv"
        dir_df.to_csv(dir_csv, index=False)
        print(f"  Saved directional bias data → {dir_csv}")
        generate_directional_bias_plots(dir_df, viz_dir)
    print()

    # ------------------------------------------------------------------
    # 4. Feature importance (RF + SHAP)
    # ------------------------------------------------------------------
    print("4. Computing feature importance (Random Forest + SHAP)...")
    print("  (This may take 15–30 minutes on first run; results are cached.)")
    cache_path = args.output_dir / "feature_importance_data.csv"
    importance_df = compute_feature_importance(experiment_dirs, cache_path)
    if not importance_df.empty:
        generate_feature_importance_plots(importance_df, viz_dir)
    print()

    # ------------------------------------------------------------------
    # 5. Importance vs Bias comparison scatter
    # ------------------------------------------------------------------
    print("5. Generating importance vs bias comparison...")
    if not bias_df.empty and not importance_df.empty:
        plot_importance_vs_bias(bias_df, importance_df, viz_dir)
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Visualizations: {viz_dir}")
    if not bias_df.empty:
        n_sig = (bias_df["p_value"] < 0.05).sum()
        total = len(bias_df)
        print(f"  Significant conditions: {n_sig}/{total} ({n_sig/total*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
