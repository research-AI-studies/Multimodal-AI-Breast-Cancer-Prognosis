"""
SHAP feature importance analysis (Table 5.2) and hierarchical clustering (Section 5.1).
"""
import numpy as np
import pandas as pd
import shap
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def compute_shap_values(model, X: np.ndarray, feature_names: list) -> dict:
    """Compute SHAP values for an XGBoost model.
    Returns dict with shap_values array and importance DataFrame.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total = mean_abs_shap.sum()
    importance_pct = mean_abs_shap / total if total > 0 else mean_abs_shap

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "importance_pct": importance_pct,
    }).sort_values("importance_pct", ascending=False).reset_index(drop=True)

    print(f"[shap] Top 10 features by SHAP importance:")
    for _, row in df_imp.head(10).iterrows():
        print(f"  {row['feature']:35s} {row['importance_pct']*100:.1f}%")

    return {
        "shap_values": shap_values,
        "importance": df_imp,
        "explainer": explainer,
    }


def hierarchical_cluster_shap(shap_values: np.ndarray, feature_names: list,
                               n_clusters: int = 5) -> pd.DataFrame:
    """Hierarchical clustering of SHAP interaction patterns."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    # Cluster features by their SHAP value correlation
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    corr = shap_df.corr().values
    dist = pdist(corr, metric="correlation")
    Z = linkage(dist, method="ward")
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    cluster_df = pd.DataFrame({
        "feature": feature_names,
        "cluster": clusters,
    }).sort_values("cluster")
    return cluster_df
