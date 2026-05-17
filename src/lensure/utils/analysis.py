"""
Post-execution analysis: per-attack summary and ROC curve.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

_EXCLUDED = {"change"}
_SEMANTIC_PREFIX = "semantic-"


def _classify_attacks(attacks: list[str]) -> tuple[set[str], set[str]]:
    semantic = {a for a in attacks if a.startswith(_SEMANTIC_PREFIX)}
    benign = {a for a in attacks if a not in semantic and a not in _EXCLUDED}
    return benign, semantic


def analyze_results(csv_path: str, embed_og_hash: bool, output_dir: str) -> None:
    """
    Writes a per-attack summary (matching analyze.py output) to analysis.txt.
    """
    df = pl.read_csv(csv_path)
    attacks = sorted(df["attack"].unique().to_list())
    total_images = df["image"].n_unique()

    lines = []
    lines.append(f"{'─' * 56}")
    lines.append(f"  {csv_path}")
    lines.append(f"  {total_images} images · {len(attacks)} attacks")
    lines.append(f"{'─' * 56}")

    if embed_og_hash:
        header = f"{'Attack':<32}  {'Sig OK':>7}  {'Sig KO':>7}  {'Mean dist':>9}"
    else:
        header = f"{'Attack':<32}  {'Sig OK':>7}  {'Sig KO':>7}"
    lines.append(header)
    lines.append("─" * len(header))

    for attack in attacks:
        adf = df.filter(pl.col("attack") == attack)
        n = len(adf)
        sig_ok = adf["signature_valid"].sum()
        ok_pct = sig_ok / n
        ko_pct = (n - sig_ok) / n
        if embed_og_hash:
            mean_dist = adf["distance"].mean()
            lines.append(
                f"{attack:<32}  {ok_pct:>6.1%}  {ko_pct:>7.1%}  {mean_dist:>9.2f}"
            )
        else:
            lines.append(f"{attack:<32}  {ok_pct:>6.1%}  {ko_pct:>7.1%}")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(output_dir, "analysis.txt"), "w") as f:
        f.write(text + "\n")


def compute_roc(csv_path: str, output_dir: str) -> None:
    """
    Computes ROC curve (semantic FAR vs benign acceptance) and saves
    roc_curve.png and roc_metrics.txt to output_dir.

    Attack groups are inferred automatically:
      - semantic-*          → semantic (manipulations to detect)
      - everything else     → benign   (except 'change', excluded)
    Only rows where signature_valid=True are used.
    """
    df = pl.read_csv(csv_path)
    attacks = df["attack"].unique().to_list()
    benign_set, semantic_set = _classify_attacks(attacks)

    if not semantic_set:
        print("ROC: no semantic attacks found, skipping.")
        return

    benign_dists = (
        df.filter(pl.col("attack").is_in(list(benign_set)) & pl.col("signature_valid"))
        ["distance"].to_list()
    )
    semantic_dists = (
        df.filter(pl.col("attack").is_in(list(semantic_set)) & pl.col("signature_valid"))
        ["distance"].to_list()
    )

    if not benign_dists or not semantic_dists:
        print("ROC: not enough data after filtering, skipping.")
        return

    max_dist = int(max(max(benign_dists), max(semantic_dists))) + 1

    frr_list, far_list = [], []
    for thresh in range(0, max_dist + 1):
        frr = 1 - sum(1 for d in benign_dists if d < thresh) / len(benign_dists)
        far = sum(1 for d in semantic_dists if d < thresh) / len(semantic_dists)
        frr_list.append(frr)
        far_list.append(far)

    tpr = [1 - f for f in frr_list]
    auc = sum(
        0.5 * (tpr[i] + tpr[i - 1]) * abs(far_list[i] - far_list[i - 1])
        for i in range(1, len(far_list))
    )
    best_thresh = next(
        (t for t, (f, fa) in enumerate(zip(frr_list, far_list)) if f == 0.0 and fa == 0.0),
        None,
    )

    metrics_lines = [
        f"AUC:              {auc:.4f}",
        f"Best threshold:   {best_thresh if best_thresh is not None else 'N/A'}",
        f"Benign attacks:   {sorted(benign_set)}",
        f"Semantic attacks: {sorted(semantic_set)}",
        f"Benign samples:   {len(benign_dists)}",
        f"Semantic samples: {len(semantic_dists)}",
    ]
    metrics_text = "\n".join(metrics_lines)
    print("\n" + metrics_text)
    with open(os.path.join(output_dir, "roc_metrics.txt"), "w") as f:
        f.write(metrics_text + "\n")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([f * 100 for f in far_list], [t * 100 for t in tpr], lw=2)
    ax.set_xlabel("FAR (False Acceptance Rate) %")
    ax.set_ylabel("1 − FRR (Benign Acceptance Rate) %")
    ax.set_title(f"ROC Curve — AUC = {auc:.4f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"ROC curve saved to {output_dir}/roc_curve.png")
