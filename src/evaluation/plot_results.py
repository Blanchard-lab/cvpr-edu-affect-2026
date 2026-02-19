import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Mapping from short filename tokens → human-readable axis labels
# ---------------------------------------------------------------------------
AXIS_LABELS = {
    "gt":        "Epistemic Emotions",
    "openface":  "OpenFace Basic Emotions",
    "libreface": "LibreFace Basic Emotions",
}


def _infer_axis_label(key: str) -> str:
    key_lower = key.lower()
    for token, label in AXIS_LABELS.items():
        if token in key_lower:
            return label
    return key  # fallback


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_matrix_csv(path: Path):
    with path.open("r", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    col_labels = rows[0][1:]
    row_labels = []
    data = []
    for line in rows[1:]:
        row_labels.append(line[0])
        data.append([float(x) for x in line[1:]])
    return row_labels, col_labels, np.array(data, dtype=np.float64)


def _cell_text_color(value, vmin, vmax, cmap):
    """Return 'black' or 'white' based on background luminance."""
    norm = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    rgba = cmap(norm)
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if lum > 0.45 else "white"


# ---------------------------------------------------------------------------
# Core plotting
# ---------------------------------------------------------------------------
def save_heatmap(
    fig_path: Path,
    row_labels,
    col_labels,
    M,
    title: str,
    save_pdf: bool,
    x_axis_label: str = "Predicted",
    y_axis_label: str = "Ground Truth",
):
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    is_counts = np.all(np.mod(M, 1) == 0)
    cmap = plt.get_cmap("YlOrRd")
    vmin, vmax = M.min(), M.max()

    fig_w = max(8, 0.6 * len(col_labels))
    fig_h = max(6, 0.5 * len(row_labels))

    def _draw(fig, ax):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticklabels(row_labels)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Count" if is_counts else "Proportion")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                s = str(int(v)) if is_counts else f"{v:.2f}"
                color = _cell_text_color(v, vmin, vmax, cmap)
                ax.text(j, i, s, ha="center", va="center", fontsize=8, color=color)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw(fig, ax)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    if save_pdf:
        pdf_path = fig_path.with_suffix(".pdf")
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        _draw(fig, ax)
        fig.tight_layout()
        fig.savefig(pdf_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------
def summarize_gt_to_pred(cm_raw_path: Path):
    rlab, clab, M = read_matrix_csv(cm_raw_path)
    row_sums = M.sum(axis=1)
    total = float(M.sum()) if M.size else 0.0
    top = []
    for i, gt in enumerate(rlab):
        if row_sums[i] <= 0:
            top.append((gt, "", 0.0))
            continue
        j = int(np.argmax(M[i]))
        top.append((gt, clab[j], float(M[i, j] / row_sums[i])))
    overall = []
    if total > 0:
        col_sums = M.sum(axis=0)
        order = np.argsort(-col_sums)
        for j in order[:5]:
            overall.append((clab[j], float(col_sums[j] / total)))
    return rlab, clab, M, top, overall


def write_summary_table(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "gt_label", "top_pred_label", "top_pred_share"])
        for comp, gt, pred, share in rows:
            w.writerow([comp, gt, pred, f"{share:.4f}"])


# ---------------------------------------------------------------------------
# Title / filename helpers for flipped variants
# ---------------------------------------------------------------------------
def _flip_title(title: str) -> str:
    parts = title.split(" vs ", 1)
    if len(parts) == 2:
        left, right = parts
        if " — " in right:
            right_label, suffix = right.split(" — ", 1)
            return f"{right_label} vs {left} — {suffix} (Flipped)"
        return f"{right} vs {left} (Flipped)"
    return title + " (Flipped)"


def _flip_stem(stem: str) -> str:
    # e.g. gt_vs_openface_raw  ->  openface_vs_gt_raw
    parts = stem.split("_vs_", 1)
    if len(parts) == 2:
        left, right = parts
        right_parts = right.split("_", 1)
        right_label = right_parts[0]
        suffix = "_" + right_parts[1] if len(right_parts) > 1 else ""
        return f"{right_label}_vs_{left}{suffix}"
    return stem + "_flipped"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--tab_dir", required=True)
    ap.add_argument("--save_pdf", type=int, default=1)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    fig_dir = Path(args.fig_dir)
    tab_dir = Path(args.tab_dir)
    save_pdf = bool(args.save_pdf)

    specs = [
        ("cm_gt_vs_openface_raw.csv",            "GT (Epistemic) vs OpenFace (Basic) — Raw Counts",      "gt_vs_openface_raw.png"),
        ("cm_gt_vs_openface_row_norm.csv",        "GT (Epistemic) vs OpenFace (Basic) — Row-Normalized",  "gt_vs_openface_row_norm.png"),
        ("cm_gt_vs_libreface_raw.csv",            "GT (Epistemic) vs LibreFace (Basic) — Raw Counts",     "gt_vs_libreface_raw.png"),
        ("cm_gt_vs_libreface_row_norm.csv",       "GT (Epistemic) vs LibreFace (Basic) — Row-Normalized", "gt_vs_libreface_row_norm.png"),
        ("cm_openface_vs_libreface_raw.csv",      "OpenFace vs LibreFace — Raw Counts",                   "openface_vs_libreface_raw.png"),
        ("cm_openface_vs_libreface_row_norm.csv", "OpenFace vs LibreFace — Row-Normalized",               "openface_vs_libreface_row_norm.png"),
    ]

    for fname, title, out_png in specs:
        p = eval_dir / fname
        if not p.exists():
            continue
        rlab, clab, M = read_matrix_csv(p)

        # Parse row/col source tokens from filename: cm_<row>_vs_<col>_*.csv
        stem_parts = Path(fname).stem.split("_vs_", 1)  # e.g. ['cm_gt', 'openface_raw']
        row_key = stem_parts[0].replace("cm_", "")       # 'gt'
        col_key = stem_parts[1].split("_")[0] if len(stem_parts) > 1 else ""  # 'openface'
        y_label = _infer_axis_label(row_key)   # e.g. "Epistemic Emotions"
        x_label = _infer_axis_label(col_key)   # e.g. "OpenFace Basic Emotions"

        # Original orientation
        save_heatmap(
            fig_dir / out_png, rlab, clab, M, title, save_pdf,
            x_axis_label=x_label,
            y_axis_label=y_label,
        )

        # Flipped orientation (transpose) — axes swapped
        flipped_title = _flip_title(title)
        flipped_stem = _flip_stem(Path(out_png).stem)
        flipped_png = Path(flipped_stem + ".png")
        save_heatmap(
            fig_dir / flipped_png, clab, rlab, M.T, flipped_title, save_pdf,
            x_axis_label=y_label,
            y_axis_label=x_label,
        )

    summary_rows = []
    for comp_name, raw_csv in [
        ("gt_vs_openface", eval_dir / "cm_gt_vs_openface_raw.csv"),
        ("gt_vs_libreface", eval_dir / "cm_gt_vs_libreface_raw.csv"),
    ]:
        if not raw_csv.exists():
            continue
        _, _, _, top, _ = summarize_gt_to_pred(raw_csv)
        for gt, pred, share in top:
            summary_rows.append((comp_name, gt, pred, share))

    write_summary_table(tab_dir / "top_mapping_summary.csv", summary_rows)

    print(f"Wrote figures to: {fig_dir}")
    print(f"Wrote tables to: {tab_dir}")


if __name__ == "__main__":
    main()