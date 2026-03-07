import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_matrix_csv(path: Path):
    with path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    col_labels = rows[0][1:]
    row_labels = []
    data = []
    for row in rows[1:]:
        row_labels.append(row[0])
        data.append([float(x) for x in row[1:]])
    return row_labels, col_labels, np.array(data, dtype=np.float64)


def plot_heatmap(mat, row_labels, col_labels, title, out_path: Path, save_pdf: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig_w = max(7, 0.9 * len(col_labels))
    fig_h = max(5, 0.7 * len(row_labels))
    vmin, vmax = mat.min(), mat.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", norm=norm)
    

    ax.set_title(title)
    ax.set_xlabel("LibreFace")
    ax.set_ylabel("OpenFace 3.0")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticklabels(row_labels)

    is_float = np.any(np.mod(mat, 1) != 0)
    cmap = plt.get_cmap("YlGnBu")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = f"{v:.2f}" if is_float else str(int(v))
            rgba = cmap(norm(v))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "black" if luminance > 0.4 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion" if is_float else "Count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    if save_pdf:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mat, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("LibreFace")
        ax.set_ylabel("OpenFace 3.0")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=25, ha="right")
        ax.set_yticklabels(row_labels)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                txt = f"{v:.2f}" if is_float else str(int(v))
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Proportion" if is_float else "Count")

        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--save_pdf", type=int, default=1)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    fig_dir = Path(args.fig_dir)
    save_pdf = bool(args.save_pdf)

    raw_csv = eval_dir / "cm_openface_vs_libreface_affectnet_raw.csv"
    norm_csv = eval_dir / "cm_openface_vs_libreface_affectnet_norm.csv"

    if raw_csv.exists():
        rows, cols, mat = read_matrix_csv(raw_csv)
        plot_heatmap(
            mat,
            rows,
            cols,
            "OpenFace 3.0 vs LibreFace on AffectNet (Raw Counts)",
            fig_dir / "openface_vs_libreface_affectnet_raw.png",
            save_pdf,
        )

    if norm_csv.exists():
        rows, cols, mat = read_matrix_csv(norm_csv)
        plot_heatmap(
            mat,
            rows,
            cols,
            "OpenFace 3.0 vs LibreFace on AffectNet (Row-Normalized)",
            fig_dir / "openface_vs_libreface_affectnet_norm.png",
            save_pdf,
        )

    print(f"Wrote figures to: {fig_dir}")


if __name__ == "__main__":
    main()