import argparse
import csv
from pathlib import Path
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


EPISTEMIC_7 = [
    "Curious",
    "Confused",
    "Disengaged",
    "Surprised",
    "Conflicted",
    "Optimistic",
    "Frustrated",
]

HSE_BASIC_8 = [
    "Neutral",
    "Happy",
    "Sad",
    "Angry",
    "Surprise",
    "Fear",
    "Disgust",
    "Contempt",
]

HSE_MODE_MAP = {
    "Happiness": "Happy",
    "Sadness": "Sad",
    "Anger": "Angry",
    "Neutral": "Neutral",
    "Surprise": "Surprise",
    "Fear": "Fear",
    "Disgust": "Disgust",
    "Contempt": "Contempt",
    "Happy": "Happy",
    "Sad": "Sad",
    "Angry": "Angry",
}


def read_rows(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def normalize_mode(mode: str):
    m = str(mode).strip()
    m = HSE_MODE_MAP.get(m, m)
    return m if m in HSE_BASIC_8 else None


def filter_epistemic_va(rows):
    out = []
    for r in rows:
        lab = str(r.get("labels", "")).strip()
        if lab in EPISTEMIC_7:
            v = to_float(r.get("valence_mean", None))
            a = to_float(r.get("arousal_mean", None))
            if v is None or a is None:
                continue
            out.append((lab, v, a))
    return out


def filter_mode_va(rows):
    out = []
    for r in rows:
        mode = normalize_mode(r.get("emotion_mode", ""))
        if mode is None:
            continue
        v = to_float(r.get("valence_mean", None))
        a = to_float(r.get("arousal_mean", None))
        if v is None or a is None:
            continue
        out.append((mode, v, a))
    return out


def write_summary_table(path: Path, triples, label_order):
    by = defaultdict(list)
    for lab, v, a in triples:
        by[lab].append((v, a))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n", "valence_mean", "valence_std", "arousal_mean", "arousal_std"])
        for lab in label_order:
            vals = by.get(lab, [])
            if not vals:
                w.writerow([lab, 0, "", "", "", ""])
                continue
            V = np.array([x[0] for x in vals], dtype=np.float64)
            A = np.array([x[1] for x in vals], dtype=np.float64)
            w.writerow([
                lab,
                int(len(vals)),
                f"{float(V.mean()):.6f}",
                f"{float(V.std()):.6f}",
                f"{float(A.mean()):.6f}",
                f"{float(A.std()):.6f}",
            ])


def plot_box(values_by_label, labels, title, ylabel, out_path: Path, save_pdf: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [values_by_label.get(lab, []) for lab in labels]

    fig_w = max(9, 0.9 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    if save_pdf:
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


def plot_va_scatter(triples, labels, title, out_path: Path, save_pdf: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by = defaultdict(list)
    for lab, v, a in triples:
        by[lab].append((v, a))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(0.0)
    ax.axvline(0.0)

    for lab in labels:
        pts = by.get(lab, [])
        if not pts:
            continue
        V = [p[0] for p in pts]
        A = [p[1] for p in pts]
        ax.scatter(V, A, label=lab, alpha=0.5, s=14)
        # ax.scatter([float(np.mean(V))], [float(np.mean(A))], marker="x", s=70)

    ax.set_title(title)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    if save_pdf:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.axhline(0.0)
        ax.axvline(0.0)
        for lab in labels:
            pts = by.get(lab, [])
            if not pts:
                continue
            V = [p[0] for p in pts]
            A = [p[1] for p in pts]
            ax.scatter(V, A, label=lab, alpha=0.5, s=14)
            ax.scatter([float(np.mean(V))], [float(np.mean(A))], marker="x", s=70)
        ax.set_title(title)
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


def confusion_gt_vs_mode(rows):
    mat = np.zeros((len(EPISTEMIC_7), len(HSE_BASIC_8)), dtype=np.int64)
    for r in rows:
        gt = str(r.get("labels", "")).strip()
        if gt not in EPISTEMIC_7:
            continue
        mode = normalize_mode(r.get("emotion_mode", ""))
        if mode is None:
            continue
        i = EPISTEMIC_7.index(gt)
        j = HSE_BASIC_8.index(mode)
        mat[i, j] += 1
    return mat


def row_normalize(mat):
    denom = mat.sum(axis=1, keepdims=True).astype(np.float64)
    out = np.zeros_like(mat, dtype=np.float64)
    mask = denom.squeeze() > 0
    out[mask] = mat[mask] / denom[mask]
    return out


def write_matrix_csv(path: Path, mat, row_labels, col_labels, float_fmt=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + col_labels)
        for i, rlab in enumerate(row_labels):
            row = [rlab]
            for j in range(len(col_labels)):
                v = mat[i, j]
                if float_fmt is None:
                    row.append(int(v))
                else:
                    row.append(float_fmt.format(float(v)))
            w.writerow(row)



def plot_heatmap(mat, row_labels, col_labels, title, out_path: Path, save_pdf: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig_w = max(9, 0.9 * len(col_labels))
    fig_h = max(5, 0.6 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Normalize for colormap lookup (used to pick text color)
    vmin, vmax = mat.min(), mat.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", norm=norm)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticklabels(row_labels)

    cmap = plt.get_cmap("YlGnBu")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(int(v))

            # Pick black or white text based on perceived brightness of the cell
            rgba = cmap(norm(v))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "black" if luminance > 0.4 else "white"

            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color=text_color, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    if save_pdf:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mat, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=25, ha="right")
        ax.set_yticklabels(row_labels)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if isinstance(v, (float, np.floating)):
                    txt = f"{v:.2f}"
                else:
                    txt = str(int(v))
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


def run_va_grouping(model_name, triples, label_order, fig_dir: Path, tab_dir: Path, save_pdf: bool, tag: str):
    if not triples:
        print(f"[WARN] No rows for {model_name} ({tag})")
        return

    write_summary_table(tab_dir / f"va_summary_{model_name}_{tag}.csv", triples, label_order)

    vals_by = defaultdict(list)
    aros_by = defaultdict(list)
    for lab, v, a in triples:
        vals_by[lab].append(v)
        aros_by[lab].append(a)

    plot_box(
        vals_by,
        labels=label_order,
        title=f"{model_name.upper()} Valence by {tag.replace('_',' ').title()}",
        ylabel="Valence",
        out_path=fig_dir / f"{model_name}_valence_by_{tag}.png",
        save_pdf=save_pdf,
    )
    plot_box(
        aros_by,
        labels=label_order,
        title=f"{model_name.upper()} Arousal by {tag.replace('_',' ').title()}",
        ylabel="Arousal",
        out_path=fig_dir / f"{model_name}_arousal_by_{tag}.png",
        save_pdf=save_pdf,
    )
    plot_va_scatter(
        triples,
        labels=label_order,
        title=f"{model_name.upper()} Valence--Arousal by {tag.replace('_',' ').title()}",
        out_path=fig_dir / f"{model_name}_va_scatter_by_{tag}.png",
        save_pdf=save_pdf,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cage_csv", required=True)
    ap.add_argument("--hsemotion_csv", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--tab_dir", required=True)
    ap.add_argument("--save_pdf", type=int, default=1)
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    tab_dir = Path(args.tab_dir)
    save_pdf = bool(args.save_pdf)

    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    cage_rows = read_rows(Path(args.cage_csv)) if Path(args.cage_csv).exists() else []
    cage_triples = filter_epistemic_va(cage_rows)
    run_va_grouping("cage", cage_triples, EPISTEMIC_7, fig_dir, tab_dir, save_pdf, tag="epistemic")

    hse_path = Path(args.hsemotion_csv)
    if not hse_path.exists():
        print(f"[WARN] Missing: {hse_path}")
        return

    hse_rows = read_rows(hse_path)

    hse_triples_epi = filter_epistemic_va(hse_rows)
    run_va_grouping("hsemotion", hse_triples_epi, EPISTEMIC_7, fig_dir, tab_dir, save_pdf, tag="epistemic")

    hse_triples_mode = filter_mode_va(hse_rows)
    run_va_grouping("hsemotion", hse_triples_mode, HSE_BASIC_8, fig_dir, tab_dir, save_pdf, tag="mode")

    raw = confusion_gt_vs_mode(hse_rows)
    norm = row_normalize(raw)

    write_matrix_csv(tab_dir / "hsemotion_gt_vs_emotion_mode_raw.csv", raw, EPISTEMIC_7, HSE_BASIC_8)
    write_matrix_csv(tab_dir / "hsemotion_gt_vs_emotion_mode_norm.csv", norm, EPISTEMIC_7, HSE_BASIC_8, float_fmt="{:.6f}")

    plot_heatmap(
        raw,
        EPISTEMIC_7,
        HSE_BASIC_8,
        "HSEmotion: Epistemic (GT) vs Emotion Mode (Raw Counts)",
        fig_dir / "hsemotion_gt_vs_emotion_mode_raw.png",
        save_pdf,
    )
    plot_heatmap(
        norm,
        EPISTEMIC_7,
        HSE_BASIC_8,
        "HSEmotion: Epistemic (GT) vs Emotion Mode (Row-Normalized)",
        fig_dir / "hsemotion_gt_vs_emotion_mode_norm.png",
        save_pdf,
    )

    print("[OK] Plots and tables written.")


if __name__ == "__main__":
    main()
