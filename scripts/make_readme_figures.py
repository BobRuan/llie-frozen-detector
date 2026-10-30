# -*- coding: utf-8 -*-
"""
README 用的三张结果图（不重新检测，读 eval_lis_robustness.py 缓存在 outputs\\lis_test\\dets\\ 的 YOLO11s 结果）：

  1) docs\\figures\\lis_map50.png：LIS 测试集上各输入的 mAP50 柱状图（数字来自 logs\\ 里的 json）。
  2) docs\\figures\\lis_input_output_gt.jpg：输入 | 输出（λ=10）| 真值，每行一个场景，不画框。
  3) docs\\figures\\lis_examples.jpg：若干场景的检测对比，每行一个场景，
     列为 暗图 | 现成 CIDNet | 微调 λ=0 | 微调 λ=10 | 正常光图；绿框是标注，红框是 YOLO11s 检测（置信度 >= 0.25）。

    python scripts\\make_readme_figures.py --list            # 按“λ=10 比 λ=0 多检出几个”列出候选场景
    python scripts\\make_readme_figures.py --triplet_ids 139 359 1519 3461 3369 --ids 379 3787 183 1639 3831 3209
--triplet_ids 是从等间隔取的 30 个测试场景里按场景类型挑的（没有看结果）。
--ids 的前 5 个是 --list 里 λ=10 比 λ=0 多检出最多的场景，最后一个（3209）是 λ=10 检出更少的反例。
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_lis_robustness import match_flags  # noqa: E402
from run_pipeline_lis import CONF_SHOW, LIS_TO_COCO, NAME_FIX, imread_rgb, save_jpg  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "LIS" / "test"
ENH_DIR = PROJECT_ROOT / "outputs" / "lis_test"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
# 完整测试集的两次评测：前一次有现成方法和 λ = 0/1，后一次补上了 λ = 10
PIPELINE_JSONS = [PROJECT_ROOT / "logs" / "pipeline_lis_20261014_132848.json",
                  PROJECT_ROOT / "logs" / "pipeline_lis_20261016_034536.json"]
COLUMNS = [("dark", "Dark input"), ("cidnet_lolv2real", "Off-the-shelf CIDNet"),
           ("ft_image_only", "Fine-tuned, image loss"), ("ft_yolo_feat_w10", "Ours (+ detector loss)"),
           ("normal", "Normal light (reference)")]
CELL_W = 480


def load_dets(group):
    z = np.load(ENH_DIR / "dets" / f"yolo11s__{group}.npz")
    splits = np.cumsum(z["n"])[:-1]
    return list(zip(np.split(z["boxes"], splits), np.split(z["conf"], splits), np.split(z["cls"], splits)))


def true_positives(det, entry):
    """conf >= CONF_SHOW 的检测里，按类别匹配上标注（IoU 0.5）的个数。"""
    b, s, k = det
    n = 0
    for c, coco_id in LIS_TO_COCO.items():
        gt = np.array([[x["x1"], x["y1"], x["x2"], x["y2"]] for x in entry["boxes"] if NAME_FIX[x["category"]] == c],
                      dtype=np.float32).reshape(-1, 4)
        m = (k == coco_id) & (s >= CONF_SHOW)
        n += int(match_flags(b[m], s[m], gt).sum())
    return n


def map50_chart(out):
    """横向柱状图：每种输入的 mAP50，正常光图画成竖直参考线。本方法用蓝色突出，其余为灰色。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = {}
    for f in PIPELINE_JSONS:
        m.update({g: v["mAP50"] for g, v in json.load(open(f, encoding="utf-8"))["detection"].items()})
    rows = [("Dark input (no enhancement)", m["dark"], "base"),
            ("Gamma correction", m["auto_gamma"], "base"),
            ("Off-the-shelf CIDNet (generalization)", m["cidnet_generalization"], "base"),
            ("Off-the-shelf CIDNet (LOL-v2)", m["cidnet_lolv2real"], "base"),
            ("Fine-tuned, image loss only (λ = 0)", m["ft_image_only"], "control"),
            ("Fine-tuned + detector loss (λ = 1)", m["ft_yolo_feat"], "ours"),
            ("Fine-tuned + detector loss (λ = 10)", m["ft_yolo_feat_w10"], "ours")]
    colors = {"base": "#b4b2ab", "control": "#8fb3e0", "ours": "#2a78d6"}
    ink, ink2 = "#0b0b0b", "#52514e"

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.62, color=[colors[r[2]] for r in rows], zorder=2)
    for yi, (label, v, kind) in zip(y, rows):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=10, color=ink,
                fontweight="bold" if kind == "ours" else "normal")
    top = y[0] + 0.5
    ax.vlines(m["dark"], -0.6, top, color=ink2, lw=1, ls=":", zorder=1)
    ax.vlines(m["normal"], -0.6, top, color=ink, lw=1.5, ls="--", zorder=3)
    ax.text(m["normal"], top + 0.08, f"normal-light input {m['normal']:.3f}", fontsize=9, color=ink,
            ha="center", va="bottom")
    ax.text(m["dark"], top + 0.08, "dark input", fontsize=9, color=ink2, ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=ink)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("mAP50 of the frozen YOLO11s on the LIS test set (669 images)", fontsize=10, color=ink2)
    ax.tick_params(axis="x", colors=ink2, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(out)


def examples_grid(ids, entries, dets, out):
    """每行一个场景；格子标题写“检出 k / 共 n 个”（k 为匹配上标注的检测框数）。"""
    index = {e["id"]: k for k, e in enumerate(entries)}
    scale = CELL_W / 1200
    rows = []
    header = np.full((44, CELL_W * len(COLUMNS), 3), 255, np.uint8)
    for c, (_, title) in enumerate(COLUMNS):
        cv2.putText(header, title, (c * CELL_W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    rows.append(header)
    for i in ids:
        e = entries[index[i]]
        cells = []
        for g, _ in COLUMNS:
            path = next((DATA_DIR / g).glob(f"{i}.*")) if g in ("dark", "normal") else ENH_DIR / g / f"{i}.png"
            img = cv2.cvtColor(imread_rgb(path), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (CELL_W, round(img.shape[0] * CELL_W / img.shape[1])), interpolation=cv2.INTER_AREA)
            for x in e["boxes"]:
                p1, p2 = (int(x["x1"] * scale), int(x["y1"] * scale)), (int(x["x2"] * scale), int(x["y2"] * scale))
                cv2.rectangle(img, p1, p2, (0, 220, 0), 1)
            b, s, _ = dets[g][index[i]]
            for bb in b[s >= CONF_SHOW]:
                cv2.rectangle(img, (int(bb[0] * scale), int(bb[1] * scale)), (int(bb[2] * scale), int(bb[3] * scale)),
                              (0, 0, 255), 2)
            tag = f"found {true_positives(dets[g][index[i]], e)} / {len(e['boxes'])}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (0, 0), (tw + 14, th + 14), (0, 0, 0), -1)
            cv2.putText(img, tag, (7, th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(img)
        sep = np.full((cells[0].shape[0], 4, 3), 255, np.uint8)
        row = np.hstack([x for c in cells for x in (c, sep)][:-1])
        rows.append(cv2.resize(row, (CELL_W * len(COLUMNS), row.shape[0])))
        rows.append(np.full((4, CELL_W * len(COLUMNS), 3), 255, np.uint8))
    save_jpg(out, np.vstack(rows[:-1]), 88)
    print(out)


def enhancement_triplets(ids, out, width=640):
    """每行一个场景：输入（暗图，原样不提亮）| 输出（λ=10）| 真值（正常光图）。输出格写上和真值的 PSNR。"""
    titles = ["Input (dark image)", "Output (ours, lambda = 10)", "Ground truth (normal-light image)"]
    sep_w = 6
    total_w = 3 * width + 2 * sep_w
    header = np.full((48, total_w, 3), 255, np.uint8)
    for c, t in enumerate(titles):
        cv2.putText(header, t, (c * (width + sep_w) + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2,
                    cv2.LINE_AA)
    rows = [header]
    for i in ids:
        dark = cv2.cvtColor(imread_rgb(next((DATA_DIR / "dark").glob(f"{i}.*"))), cv2.COLOR_RGB2BGR)
        ours = cv2.cvtColor(imread_rgb(ENH_DIR / "ft_yolo_feat_w10" / f"{i}.png"), cv2.COLOR_RGB2BGR)
        gt = cv2.cvtColor(imread_rgb(next((DATA_DIR / "normal").glob(f"{i}.*"))), cv2.COLOR_RGB2BGR)
        tags = [f"PSNR {cv2.PSNR(dark, gt):.1f} dB", f"PSNR {cv2.PSNR(ours, gt):.1f} dB", f"scene {i}"]
        cells = []
        for img, tag in zip([dark, ours, gt], tags):
            img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(img, (0, 0), (tw + 14, th + 14), (0, 0, 0), -1)
            cv2.putText(img, tag, (7, th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(img)
        sep = np.full((cells[0].shape[0], sep_w, 3), 255, np.uint8)
        rows.append(np.hstack([cells[0], sep, cells[1], sep, cells[2]]))
        rows.append(np.full((sep_w, total_w, 3), 255, np.uint8))
    save_jpg(out, np.vstack(rows[:-1]), 88)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ids", type=int, nargs="*", default=[])
    ap.add_argument("--triplet_ids", type=int, nargs="*", default=[])
    args = ap.parse_args()

    entries = json.load(open(DATA_DIR / "labels.json", encoding="utf-8"))
    dets = {g: load_dets(g) for g, _ in COLUMNS}

    if args.list:
        rows = []
        for i, e in enumerate(entries):
            tp = {g: true_positives(dets[g][i], e) for g, _ in COLUMNS}
            rows.append((tp["ft_yolo_feat_w10"] - tp["ft_image_only"], e["id"], len(e["boxes"]), tp))
        rows.sort(key=lambda r: (-r[0], -r[2]))
        gain = np.array([r[0] for r in rows])
        print(f"λ=10 比 λ=0 多检出：更多 {int((gain > 0).sum())} 张，相同 {int((gain == 0).sum())} 张，更少 {int((gain < 0).sum())} 张")
        for r in rows[:25] + rows[-5:]:
            print(r[1], "gt", r[2], {g: r[3][g] for g, _ in COLUMNS})
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    map50_chart(FIG_DIR / "lis_map50.png")
    if args.ids:
        examples_grid(args.ids, entries, dets, FIG_DIR / "lis_examples.jpg")
    if args.triplet_ids:
        enhancement_triplets(args.triplet_ids, FIG_DIR / "lis_input_output_gt.jpg")


if __name__ == "__main__":
    main()
