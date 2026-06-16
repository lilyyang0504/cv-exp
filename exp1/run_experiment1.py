from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_image_float01(image_path: Path) -> np.ndarray:
    img = plt.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    arr = np.asarray(img)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
        return np.clip(gray, 0.0, 1.0)

    if arr.shape[2] == 4:
        arr = arr[..., :3]

    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def rgb2gray(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.ndim == 2:
        return img_rgb.astype(np.float32)
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray.astype(np.float32), 0.0, 1.0)


def stretch_to_01(gray: np.ndarray) -> np.ndarray:
    gmin, gmax = float(gray.min()), float(gray.max())
    if np.isclose(gmax, gmin):
        return np.zeros_like(gray)
    return (gray - gmin) / (gmax - gmin)


def gamma_transform(gray: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(np.power(np.clip(gray, 0.0, 1.0), gamma), 0.0, 1.0)


def log_transform(gray: np.ndarray, c: float = 1.0) -> np.ndarray:
    out = c * np.log1p(np.clip(gray, 0.0, 1.0))
    return stretch_to_01(out)


def hist_equalize(gray: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    # gk = INT[(L-1) * sk + 0.5], 其中 sk 是CDF
    l = int(bins)
    gray_u8 = np.clip(np.round(gray * (l - 1)), 0, l - 1).astype(np.int32)

    hist = np.zeros(l, dtype=np.int64)
    flat = gray_u8.flatten()
    for v in flat:
        hist[int(v)] += 1

    pr = hist.astype(np.float64) / float(flat.size)
    cdf = np.cumsum(pr)
    mapping = np.floor((l - 1) * cdf + 0.5).astype(np.int32)

    out_u8 = mapping[gray_u8]
    out = out_u8.astype(np.float32) / float(l - 1)
    return out, cdf


def piecewise_transform(gray: np.ndarray) -> np.ndarray:
    # 来自变换函数r1=1/8, s1=2/8, r2=6/8, s2=5/8 (归一化后)
    r1, s1 = 1.0 / 8.0, 2.0 / 8.0
    r2, s2 = 6.0 / 8.0, 5.0 / 8.0

    out = np.zeros_like(gray, dtype=np.float32)
    m1 = gray < r1
    m2 = (gray >= r1) & (gray < r2)
    m3 = gray >= r2

    out[m1] = (s1 / r1) * gray[m1]
    out[m2] = ((s2 - s1) / (r2 - r1)) * (gray[m2] - r1) + s1
    out[m3] = ((1.0 - s2) / (1.0 - r2)) * (gray[m3] - r2) + s2
    return np.clip(out, 0.0, 1.0)


def save_img(path: Path, img: np.ndarray, cmap: str | None = None) -> None:
    plt.figure(figsize=(5, 4))
    if img.ndim == 2:
        plt.imshow(img, cmap=cmap or "gray", vmin=0.0, vmax=1.0)
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_hist(ax, gray: np.ndarray, title: str) -> None:
    ax.hist(np.clip(gray.flatten(), 0.0, 1.0), bins=256, range=(0, 1), color="tab:blue")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_xlabel("灰度值")
    ax.set_ylabel("像素数")


def plot_bar_with_values(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    bar_width: float | None = None,
) -> None:
    if bar_width is None:
        if x.size <= 1:
            bar_width = 0.3
        else:
            sorted_x = np.sort(x.astype(np.float64))
            min_gap = float(np.min(np.diff(sorted_x)))
            bar_width = max(0.01, min(0.4, 0.6 * min_gap))
    ax.bar(x, y, width=bar_width)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for xi, yi in zip(x, y):
        ax.text(float(xi), float(yi), f"{int(yi)}", ha="center", va="bottom", fontsize=9)


def visualize_matrix_with_values(path: Path, matrix: np.ndarray, title: str, decimals: int = 3) -> None:
    h, w = matrix.shape
    fig, ax = plt.subplots(figsize=(w * 1.8, h * 1.8))
    im = ax.imshow(matrix, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(np.arange(w))
    ax.set_yticks(np.arange(h))
    for i in range(h):
        for j in range(w):
            v = float(matrix[i, j])
            txt_color = "white" if v < 0.5 else "black"
            ax.text(j, i, f"{v:.{decimals}f}", ha="center", va="center", color=txt_color, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def mean_filter3x3_with_edge_pad(img: np.ndarray) -> np.ndarray:
    pad = np.pad(img, ((1, 1), (1, 1)), mode="edge")
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            out[i, j] = pad[i : i + 3, j : j + 3].mean()
    return out


def hist_counts(img: np.ndarray) -> Dict[float, int]:
    vals, counts = np.unique(img, return_counts=True)
    return {float(v): int(c) for v, c in zip(vals, counts)}


def bilinear_sample_with_border(img: np.ndarray, x: float, y: float, border_value: float = 0.0) -> float:
    h, w = img.shape
    if x < 0 or x > (w - 1) or y < 0 or y > (h - 1):
        return float(border_value)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    def get_pixel(xx: int, yy: int) -> float:
        if 0 <= xx < w and 0 <= yy < h:
            return float(img[yy, xx])
        return float(border_value)

    p00 = get_pixel(x0, y0)
    p10 = get_pixel(x1, y0)
    p01 = get_pixel(x0, y1)
    p11 = get_pixel(x1, y1)

    top = (1.0 - dx) * p00 + dx * p10
    bottom = (1.0 - dx) * p01 + dx * p11
    return float((1.0 - dy) * top + dy * bottom)


def warp_affine_bilinear(
    src: np.ndarray,
    m: np.ndarray,
    out_h: int,
    out_w: int,
    border_value: float = 0.0,
) -> np.ndarray:
    m33 = np.array([[m[0, 0], m[0, 1], m[0, 2]], [m[1, 0], m[1, 1], m[1, 2]], [0.0, 0.0, 1.0]], dtype=np.float64)
    inv = np.linalg.inv(m33)

    out = np.zeros((out_h, out_w), dtype=np.float32)
    for y_d in range(out_h):
        for x_d in range(out_w):
            src_pos = inv @ np.array([x_d, y_d, 1.0], dtype=np.float64)
            x_s, y_s = float(src_pos[0]), float(src_pos[1])
            out[y_d, x_d] = bilinear_sample_with_border(src, x_s, y_s, border_value=border_value)
    return np.clip(out, 0.0, 1.0)


def apply_translation_and_rotation(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    src = np.clip(gray, 0.0, 1.0).astype(np.float32)

    m_translate = np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 20.0]], dtype=np.float64)
    translated = warp_affine_bilinear(src, m_translate, h, w, border_value=0.0)

    center = (w / 2.0, h / 2.0)
    angle_deg = 25.0
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    cx, cy = center
    tx = cx - cos_t * cx + sin_t * cy
    ty = cy - sin_t * cx - cos_t * cy
    m_rotate = np.array([[cos_t, -sin_t, tx], [sin_t, cos_t, ty]], dtype=np.float64)
    rotated = warp_affine_bilinear(src, m_rotate, h, w, border_value=0.0)

    return translated.astype(np.float32), rotated.astype(np.float32)


def solve_q5_equalization(matrix: np.ndarray, levels: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = matrix.flatten().astype(int)
    hist = np.zeros(levels, dtype=int)
    for v in flat:
        hist[v] += 1

    cdf = np.cumsum(hist) / flat.size
    mapping = np.floor((levels - 1) * cdf + 0.5).astype(int)
    eq = np.vectorize(lambda x: mapping[x])(matrix)

    hist_eq = np.zeros(levels, dtype=int)
    for v in eq.flatten():
        hist_eq[v] += 1
    return hist, hist_eq, eq


def run_experiment_1(work_dir: Path, out_dir: Path, image_name: str | None) -> None:
    ensure_dir(out_dir)
    q1_dir = out_dir / "q1"
    q2_dir = out_dir / "q2"
    q3_dir = out_dir / "q3"
    q4_dir = out_dir / "q4"
    q5_dir = out_dir / "q5"
    for d in [q1_dir, q2_dir, q3_dir, q4_dir, q5_dir]:
        ensure_dir(d)

    if image_name is None:
        candidates = sorted([p for p in work_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        if not candidates:
            raise FileNotFoundError(f"work目录中未找到图片: {work_dir}")
        image_path = candidates[0]
    else:
        image_path = work_dir / image_name

    img_rgb = read_image_float01(image_path)
    gray = rgb2gray(img_rgb)

    # Q1: 读图 + 常用灰度变换
    gray_stretch = stretch_to_01(gray)
    gray_gamma_05 = gamma_transform(gray, 0.5)
    gray_gamma_15 = gamma_transform(gray, 1.5)
    gray_log = log_transform(gray)

    fig, axes = plt.subplots(6, 2, figsize=(12, 20))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("原始彩色图")
    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5, "原图不绘制直方图", ha="center", va="center", fontsize=11)

    q1_rows = [
        (gray, "灰度图", "灰度图直方图"),
        (gray_stretch, "线性拉伸到[0,1]", "拉伸后直方图"),
        (gray_gamma_05, "伽马变换(0.5)", "伽马0.5直方图"),
        (gray_gamma_15, "伽马变换(1.5)", "伽马1.5直方图"),
        (gray_log, "对数变换", "对数变换直方图"),
    ]

    for row_idx, (im, img_title, hist_title) in enumerate(q1_rows, start=1):
        axes[row_idx, 0].imshow(im, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 0].set_title(img_title)
        axes[row_idx, 0].axis("off")
        plot_hist(axes[row_idx, 1], im, hist_title)

    fig.tight_layout()
    fig.savefig(q1_dir / "q1_gray_transforms_and_hists.png", dpi=160)
    plt.close(fig)

    # Q2: B直方图、均衡化、分段线性变换
    b = gray
    b_norm = stretch_to_01(b)
    b_eq, _ = hist_equalize(b)
    b_piece = piecewise_transform(b)

    fig, axes = plt.subplots(4, 2, figsize=(11, 14))
    axes[0, 0].imshow(b, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("B灰度图")
    axes[0, 0].axis("off")
    plot_hist(axes[0, 1], b, "B的灰度直方图")

    axes[1, 0].imshow(b_norm, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("B灰度拉伸到[0,1]")
    axes[1, 0].axis("off")
    plot_hist(axes[1, 1], b_norm, "拉伸后直方图")

    axes[2, 0].imshow(b_eq, cmap="gray", vmin=0, vmax=1)
    axes[2, 0].set_title("B直方图均衡化后")
    axes[2, 0].axis("off")
    plot_hist(axes[2, 1], b_eq, "均衡化后直方图")

    axes[3, 0].imshow(b_piece, cmap="gray", vmin=0, vmax=1)
    axes[3, 0].set_title("B分段线性变换后")
    axes[3, 0].axis("off")
    plot_hist(axes[3, 1], b_piece, "分段线性后直方图")

    fig.tight_layout()
    fig.savefig(q2_dir / "q2_b_hist_equalization_piecewise.png", dpi=160)
    plt.close(fig)

    # Q3: 平移与旋转
    translated, rotated = apply_translation_and_rotation(b)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(b, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("原图(B)")
    axes[0].axis("off")
    axes[1].imshow(translated, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("平移(+30,+20)")
    axes[1].axis("off")
    axes[2].imshow(rotated, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("旋转25度")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(q3_dir / "q3_translate_rotate.png", dpi=160)
    plt.close(fig)

    # Q4: 两幅4x4二值图 + 3x3均值滤波
    left = np.array(
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 0, 0]],
        dtype=np.float32,
    )
    right = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 1, 0, 1]],
        dtype=np.float32,
    )

    hist_left = hist_counts(left)
    hist_right = hist_counts(right)
    same_before = hist_left == hist_right

    left_f = mean_filter3x3_with_edge_pad(left)
    right_f = mean_filter3x3_with_edge_pad(right)

    hist_left_f = hist_counts(np.round(left_f, 6))
    hist_right_f = hist_counts(np.round(right_f, 6))
    same_after = hist_left_f == hist_right_f

    # 按2行4列排版：每行分别表示滤波前/后；每组按“图像+对应直方图”并排
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    axes[0, 0].imshow(left, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 0].set_title("left（原始）")
    axes[0, 0].axis("off")
    plot_bar_with_values(
        axes[0, 1],
        np.array([0, 1]),
        np.array([hist_left.get(0.0, 0), hist_left.get(1.0, 0)]),
        "left滤波前直方图",
        "灰度级",
        "像素数",
        bar_width=0.25,
    )

    axes[0, 2].imshow(right, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 2].set_title("right（原始）")
    axes[0, 2].axis("off")
    plot_bar_with_values(
        axes[0, 3],
        np.array([0, 1]),
        np.array([hist_right.get(0.0, 0), hist_right.get(1.0, 0)]),
        "right滤波前直方图",
        "灰度级",
        "像素数",
        bar_width=0.25,
    )

    left_after_keys = np.array(sorted(hist_left_f.keys()), dtype=np.float64)
    left_after_vals = np.array([hist_left_f[k] for k in left_after_keys], dtype=np.int64)
    right_after_keys = np.array(sorted(hist_right_f.keys()), dtype=np.float64)
    right_after_vals = np.array([hist_right_f[k] for k in right_after_keys], dtype=np.int64)

    axes[1, 0].imshow(left_f, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 0].set_title("left（3x3均值滤波后）")
    axes[1, 0].axis("off")
    plot_bar_with_values(
        axes[1, 1],
        left_after_keys,
        left_after_vals,
        "left滤波后直方图",
        "灰度级",
        "像素数",
        bar_width=0.04,
    )

    axes[1, 2].imshow(right_f, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 2].set_title("right（3x3均值滤波后）")
    axes[1, 2].axis("off")
    plot_bar_with_values(
        axes[1, 3],
        right_after_keys,
        right_after_vals,
        "right滤波后直方图",
        "灰度级",
        "像素数",
        bar_width=0.04,
    )
    fig.tight_layout()
    fig.savefig(q4_dir / "q4_binary_and_filtered.png", dpi=200)
    plt.close(fig)

    visualize_matrix_with_values(
        q4_dir / "q4_left_filtered_gray_levels.png",
        left_f,
        "left图像滤波后各像素灰度值（可视化）",
        decimals=3,
    )


    # Q5: 给定5x5矩阵均衡化
    m = np.array(
        [[3, 3, 0, 0, 1],
         [3, 3, 1, 1, 4],
         [3, 3, 2, 1, 1],
         [3, 3, 2, 4, 1],
         [2, 2, 1, 1, 4]],
        dtype=int,
    )

    hist_m, hist_m_eq, m_eq = solve_q5_equalization(m, levels=5)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(m, cmap="gray", vmin=0, vmax=4, interpolation="nearest")
    axes[0, 0].set_title("原始5x5矩阵")
    axes[0, 0].axis("off")
    plot_bar_with_values(axes[0, 1], np.arange(5), hist_m, "原始直方图", "灰度级", "像素数", bar_width=0.35)
    axes[0, 1].set_xticks(np.arange(5))

    axes[1, 0].imshow(m_eq, cmap="gray", vmin=0, vmax=4, interpolation="nearest")
    axes[1, 0].set_title("均衡化后5x5矩阵")
    axes[1, 0].axis("off")
    plot_bar_with_values(axes[1, 1], np.arange(5), hist_m_eq, "均衡化后直方图", "灰度级", "像素数", bar_width=0.35)
    axes[1, 1].set_xticks(np.arange(5))

    fig.tight_layout()
    fig.savefig(q5_dir / "q5_equalization.png", dpi=180)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("exp1/work"), help="输入图像目录")
    parser.add_argument("--out-dir", type=Path, default=Path("exp1/results"), help="输出结果目录")
    parser.add_argument("--image-name", type=str, default=None, help="指定输入图像文件名，不指定则自动选取第一张")
    args = parser.parse_args()

    run_experiment_1(args.work_dir, args.out_dir, args.image_name)
