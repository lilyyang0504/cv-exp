# 图像处理与视觉感知 — 实验代码

本仓库包含《图像处理与视觉感知》课程两个上机实验的完整 Python 实现，实验内容覆盖空间域图像处理与频率域图像处理的核心算法，所有算法均**手动实现**（不依赖 OpenCV 等库的高级封装），计算过程透明可追溯。

---

## 目录结构

```
cv-exp/
├── exp1/                          # 实验一：空间域图像处理基础
│   ├── work/                      # 输入图像目录（放置待处理图片）
│   │   └── two_cats.jpg           # 示例输入图像
│   ├── results/                   # 运行后自动生成的结果目录
│   │   ├── q1/                    # 灰度变换结果
│   │   ├── q2/                    # 直方图处理结果
│   │   ├── q3/                    # 几何变换结果
│   │   ├── q4/                    # 均值滤波结果
│   │   └── q5/                    # 直方图均衡化（矩阵题）结果
│   └── run_experiment1.py         # 实验一主程序
│
├── exp2/                          # 实验二：空间滤波与频率域处理
│   ├── img.jpg                    # 输入图像
│   ├── results/                   # 运行后自动生成的结果目录
│   │   ├── q1/                    # 噪声与空间平滑滤波结果
│   │   ├── q2/                    # 拉普拉斯锐化结果
│   │   ├── q3/                    # FFT 分析结果
│   │   ├── q4/                    # 频率域低通滤波结果
│   │   └── q5/                    # 频率域高通滤波结果
│   └── run_experiment2.py         # 实验二主程序
│
└── README.md
```

---

## 实验内容

### 实验一：空间域图像处理基础

| 题目 | 内容 |
|------|------|
| Q1 | 图像读取与常用灰度变换（线性拉伸、Gamma 变换、对数变换），并绘制各变换的灰度直方图 |
| Q2 | 灰度直方图分析、直方图均衡化、分段线性变换，对比变换前后效果 |
| Q3 | 几何变换：仿射平移（+30, +20 像素）与绕中心旋转 25°，采用双线性插值逆映射实现 |
| Q4 | 构造两幅 4×4 二值图像，分析 3×3 均值滤波（边缘复制填充）前后直方图的变化 |
| Q5 | 对给定 5×5 灰度矩阵（5 级灰度）手动执行直方图均衡化，可视化变换前后结果 |

**核心特点：**
- 不依赖 OpenCV，所有算法（灰度转换、直方图均衡、仿射变换、双线性插值等）均用纯 NumPy 手工实现
- 结果图像自动保存至 `exp1/results/` 对应子目录

### 实验二：空间滤波与频率域处理

| 题目 | 内容 |
|------|------|
| Q1 | 添加椒盐噪声与高斯噪声；对比 3×3 均值/中值滤波效果；对比零填充、边缘复制、镜像填充三种边界处理方式；设计自定义加权平滑核 |
| Q2 | 拉普拉斯锐化增强：使用 3×3、5×5、9×9、15×15、25×25 五种尺寸算子，对比锐化效果 |
| Q3 | FFT 特性分析：幅度谱与相位谱可视化；仅保留相位/幅度的逆变换；复共轭逆变换；4×4 矩阵的二维 DFT 计算 |
| Q4 | 频率域低通滤波：理想低通、巴特沃斯低通（n=2）、高斯低通，截止频率 D0 分别取 30 和 80 |
| Q5 | 频率域高通滤波：理想高通、巴特沃斯高通、高斯高通，截止频率 D0 分别取 30 和 80 |

---

## 环境依赖

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.9 | 推荐 3.10 / 3.11 |
| NumPy | ≥ 1.23 | 核心数值计算 |
| Matplotlib | ≥ 3.5 | 结果可视化与保存 |
| OpenCV (`cv2`) | ≥ 4.5 | 仅实验二用于图像读取与归一化 |

使用 pip 安装依赖：

```bash
pip install numpy matplotlib opencv-python
```

---

## 运行方法

所有命令均在**项目根目录**下执行。

### 实验一

```bash
# 使用默认参数运行（自动选取 exp1/work/ 下第一张图片）
python exp1/run_experiment1.py

# 指定输入图像
python exp1/run_experiment1.py --image-name two_cats.jpg

# 自定义输入/输出目录
python exp1/run_experiment1.py --work-dir exp1/work --out-dir exp1/results
```

运行完成后，结果图像会保存在 `exp1/results/q1/` ～ `exp1/results/q5/` 目录中。

### 实验二

```bash
# 使用默认参数运行（读取 exp2/img.jpg）
python exp2/run_experiment2.py

# 指定输入图像与输出目录
python exp2/run_experiment2.py --image-name img.jpg --work-dir exp2 --out-dir exp2/results
```

运行完成后，结果图像会保存在 `exp2/results/q1/` ～ `exp2/results/q5/` 目录中。

---

## 输出示例

每次运行后，各题目对应目录下会自动生成若干 PNG 图像，例如：

- `exp1/results/q1/q1_gray_transforms_and_hists.png` — 灰度变换对比图
- `exp1/results/q2/q2_b_hist_equalization_piecewise.png` — 直方图均衡化对比图
- `exp1/results/q3/q3_translate_rotate.png` — 平移与旋转结果
- `exp1/results/q4/q4_binary_and_filtered.png` — 二值图均值滤波前后对比
- `exp1/results/q5/q5_equalization.png` — 5×5 矩阵均衡化可视化
- `exp2/results/q1/q1_1_noise.png` — 噪声添加效果
- `exp2/results/q2/q2_sharpen.png` — 拉普拉斯多尺度锐化
- `exp2/results/q3/q3_123_fft.png` — FFT 幅度谱与相位谱
- `exp2/results/q4/q4_lpf_D0_30.png` — 低通滤波效果（D0=30）
- `exp2/results/q5/q5_hpf_D0_80.png` — 高通滤波效果（D0=80）

---

## 注意事项

1. **中文字体**：程序已配置优先使用 SimHei / Microsoft YaHei，若在 Linux/macOS 环境运行且无上述字体，图表中的中文可能显示为方框，可修改脚本顶部的 `plt.rcParams["font.sans-serif"]` 为本地可用字体。
2. **输入图像**：实验一的输入图片需放置在 `exp1/work/` 目录下（支持 `.jpg`、`.jpeg`、`.png`、`.bmp`）；实验二默认读取 `exp2/img.jpg`。
3. **结果目录**：程序运行时会自动创建缺失的输出目录，无需手动创建。
