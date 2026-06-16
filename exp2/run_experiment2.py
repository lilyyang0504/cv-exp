import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
from numpy.lib.stride_tricks import sliding_window_view

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-name', type=str, default='img.jpg')
    parser.add_argument('--work-dir', type=str, default='exp2/work')
    parser.add_argument('--out-dir', type=str, default='exp2/results')
    return parser.parse_args()

# --- UTILS ---
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def add_noise(image, noise_type='salt_pepper', param=0.05):
    noisy = image.copy()
    if noise_type == 'salt_pepper':
        prob = param
        rnd = np.random.rand(*noisy.shape)
        noisy[rnd < prob/2] = 0
        noisy[rnd > 1 - prob/2] = 255
    elif noise_type == 'gaussian':
        std = param
        gauss = np.random.normal(0, std, noisy.shape)
        noisy = np.clip(noisy.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return noisy

def manual_convolve2d(image, kernel, padding='zero'):
    h, w = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    
    if padding == 'zero':
        mode = 'constant'
    elif padding == 'replicate':
        mode = 'edge'
    elif padding == 'symmetric':
        mode = 'symmetric'
    else:
        mode = 'constant'

    padded = np.pad(image, ((ph, ph), (pw, pw)), mode=mode)
    windows = sliding_window_view(padded, (kh, kw))
    output = np.einsum('ijkl,kl->ij', windows, kernel)
    return output

def manual_median_filter(image, size=3, padding='symmetric'):
    h, w = image.shape
    p = size // 2
    padded = np.pad(image, ((p, p), (p, p)), mode='symmetric' if padding=='symmetric' else 'constant')
    windows = sliding_window_view(padded, (size, size))
    windows_flat = windows.reshape(h, w, -1)
    output = np.median(windows_flat, axis=-1)
    return output

def genlaplacian(n):
    w = np.ones((n, n), dtype=np.float32)
    w[n//2, n//2] = -(n*n - 1)
    return w

def get_meshgrid_distances(h, w):
    u = np.arange(h)
    v = np.arange(w)
    u[u > h/2] -= h
    v[v > w/2] -= w
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    return D

def get_ideal_lpf(h, w, D0):
    D = get_meshgrid_distances(h, w)
    H = (D <= D0).astype(np.float32)
    return H

def get_butterworth_lpf(h, w, D0, n=2):
    D = get_meshgrid_distances(h, w)
    H = 1 / (1 + (D / D0)**(2 * n))
    return H

def get_gaussian_lpf(h, w, D0):
    D = get_meshgrid_distances(h, w)
    H = np.exp(- (D**2) / (2 * D0**2))
    return H

def apply_frequency_filter(image, H):
    f = np.fft.fft2(image)
    # H is designed without fftshift, the corners are low frequencies
    g = f * H
    img_back = np.fft.ifft2(g)
    img_back = np.abs(img_back)
    return np.clip(img_back, 0, 255).astype(np.uint8)

# --- Q1 ---
def run_q1(image, out_dir):
    q1_dir = os.path.join(out_dir, 'q1')
    ensure_dir(q1_dir)
    
    # 1.1 Add noise
    sp_noise = add_noise(image, 'salt_pepper', 0.05)
    gs_noise = add_noise(image, 'gaussian', 25)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.axis('off'), plt.imshow(image, cmap='gray'), plt.title('原图')
    plt.subplot(132), plt.axis('off'), plt.imshow(sp_noise, cmap='gray'), plt.title('椒盐噪声')
    plt.subplot(133), plt.axis('off'), plt.imshow(gs_noise, cmap='gray'), plt.title('高斯噪声')
    plt.savefig(os.path.join(q1_dir, 'q1_1_noise.png')), plt.close()
    
    # 1.2 Compare smooth templates
    k3 = np.ones((3, 3)) / 9
    k5 = np.ones((5, 5)) / 25
    
    mean3_sp = manual_convolve2d(sp_noise, k3)
    med3_sp = manual_median_filter(sp_noise, 3)
    mean3_gs = manual_convolve2d(gs_noise, k3)
    med3_gs = manual_median_filter(gs_noise, 3)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(231), plt.axis('off'), plt.imshow(sp_noise, cmap='gray'), plt.title('椒盐噪声')
    plt.subplot(232), plt.axis('off'), plt.imshow(mean3_sp, cmap='gray'), plt.title('3x3均值 (椒盐)')
    plt.subplot(233), plt.axis('off'), plt.imshow(med3_sp, cmap='gray'), plt.title('3x3中值 (椒盐)')
    plt.subplot(234), plt.axis('off'), plt.imshow(gs_noise, cmap='gray'), plt.title('高斯噪声')
    plt.subplot(235), plt.axis('off'), plt.imshow(mean3_gs, cmap='gray'), plt.title('3x3均值 (高斯)')
    plt.subplot(236), plt.axis('off'), plt.imshow(med3_gs, cmap='gray'), plt.title('3x3中值 (高斯)')
    plt.savefig(os.path.join(q1_dir, 'q1_2_smooth_compare.png')), plt.close()
    
    # 1.3 Padding methods
    p_zero = manual_convolve2d(sp_noise, k5, 'zero')
    p_repl = manual_convolve2d(sp_noise, k5, 'replicate')
    p_symm = manual_convolve2d(sp_noise, k5, 'symmetric')
    
    plt.figure(figsize=(15, 4))
    plt.subplot(141), plt.axis('off'), plt.imshow(sp_noise, cmap='gray'), plt.title('椒盐噪声')
    plt.subplot(142), plt.axis('off'), plt.imshow(p_zero, cmap='gray'), plt.title('零填充 (Zero)')
    plt.subplot(143), plt.axis('off'), plt.imshow(p_repl, cmap='gray'), plt.title('边缘复制 (Replicate)')
    plt.subplot(144), plt.axis('off'), plt.imshow(p_symm, cmap='gray'), plt.title('镜像填充 (Symmetric)')
    plt.savefig(os.path.join(q1_dir, 'q1_3_padding.png')), plt.close()
    
    # 1.4 Custom smoothing filter (Weighted Mean)
    custom_k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    custom_sp = manual_convolve2d(sp_noise, custom_k)
    custom_gs = manual_convolve2d(gs_noise, custom_k)
    
    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(131)
    ax1.axis('off')
    ax1.imshow(custom_k, cmap='gray')
    ax1.set_title('自定义加权核 (3x3)')
    # Show values on the kernel
    for i in range(3):
        for j in range(3):
            val = int(round(custom_k[i, j] * 16))
            # Determine font color for contrast: threshold is 2.5/16 given min=1/16, max=4/16
            font_color = "black" if custom_k[i, j] > 2.5/16 else "white"
            ax1.text(j, i, f"{val}/16", ha="center", va="center", color=font_color, fontsize=14, fontweight='bold')
            
    plt.subplot(132), plt.axis('off'), plt.imshow(custom_sp, cmap='gray'), plt.title('自定义平滑 (椒盐)')
    plt.subplot(133), plt.axis('off'), plt.imshow(custom_gs, cmap='gray'), plt.title('自定义平滑 (高斯)')
    plt.savefig(os.path.join(q1_dir, 'q1_4_custom_filter.png')), plt.close()


# --- Q2 ---
def run_q2(image, out_dir):
    q2_dir = os.path.join(out_dir, 'q2')
    ensure_dir(q2_dir)
    image_f = image.astype(np.float32)
    
    sizes = [3, 5, 9, 15, 25]
    
    plt.figure(figsize=(20, 8))
    ax_orig = plt.subplot2grid((2, 6), (0, 0), rowspan=2)
    ax_orig.axis('off')
    ax_orig.imshow(image, cmap='gray')
    # Because it spans two rows, the font size can be slightly larger.
    ax_orig.set_title('原图', fontsize=16)
    
    for i, s in enumerate(sizes):
        if s == 3:
            w = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        else:
            w = genlaplacian(s)
        
        # Laplacian filter
        lap = manual_convolve2d(image_f, w, 'symmetric')
        
        # Enhanced image g = f - nabla^2 f
        g = image_f - lap
        g = np.clip(g, 0, 255).astype(np.uint8)
        
        lap_vis = np.clip(lap + 128, 0, 255).astype(np.uint8) # offset for visualization
        
        ax_lap = plt.subplot2grid((2, 6), (0, i+1))
        ax_lap.axis('off')
        ax_lap.imshow(lap_vis, cmap='gray')
        ax_lap.set_title(f'拉普拉斯 {s}x{s}')
        
        ax_g = plt.subplot2grid((2, 6), (1, i+1))
        ax_g.axis('off')
        ax_g.imshow(g, cmap='gray')
        ax_g.set_title(f'锐化增强 {s}x{s}')
        
    plt.tight_layout()
    plt.savefig(os.path.join(q2_dir, 'q2_sharpen.png')), plt.close()


# --- Q3 ---
def run_q3(image, out_dir):
    q3_dir = os.path.join(out_dir, 'q3')
    ensure_dir(q3_dir)
    
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    mag_vis = np.log(1 + magnitude)
    
    # 3.1 Mag & Phase
    plt.figure(figsize=(18, 4))
    plt.subplot(151), plt.axis('off'), plt.imshow(image, cmap='gray'), plt.title('原图')
    plt.subplot(152), plt.axis('off'), plt.imshow(mag_vis, cmap='gray'), plt.title('幅度谱')
    plt.subplot(153), plt.axis('off'), plt.imshow(phase, cmap='gray'), plt.title('相位谱')
    
    # 3.2 Only Phase (Mag = 1)
    f_phase_only = 1 * np.exp(1j * np.angle(f))
    img_phase_only = np.real(np.fft.ifft2(f_phase_only))
    
    # Normalize for display
    img_phase_only = cv2.normalize(img_phase_only, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3.3 Only Mag (Phase = 0)
    f_mag_only = np.abs(f) * np.exp(1j * 0)
    img_mag_only = np.real(np.fft.ifft2(f_mag_only))
    img_mag_only = np.log(1 + np.abs(img_mag_only))
    img_mag_only = cv2.normalize(img_mag_only, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    plt.subplot(154), plt.axis('off'), plt.imshow(img_phase_only, cmap='gray'), plt.title('仅保留相位逆变换')
    plt.subplot(155), plt.axis('off'), plt.imshow(img_mag_only, cmap='gray'), plt.title('仅保留幅度逆变换')
    plt.savefig(os.path.join(q3_dir, 'q3_123_fft.png')), plt.close()
    
    # 3.4 Conjugate
    f_conj = np.conj(f)
    img_conj = np.real(np.fft.ifft2(f_conj))
    img_conj = np.clip(img_conj, 0, 255).astype(np.uint8)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(121), plt.axis('off'), plt.imshow(image, cmap='gray'), plt.title('原图')
    plt.subplot(122), plt.axis('off'), plt.imshow(img_conj, cmap='gray'), plt.title('共轭逆变换')
    plt.savefig(os.path.join(q3_dir, 'q3_4_conjugate.png')), plt.close()

# --- Q3.5 ---
def run_q3_5(out_dir):
    q3_dir = os.path.join(out_dir, 'q3')
    ensure_dir(q3_dir)
    
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[:, 3] = 12
    
    # 傅里叶变换
    f = np.fft.fft2(mat)
    fshift = np.fft.fftshift(f) # 移到中心以便观察
    mag = np.abs(fshift)
    
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)
    ax1.axis('off')
    ax1.imshow(mat, cmap='gray', vmin=0, vmax=12)
    ax1.set_title('原始 4x4 矩阵', fontsize=14)
    for i in range(4):
        for j in range(4):
            val = int(mat[i, j])
            color = "white" if val == 0 else "black"
            ax1.text(j, i, str(val), ha="center", va="center", color=color, fontsize=16, fontweight='bold')
            
    ax2 = plt.subplot(122)
    ax2.axis('off')
    # 显示幅度，为了对比明显用 log 也可以，但这里直接显示值就行，因为只有中间行有值
    ax2.imshow(mag, cmap='gray')
    ax2.set_title('其二维DFT的幅度谱 |F(u,v)|', fontsize=14)
    for i in range(4):
        for j in range(4):
            val = int(round(mag[i, j]))
            color = "red" if val > 0 else "white"
            ax2.text(j, i, str(val), ha="center", va="center", color=color, fontsize=16, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(q3_dir, 'q3_5_matrix_fft.png'))
    plt.close()

# --- Q4 & Q5 ---
def run_q4_q5(image, out_dir):
    q4_dir = os.path.join(out_dir, 'q4')
    q5_dir = os.path.join(out_dir, 'q5')
    ensure_dir(q4_dir)
    ensure_dir(q5_dir)
    
    h, w = image.shape
    D0_list = [30, 80]
    
    for D0 in D0_list:
        ilpf = get_ideal_lpf(h, w, D0)
        blpf = get_butterworth_lpf(h, w, D0, 2)
        glpf = get_gaussian_lpf(h, w, D0)
        
        ihpf = 1 - ilpf
        bhpf = 1 - blpf
        ghpf = 1 - glpf
        
        # LPF
        img_ilpf = apply_frequency_filter(image, ilpf)
        img_blpf = apply_frequency_filter(image, blpf)
        img_glpf = apply_frequency_filter(image, glpf)
        
        plt.figure(figsize=(16, 4))
        plt.subplot(141), plt.axis('off'), plt.imshow(image, cmap='gray'), plt.title(f'原图 (D0={D0})')
        plt.subplot(142), plt.axis('off'), plt.imshow(img_ilpf, cmap='gray'), plt.title('理想低通 (注意振铃效应)')
        plt.subplot(143), plt.axis('off'), plt.imshow(img_blpf, cmap='gray'), plt.title('巴特沃斯低通')
        plt.subplot(144), plt.axis('off'), plt.imshow(img_glpf, cmap='gray'), plt.title('高斯低通')
        plt.savefig(os.path.join(q4_dir, f'q4_lpf_D0_{D0}.png')), plt.close()
        
        # HPF
        img_ihpf = apply_frequency_filter(image, ihpf)
        img_bhpf = apply_frequency_filter(image, bhpf)
        img_ghpf = apply_frequency_filter(image, ghpf)
        
        plt.figure(figsize=(16, 4))
        plt.subplot(141), plt.axis('off'), plt.imshow(image, cmap='gray'), plt.title(f'原图 (D0={D0})')
        plt.subplot(142), plt.axis('off'), plt.imshow(img_ihpf, cmap='gray'), plt.title('理想高通 (注意振铃效应)')
        plt.subplot(143), plt.axis('off'), plt.imshow(img_bhpf, cmap='gray'), plt.title('巴特沃斯高通')
        plt.subplot(144), plt.axis('off'), plt.imshow(img_ghpf, cmap='gray'), plt.title('高斯高通')
        plt.savefig(os.path.join(q5_dir, f'q5_hpf_D0_{D0}.png')), plt.close()

def main():
    args = get_args()
    img_path = os.path.join(os.path.dirname(__file__), args.image_name)
    print(f"Reading image from {img_path}")
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read {img_path}. Make sure it exists.")
        
    print("Running Q1: Spatial Smooth Filters...")
    run_q1(image, args.out_dir)
    print("Running Q2: Spatial Sharpen Filters...")
    run_q2(image, args.out_dir)
    print("Running Q3: FFT Properties...")
    run_q3(image, args.out_dir)
    run_q3_5(args.out_dir)
    print("Running Q4 & Q5: Frequency Filters...")
    run_q4_q5(image, args.out_dir)
    print("Optimization finished successfully.")

if __name__ == "__main__":
    main()
