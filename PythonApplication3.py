import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import data
import time

# ================= 1. 工具函数 =================

def calc_entropy(img):
    """计算图像信息熵"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calc_psnr_ssim(img_ref, img_test):
    """计算 PSNR 和 SSIM（以原图为参考）"""
    # 确保是 RGB 格式
    if len(img_ref.shape) == 3:
        p = psnr(img_ref, img_test, data_range=255)
        s = ssim(img_ref, img_test, channel_axis=2, data_range=255)
    else:
        p = psnr(img_ref, img_test, data_range=255)
        s = ssim(img_ref, img_test, data_range=255)
    return p, s

# ================= 2. 原版 SSR =================
def original_ssr(img, sigma=30):
    """原版单尺度 Retinex（高斯滤波）"""
    img_float = img.astype(np.float32) + 1.0
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
    retinex = np.log(img_float) - np.log(blurred)
    retinex_min = np.min(retinex, axis=(0, 1), keepdims=True)
    retinex_max = np.max(retinex, axis=(0, 1), keepdims=True)
    retinex = (retinex - retinex_min) / (retinex_max - retinex_min + 1e-6)
    return (retinex * 255).astype(np.uint8)

# ================= 3. 改进版 AGR =================
def guided_filter(guide, src, radius, eps):
    """引导滤波"""
    mean_I = cv2.boxFilter(guide, -1, (radius, radius))
    mean_p = cv2.boxFilter(src, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(guide * src, -1, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, -1, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    return mean_a * guide + mean_b

def improved_ssr(img, eps=100.0):
    """改进版 SSR：引导滤波 + 自适应半径 + 自适应增益"""
    img_float = img.astype(np.float32) + 1.0
    mean_brightness = np.mean(img_float)

    # 修改点 1: 自适应滤波半径
    radius = int(15 + (255.0 - mean_brightness) / 8.0)
    radius = max(5, radius)

    # 修改点 2: 引导滤波替代高斯滤波
    illumination = guided_filter(img_float, img_float, radius, eps)

    retinex = np.log(img_float) - np.log(illumination + 1.0)

    # 修改点 3: 自适应增益控制
    retinex_min = np.min(retinex, axis=(0, 1), keepdims=True)
    retinex_max = np.max(retinex, axis=(0, 1), keepdims=True)
    dynamic_range = retinex_max - retinex_min + 1e-8

    if mean_brightness < 100:
        gain = 1.15
    elif mean_brightness < 180:
        gain = 1.05
    else:
        gain = 0.95
    retinex = (retinex - retinex_min) / dynamic_range * gain
    retinex = np.clip(retinex, 0, 1)
    return (retinex * 255).astype(np.uint8)

# ================= 4. 主实验流程 =================
if __name__ == "__main__":
    print("=" * 60)
    print("Retinex 低照度图像增强 —— 原版 SSR vs 改进版 AGR")
    print("=" * 60)

    # 4.1 加载自然测试图像 (skimage 自带的宇航员图)
    test = data.astronaut()
    test = cv2.resize(test, (512, 512))
    img_low = (test * 0.3).astype(np.uint8)  # 人工压暗，模拟低照度

    # 4.2 运行算法
    t0 = time.time()
    img_ssr = original_ssr(img_low, sigma=30)
    time_ssr = time.time() - t0

    t1 = time.time()
    img_agr = improved_ssr(img_low, eps=100.0)
    time_agr = time.time() - t1

    # 4.3 计算指标
    ent_low = calc_entropy(img_low)
    ent_ssr = calc_entropy(img_ssr)
    ent_agr = calc_entropy(img_agr)

    # 以未压暗的清晰原图为参考标准
    psnr_ssr, ssim_ssr = calc_psnr_ssim(test, img_ssr)
    psnr_agr, ssim_agr = calc_psnr_ssim(test, img_agr)

    # 4.4 输出指标表格
    print("\n" + "=" * 75)
    print("【实验结果】（请将下方数据填入报告第 5 部分表格）")
    print(f"{'方法':<16} {'信息熵↑':<10} {'PSNR (dB)↑':<12} {'SSIM↑':<10} {'时间 (s)↓':<10}")
    print("-" * 75)
    print(f"{'低照度原图':<16} {ent_low:<10.4f} {'—':<12} {'—':<10} {'—':<10}")
    print(f"{'原版 SSR':<16} {ent_ssr:<10.4f} {psnr_ssr:<12.2f} {ssim_ssr:<10.4f} {time_ssr:<10.4f}")
    print(f"{'改进版 AGR':<16} {ent_agr:<10.4f} {psnr_agr:<12.2f} {ssim_agr:<10.4f} {time_agr:<10.4f}")
    print("=" * 75)
    print("✅ 运行完全结束！数据计算完毕，正在生成对比图...\n")

    # 4.5 可视化与保存
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["低照度原图", "原版 SSR（高斯滤波）", "改进版 AGR（引导滤波+自适应）"]
    images = [img_low, img_ssr, img_agr]

    for ax, title, im in zip(axes, titles, images):
        ax.imshow(im)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("retinex_对比结果.png", dpi=300, bbox_inches='tight')
    print("✅ 对比图已保存为 'retinex_对比结果.png'")
    plt.show()