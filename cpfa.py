"""
CPFA Raw Image Generator
========================

根据论文描述处理 scene_001 - scene_005:
1. 读取 HR GT: RGB_0.png, RGB_45.png, RGB_90.png, RGB_135.png
2. 下采样得到 LR GT (512 × 612)
3. 对 LR GT 添加 mosaic 生成 CPFA Raw

数据流程 (2× SR):
HR GT (1024×1224) --[downsample]--> LR GT (512×612) --[mosaic]--> CPFA Raw (512×612)
"""

import numpy as np
import cv2
import os


def generate_cpfa_raw(I_0, I_45, I_90, I_135, bayer_pattern='RGGB'):
    """
    将4个偏振方向的LR GT图像合成为CPFA原始图像

    CPFA 4×4 超像素结构 (RGGB模式):
    ┌──────┬──────┬──────┬──────┐
    │ 90,R │ 45,R │ 90,G │ 45,G │
    ├──────┼──────┼──────┼──────┤
    │135,R │  0,R │135,G │  0,G │
    ├──────┼──────┼──────┼──────┤
    │ 90,G │ 45,G │ 90,B │ 45,B │
    ├──────┼──────┼──────┼──────┤
    │135,G │  0,G │135,B │  0,B │
    └──────┴──────┴──────┴──────┘
    """
    H, W, C = I_0.shape

    # Bayer模式对应的颜色通道索引 (OpenCV BGR格式)
    patterns = {
        'RGGB': [2, 1, 1, 0],  # R, G, G, B
        'BGGR': [0, 1, 1, 2],
        'GBRG': [1, 0, 2, 1],
        'GRBG': [1, 2, 0, 1],
    }
    color = patterns[bayer_pattern]

    # 创建CPFA原始图像
    cpfa_raw = np.zeros((H, W), dtype=I_0.dtype)

    # 第0行: 90°, 45°, 90°, 45°
    cpfa_raw[0::4, 0::4] = I_90[0::4, 0::4, color[0]]
    cpfa_raw[0::4, 1::4] = I_45[0::4, 1::4, color[0]]
    cpfa_raw[0::4, 2::4] = I_90[0::4, 2::4, color[1]]
    cpfa_raw[0::4, 3::4] = I_45[0::4, 3::4, color[1]]

    # 第1行: 135°, 0°, 135°, 0°
    cpfa_raw[1::4, 0::4] = I_135[1::4, 0::4, color[0]]
    cpfa_raw[1::4, 1::4] = I_0[1::4, 1::4, color[0]]
    cpfa_raw[1::4, 2::4] = I_135[1::4, 2::4, color[1]]
    cpfa_raw[1::4, 3::4] = I_0[1::4, 3::4, color[1]]

    # 第2行: 90°, 45°, 90°, 45°
    cpfa_raw[2::4, 0::4] = I_90[2::4, 0::4, color[2]]
    cpfa_raw[2::4, 1::4] = I_45[2::4, 1::4, color[2]]
    cpfa_raw[2::4, 2::4] = I_90[2::4, 2::4, color[3]]
    cpfa_raw[2::4, 3::4] = I_45[2::4, 3::4, color[3]]

    # 第3行: 135°, 0°, 135°, 0°
    cpfa_raw[3::4, 0::4] = I_135[3::4, 0::4, color[2]]
    cpfa_raw[3::4, 1::4] = I_0[3::4, 1::4, color[2]]
    cpfa_raw[3::4, 2::4] = I_135[3::4, 2::4, color[3]]
    cpfa_raw[3::4, 3::4] = I_0[3::4, 3::4, color[3]]

    return cpfa_raw


def process_scenes(input_dir, output_dir, scenes=None, scale_factor=2, bayer_pattern='RGGB'):
    """
    处理多个场景，生成CPFA原始图像

    Args:
        input_dir: 输入目录，包含 scene_001, scene_002, ... 等子文件夹
        output_dir: 输出目录
        scenes: 要处理的场景列表，默认为 scene_001 到 scene_005
        scale_factor: 超分辨率倍数 (2 或 4)
        bayer_pattern: Bayer模式
    """
    # 默认处理 scene_001 到 scene_005
    if scenes is None:
        scenes = [f'scene_{i:03d}' for i in range(1, 6)]

    # 尺寸设置
    lr_size = (512, 612)  # (H, W)
    if scale_factor == 2:
        hr_size = (1024, 1224)
    elif scale_factor == 4:
        hr_size = (2048, 2448)
    else:
        raise ValueError(f"scale_factor must be 2 or 4")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'hr_gt'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'lr_gt'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'cpfa_raw'), exist_ok=True)

    print("=" * 60)
    print("CPFA Raw Image Generator")
    print("=" * 60)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Scale factor:     {scale_factor}×")
    print(f"HR size:          {hr_size[0]} × {hr_size[1]}")
    print(f"LR/CPFA size:     {lr_size[0]} × {lr_size[1]}")
    print(f"Bayer pattern:    {bayer_pattern}")
    print(f"Scenes to process: {scenes}")
    print("=" * 60)

    for scene_name in scenes:
        scene_path = os.path.join(input_dir, scene_name)

        if not os.path.isdir(scene_path):
            print(f"[SKIP] {scene_name}: 目录不存在")
            continue

        print(f"\n[Processing] {scene_name}")

        # 读取4个偏振方向的图像
        hr_0 = cv2.imread(os.path.join(scene_path, 'RGB_0.png'), -1)
        hr_45 = cv2.imread(os.path.join(scene_path, 'RGB_45.png'), -1)
        hr_90 = cv2.imread(os.path.join(scene_path, 'RGB_90.png'), -1)
        hr_135 = cv2.imread(os.path.join(scene_path, 'RGB_135.png'), -1)

        # 检查是否读取成功
        if any(img is None for img in [hr_0, hr_45, hr_90, hr_135]):
            missing = []
            if hr_0 is None: missing.append('RGB_0.png')
            if hr_45 is None: missing.append('RGB_45.png')
            if hr_90 is None: missing.append('RGB_90.png')
            if hr_135 is None: missing.append('RGB_135.png')
            print(f"  [ERROR] 缺少文件: {missing}")
            continue

        print(f"  原始图像尺寸: {hr_0.shape}")

        # 转换为float32
        if hr_0.dtype == np.uint16:
            hr_0 = hr_0.astype(np.float32) / 65535.0
            hr_45 = hr_45.astype(np.float32) / 65535.0
            hr_90 = hr_90.astype(np.float32) / 65535.0
            hr_135 = hr_135.astype(np.float32) / 65535.0
        elif hr_0.dtype == np.uint8:
            hr_0 = hr_0.astype(np.float32) / 255.0
            hr_45 = hr_45.astype(np.float32) / 255.0
            hr_90 = hr_90.astype(np.float32) / 255.0
            hr_135 = hr_135.astype(np.float32) / 255.0

        # 调整到HR尺寸
        hr_0 = cv2.resize(hr_0, (hr_size[1], hr_size[0]), interpolation=cv2.INTER_LINEAR)
        hr_45 = cv2.resize(hr_45, (hr_size[1], hr_size[0]), interpolation=cv2.INTER_LINEAR)
        hr_90 = cv2.resize(hr_90, (hr_size[1], hr_size[0]), interpolation=cv2.INTER_LINEAR)
        hr_135 = cv2.resize(hr_135, (hr_size[1], hr_size[0]), interpolation=cv2.INTER_LINEAR)

        # 下采样得到LR GT
        lr_0 = cv2.resize(hr_0, (lr_size[1], lr_size[0]), interpolation=cv2.INTER_LINEAR)
        lr_45 = cv2.resize(hr_45, (lr_size[1], lr_size[0]), interpolation=cv2.INTER_LINEAR)
        lr_90 = cv2.resize(hr_90, (lr_size[1], lr_size[0]), interpolation=cv2.INTER_LINEAR)
        lr_135 = cv2.resize(hr_135, (lr_size[1], lr_size[0]), interpolation=cv2.INTER_LINEAR)

        # 生成CPFA原始图像
        cpfa_raw = generate_cpfa_raw(lr_0, lr_45, lr_90, lr_135, bayer_pattern)

        # 保存结果
        # HR GT
        for angle, img in [('0', hr_0), ('45', hr_45), ('90', hr_90), ('135', hr_135)]:
            save_path = os.path.join(output_dir, 'hr_gt', f'{scene_name}_{angle}.png')
            cv2.imwrite(save_path, (img * 65535).astype(np.uint16))

        # LR GT
        for angle, img in [('0', lr_0), ('45', lr_45), ('90', lr_90), ('135', lr_135)]:
            save_path = os.path.join(output_dir, 'lr_gt', f'{scene_name}_{angle}.png')
            cv2.imwrite(save_path, (img * 65535).astype(np.uint16))

        # CPFA Raw
        cpfa_save_path = os.path.join(output_dir, 'cpfa_raw', f'{scene_name}.png')
        cv2.imwrite(cpfa_save_path, (cpfa_raw * 65535).astype(np.uint16))

        print(f"  HR GT:    {hr_0.shape} × 4 angles")
        print(f"  LR GT:    {lr_0.shape} × 4 angles")
        print(f"  CPFA Raw: {cpfa_raw.shape}")
        print(f"  已保存到: {output_dir}")

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"\n输出目录结构:")
    print(f"  {output_dir}/")
    print(f"  ├── hr_gt/        # HR GT (1024×1224×3) × 4 angles")
    print(f"  │   ├── scene_001_0.png")
    print(f"  │   ├── scene_001_45.png")
    print(f"  │   └── ...")
    print(f"  ├── lr_gt/        # LR GT (512×612×3) × 4 angles")
    print(f"  │   └── ...")
    print(f"  └── cpfa_raw/     # CPFA Raw (512×612) 单通道")
    print(f"      ├── scene_001.png")
    print(f"      └── ...")


if __name__ == '__main__':
    # ============================================================
    # 配置参数 - 请根据实际情况修改
    # ============================================================

    INPUT_DIR = './input_dir'      # 输入目录
    OUTPUT_DIR = './output_dir'    # 输出目录
    SCALE_FACTOR = 2               # 超分辨率倍数 (2 或 4)
    BAYER_PATTERN = 'RGGB'         # Bayer模式

    # 要处理的场景列表
    SCENES = ['scene_001', 'scene_002', 'scene_003', 'scene_004', 'scene_005']

    # ============================================================

    process_scenes(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        scenes=SCENES,
        scale_factor=SCALE_FACTOR,
        bayer_pattern=BAYER_PATTERN
    )
