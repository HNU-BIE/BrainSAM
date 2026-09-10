"""
批量提取mask的最大连通域(Largest Connected Component)
用途: 融合后的多数投票/STAPLE mask常有孤立小碎块(血管残留、噪声点等)，
      本脚本对每个mask只保留体积最大的一个连通域，可选做一次闭运算填洞。

依赖:
    pip install nibabel numpy scipy --break-system-packages
"""

import os
import numpy as np
import nibabel as nib
from scipy import ndimage

# ============================== 配置区 ==============================

# 输入mask所在目录（可以是 fused_majority / fused_staple 的输出目录）
INPUT_DIR = "/path/to/your/data/gt/ADNI"

# 只处理该目录下文件名匹配此后缀的文件（避免把prob图也当mask处理）
INPUT_SUFFIX = "_fused_majority.nii.gz"

# 输出目录（保留最大连通域后的结果）
OUTPUT_DIR = "/path/to/your/data/gt/ADNI_lcc"

# 输出文件名后缀（会替换掉 INPUT_SUFFIX 后加上此后缀）
OUTPUT_SUFFIX = "_lcc.nii.gz"

# 连通性定义: 2 = 6-连通(更严格，只认面相邻)，3 = 26-连通(含棱/角相邻，更宽松)
# 脑部mask一般建议用 26-连通，避免因为部分层面轻微断裂被切成两块
CONNECTIVITY = 3

# 是否在提取最大连通域之前，先做一次形态学闭运算填补mask内部小空洞
DO_CLOSING_BEFORE = True
CLOSING_ITER = 2  # 闭运算迭代次数(体素单位)

# 是否在提取最大连通域之后，再做一次二值填洞(把最大连通域内部的洞填死)
FILL_HOLES_AFTER = True

# 是否跳过已存在的输出文件
SKIP_EXISTING = True

# ==================================================================


def keep_largest_component(mask, connectivity=3):
    """
    输入: 二值 numpy array
    输出: 只保留最大连通域后的二值 numpy array
    """
    structure = ndimage.generate_binary_structure(3, connectivity if connectivity <= 3 else 3)
    # generate_binary_structure 第二个参数取值范围是1~3(对应6/18/26连通)，这里直接映射
    conn_map = {2: 1, 3: 3}  # 2->6-连通(rank=1), 3->26-连通(rank=3)
    rank = conn_map.get(connectivity, 3)
    structure = ndimage.generate_binary_structure(3, rank)

    labeled, num_features = ndimage.label(mask, structure=structure)
    if num_features == 0:
        return mask  # 全空，直接返回

    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label).astype(np.uint8)


def process_one(in_path, out_path):
    img = nib.load(in_path)
    data = (img.get_fdata() > 0.5).astype(np.uint8)

    if DO_CLOSING_BEFORE:
        data = ndimage.binary_closing(
            data, structure=np.ones((3, 3, 3)), iterations=CLOSING_ITER
        ).astype(np.uint8)

    lcc = keep_largest_component(data, connectivity=CONNECTIVITY)

    if FILL_HOLES_AFTER:
        lcc = ndimage.binary_fill_holes(lcc).astype(np.uint8)

    nib.save(nib.Nifti1Image(lcc, img.affine, img.header), out_path)

    orig_voxels = int(data.sum())
    kept_voxels = int(lcc.sum())
    removed = orig_voxels - kept_voxels
    return orig_voxels, kept_voxels, removed


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fnames = [f for f in os.listdir(INPUT_DIR) if f.endswith(INPUT_SUFFIX)]
    print(f"发现 {len(fnames)} 个待处理mask (匹配后缀: {INPUT_SUFFIX})")

    summary = []
    for idx, fname in enumerate(sorted(fnames), 1):
        subject_id = fname[: -len(INPUT_SUFFIX)]
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, f"{subject_id}{OUTPUT_SUFFIX}")

        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"[{idx}/{len(fnames)}] 已存在，跳过: {subject_id}")
            continue

        print(f"[{idx}/{len(fnames)}] 处理中: {subject_id}")
        orig, kept, removed = process_one(in_path, out_path)
        pct_removed = 100.0 * removed / orig if orig > 0 else 0.0
        summary.append((subject_id, orig, kept, removed, pct_removed))

        if pct_removed > 5.0:
            print(f"    ⚠ 移除体素占比较高: {pct_removed:.2f}% (原{orig} -> 保留{kept})，建议检查原mask是否严重碎裂")

    print("\n" + "=" * 50)
    print(f"处理完成，共 {len(summary)} 个被试被更新。")
    if summary:
        avg_pct = np.mean([s[4] for s in summary])
        print(f"平均移除体素占比: {avg_pct:.3f}%")
        flagged = [s for s in summary if s[4] > 5.0]
        if flagged:
            print(f"以下 {len(flagged)} 个被试移除比例超过5%，建议人工检查:")
            for sid, orig, kept, removed, pct in flagged:
                print(f"  {sid}: 原{orig} -> 保留{kept} (移除{pct:.2f}%)")


if __name__ == "__main__":
    main()