"""
多方法脑掩膜(mask)融合脚本
支持: SynthStrip / LifespanStrip / BEN / 其他任意二值mask方法的输出

融合策略:
    1. Majority Voting (多数投票) - 简单、快速、可解释
    2. STAPLE (Simultaneous Truth And Performance Level Estimation)
       - 会同时估计每个方法的"可靠性权重"，比单纯多数投票更稳健
    两种都会跑，同时输出，供你对比选择

同时会输出一份 QC 报告:
    - 各方法两两之间的 Dice 相似度 (帮你发现哪个方法系统性跑偏)
    - 每个被试所有方法的一致性得分 (低于阈值的会被标记，建议人工检查)

依赖:
    pip install SimpleITK nibabel numpy pandas --break-system-packages
    (如果在conda环境里，去掉 --break-system-packages)
"""

import os
import itertools
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd

# ============================== 配置区 ==============================

# 每个方法对应的mask输出目录，key是方法名(自己起，随便叫)，value是目录路径
METHOD_DIRS = {
    "hdbet":    "/path/to/your/data/gt/IXI_hdbet",
    "nnUnet": "/path/to/your/data/gt/IXI_nnUnet",
    "synthstrip":"/path/to/your/data/gt/IXI_SSTRIP",
    # 需要几种方法就加几行，key随便起名字
}

# 每个方法目录下，mask文件名相对"被试ID"的后缀规则，用于从文件名反推被试ID
# 例如 synthstrip 输出叫 002_S_0413_0000_mask.nii.gz，后缀是 "_0000_mask.nii.gz"
# 例如 lifespanstrip 输出叫 002_S_0413_0000_strip_mask.nii.gz，后缀不同
# 如果你的所有方法输出文件名的"主干"部分是一致的，只是后缀不同，在这里逐一声明
METHOD_SUFFIX = {
    "hdbet":    ".nii.gz",
    "nnUnet": ".nii.gz",
    "synthstrip":    ".nii.gz",
}

# 融合结果输出目录
OUTPUT_DIR = "/path/to/your/data/gt/IXI"

# STAPLE 融合参数
STAPLE_MAX_ITER = 100          # 最大迭代次数
STAPLE_CONVERGENCE = 1e-5      # 收敛阈值

# 多数投票时，需要超过多少方法数才判定为前景（默认: 半数以上）
# 例如 3种方法，MAJORITY_THRESHOLD=2 表示至少2个方法认为是脑组织才算前景
MAJORITY_THRESHOLD = None   # None = 自动取 ceil(方法数/2)

# 一致性QC: 如果某个被试所有方法两两Dice的平均值低于此阈值，标记为"建议人工检查"
QC_DICE_FLAG_THRESHOLD = 0.90

# 是否跳过已经存在的融合结果
SKIP_EXISTING = True

# ==================================================================


def find_common_subjects():
    """
    对每个方法目录，反推出 {被试ID: 完整文件路径} 的映射，
    然后取所有方法都存在mask的被试ID交集
    """
    per_method_files = {}
    for method, mdir in METHOD_DIRS.items():
        suffix = METHOD_SUFFIX[method]
        mapping = {}
        for fname in os.listdir(mdir):
            if fname.endswith(suffix):
                subject_id = fname[: -len(suffix)]
                mapping[subject_id] = os.path.join(mdir, fname)
        per_method_files[method] = mapping
        print(f"[{method}] 发现 {len(mapping)} 个mask文件")

    # 取交集：只融合所有方法都提供了mask的被试
    common_ids = set.intersection(*[set(m.keys()) for m in per_method_files.values()])
    print(f"\n所有方法共有的被试数: {len(common_ids)}")

    missing_report = {}
    all_ids = set.union(*[set(m.keys()) for m in per_method_files.values()])
    for sid in sorted(all_ids - common_ids):
        missing_from = [m for m in METHOD_DIRS if sid not in per_method_files[m]]
        missing_report[sid] = missing_from

    return per_method_files, sorted(common_ids), missing_report


def load_mask_as_array(path):
    """加载mask，返回 (numpy array, nibabel affine/header) 用于最终保存"""
    img = nib.load(path)
    data = (img.get_fdata() > 0.5).astype(np.uint8)
    return data, img.affine, img.header


def dice_score(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def majority_vote_fuse(mask_arrays, threshold):
    stacked = np.stack(mask_arrays, axis=0)  # (n_methods, H, W, D)
    vote_sum = stacked.sum(axis=0)
    fused = (vote_sum >= threshold).astype(np.uint8)
    return fused


def staple_fuse(mask_arrays):
    """
    用 SimpleITK 的 STAPLE 算法融合多个二值mask
    返回: (二值fused mask, 概率图)
    """
    sitk_images = [
        sitk.GetImageFromArray(arr.astype(np.int16)) for arr in mask_arrays
    ]
    staple_filter = sitk.STAPLEImageFilter()
    staple_filter.SetMaximumIterations(STAPLE_MAX_ITER)
    staple_filter.SetConfidenceWeight(1.0)

    prob_image = staple_filter.Execute(sitk_images)  # 输出为前景概率图 (float)
    prob_array = sitk.GetArrayFromImage(prob_image)

    fused = (prob_array > 0.5).astype(np.uint8)
    return fused, prob_array


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    per_method_files, common_ids, missing_report = find_common_subjects()

    if missing_report:
        print(f"\n以下 {len(missing_report)} 个被试至少缺少一种方法的mask，将被跳过:")
        for sid, missing_from in list(missing_report.items())[:10]:
            print(f"  {sid}: 缺少 {missing_from}")
        if len(missing_report) > 10:
            print(f"  ... 还有 {len(missing_report) - 10} 个，详见输出的 missing_subjects.csv")

        pd.DataFrame(
            [(sid, ",".join(mf)) for sid, mf in missing_report.items()],
            columns=["subject_id", "missing_from_methods"],
        ).to_csv(os.path.join(OUTPUT_DIR, "missing_subjects.csv"), index=False)

    n_methods = len(METHOD_DIRS)
    majority_threshold = MAJORITY_THRESHOLD or int(np.ceil(n_methods / 2))
    print(f"\n多数投票阈值: 至少 {majority_threshold}/{n_methods} 个方法判定为前景")

    method_names = list(METHOD_DIRS.keys())
    pairwise_dice_records = []
    qc_records = []

    for idx, sid in enumerate(common_ids, 1):
        fused_majority_path = os.path.join(OUTPUT_DIR, f"{sid}_fused_majority.nii.gz")
        fused_staple_path = os.path.join(OUTPUT_DIR, f"{sid}_fused_staple.nii.gz")
        prob_path = os.path.join(OUTPUT_DIR, f"{sid}_staple_prob.nii.gz")

        if SKIP_EXISTING and os.path.exists(fused_majority_path) and os.path.exists(fused_staple_path):
            print(f"[{idx}/{len(common_ids)}] 已存在，跳过: {sid}")
            continue

        print(f"[{idx}/{len(common_ids)}] 融合中: {sid}")

        mask_arrays = {}
        affine, header = None, None
        for method in method_names:
            arr, aff, hdr = load_mask_as_array(per_method_files[method][sid])
            mask_arrays[method] = arr
            if affine is None:
                affine, header = aff, hdr

        # ---- 两两 Dice，用于QC ----
        for m1, m2 in itertools.combinations(method_names, 2):
            d = dice_score(mask_arrays[m1], mask_arrays[m2])
            pairwise_dice_records.append({"subject_id": sid, "method_pair": f"{m1}_vs_{m2}", "dice": d})

        avg_pairwise_dice = np.mean(
            [r["dice"] for r in pairwise_dice_records if r["subject_id"] == sid]
        )
        flagged = avg_pairwise_dice < QC_DICE_FLAG_THRESHOLD
        qc_records.append({
            "subject_id": sid,
            "avg_pairwise_dice": avg_pairwise_dice,
            "flagged_for_review": flagged,
        })

        # ---- 多数投票融合 ----
        majority_fused = majority_vote_fuse(list(mask_arrays.values()), majority_threshold)
        nib.save(nib.Nifti1Image(majority_fused, affine, header), fused_majority_path)

        # ---- STAPLE 融合 ----
        staple_fused, staple_prob = staple_fuse(list(mask_arrays.values()))
        nib.save(nib.Nifti1Image(staple_fused, affine, header), fused_staple_path)
        nib.save(nib.Nifti1Image(staple_prob.astype(np.float32), affine, header), prob_path)

    # ---- 保存QC报告 ----
    pd.DataFrame(pairwise_dice_records).to_csv(
        os.path.join(OUTPUT_DIR, "pairwise_dice_report.csv"), index=False
    )
    qc_df = pd.DataFrame(qc_records).sort_values("avg_pairwise_dice")
    qc_df.to_csv(os.path.join(OUTPUT_DIR, "qc_report.csv"), index=False)

    n_flagged = qc_df["flagged_for_review"].sum()
    print("\n" + "=" * 50)
    print(f"融合完成。共处理 {len(common_ids)} 个被试。")
    print(f"其中 {n_flagged} 个被试平均两两Dice低于 {QC_DICE_FLAG_THRESHOLD}，建议人工检查。")
    print(f"详见: {os.path.join(OUTPUT_DIR, 'qc_report.csv')}")


if __name__ == "__main__":
    main()