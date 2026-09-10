"""
批量推理 + 指标计算脚本
遍历所有 case 并保存结果到 CSV。

用法:
    python batch_eval.py \
        --raw_root /path/to/raw \
        --gt_root /path/to/gt \
        --save_root /path/to/output \
        --config_file /path/to/config.yaml \
        --ckpt_path /path/to/checkpoint.pt \
        --output_csv results.csv \
        --num_points 3 \
        --frame_idx 66 \
        --device cuda:0

目录结构要求:
    raw_root/
        case_name_1/    # 包含 .jpg/.png 图片序列
        case_name_2/
        ...
    gt_root/
        case_name_1/    # 包含对应的 GT mask 图片
        case_name_2/
        ...
"""

import os
import sys
import argparse
import csv
import time
import numpy as np
from datetime import datetime


PROJECT_ROOT = "/path/to/your/BrainSAM"  # TODO: point this at your local BrainSAM project root
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from inference.base_inference import *
from sam2.build_sam import build_sam2_video_predictor
from utils.misc import get_prompts_points, calculate_metrics_single_case


def build_predictor(config_file, ckpt_path, device="cuda:0"):
    """构建 predictor，优先尝试 brainsam，失败则用 sam2"""
    # try:
    #     from sam2.build_sam import build_brainsam_predictor
    #     predictor = build_brainsam_predictor(
    #         config_file=config_file,
    #         ckpt_path=ckpt_path,
    #         device=device
    #     )
    #     print(f"[INFO] Loaded BrainSAM predictor from {ckpt_path}")
    # except Exception as e:
    #     print(f"[WARN] Failed to load BrainSAM predictor: {e}")
    #     print("[INFO] Falling back to SAM2 video predictor...")
    predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=ckpt_path,
        device=device
    )
    return predictor


def process_single_case(predictor, raw_dir, gt_dir, save_dir,
                        frame_idx, num_points, modal_id, organ_ids):
    """
    单个 case 的推理 + 指标计算
    返回指标 dict，失败返回 None
    """
    # 1. 从 GT 采样 prompt 点
    points, labels = get_prompts_points(
        gt_dir, frame_idx, num=num_points, imgae_type='png'
    )

    # 2. 推理
    os.makedirs(save_dir, exist_ok=True)
    results = seg_folder_inference(
        predictor, raw_dir, points, labels,
        nii_idx=frame_idx,
        save_dir=save_dir,
        modal_id=modal_id,
        organ_ids=organ_ids
    )

    # 3. 计算指标
    metrics = calculate_metrics_single_case(save_dir, gt_dir, image_type='png')

    return metrics


def main():
    parser = argparse.ArgumentParser(description="批量推理与指标计算")
    parser.add_argument("--raw_root", type=str, required=False,
                        default="/path/to/your/data/raw/For_brainSAM_training",
                        help="原始图片根目录，下面每个子文件夹是一个 case")
    parser.add_argument("--gt_root", type=str, required=False,
                        default="/path/to/your/data/gt/For_brainSAM_training_gt",
                        help="GT mask 根目录，子文件夹名与 raw_root 对应")
    parser.add_argument("--save_root", type=str, required=False,
                        default="/path/to/your/data/save",
                        help="推理结果保存根目录")
    parser.add_argument("--config_file", type=str, required=False,
                        default="configs/sam2.1/sam2.1_hiera_b+_512.yaml",
                        help="模型 config yaml 路径")
    parser.add_argument("--ckpt_path", type=str, required=False,
                        # default="/path/to/your/BrainSAM/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+BrainSAM.yaml/checkpoints/checkpoint.pt",
                        default="/path/to/your/BrainSAM/work_dir/sam2.1_hiera_base_plus.pt",
                        help="模型 checkpoint 路径")
    parser.add_argument("--output_csv", type=str, default="/path/to/your/data/results_origin.csv",
                        help="输出 CSV 文件路径")
    parser.add_argument("--num_points", type=int, default=3,
                        help="每个 case 的 prompt 点数")
    parser.add_argument("--frame_idx", type=int, default=66,
                        help="用于采样 prompt 的帧索引")
    parser.add_argument("--modal_id", type=int, default=0,
                        help="模态 ID")
    parser.add_argument("--organ_ids", type=int, nargs='+', default=[1, 2, 2],
                        help="器官 ID 列表")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备")
    args = parser.parse_args()

    # 构建 predictor（只加载一次）
    print("=" * 60)
    print("Loading model...")
    predictor = build_predictor(args.config_file, args.ckpt_path, args.device)
    print("=" * 60)

    # 获取所有 case
    case_names = sorted([
        d for d in os.listdir(args.raw_root)
        if os.path.isdir(os.path.join(args.raw_root, d))
    ])
    print(f"Found {len(case_names)} cases to process.")

    fieldnames = ["case", "dice_3d", "hd95_3d", "dice_slice_mean", "dice_slice_std",
              "hd95_slice_mean", "hd95_slice_std", "error"]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # 批量处理
    all_results = []
    for i, case_name in enumerate(case_names):
        raw_dir = os.path.join(args.raw_root, case_name)
        gt_dir = os.path.join(args.gt_root, case_name)
        save_dir = os.path.join(args.save_root, case_name)

        # 检查 GT 是否存在
        if not os.path.isdir(gt_dir):
            print(f"[{i+1}/{len(case_names)}] SKIP {case_name} - GT not found")
            continue

        print(f"[{i+1}/{len(case_names)}] Processing {case_name}...", end=" ")
        t0 = time.time()

        try:
            metrics = process_single_case(
                predictor, raw_dir, gt_dir, save_dir,
                frame_idx=args.frame_idx,
                num_points=args.num_points,
                modal_id=args.modal_id,
                organ_ids=args.organ_ids
            )

            if metrics is None:
                print("FAILED")
                continue

            metrics["case"] = case_name
            all_results.append(metrics)

            elapsed = time.time() - t0
            dice_key = "dice_3d" if "dice_3d" in metrics else "dice"
            hd95_key = "hd95_3d" if "hd95_3d" in metrics else "hd95"
            print(f"Dice={metrics.get(dice_key, 'N/A'):.4f}  "
                  f"HD95={metrics.get(hd95_key, 'N/A'):.4f}  "
                  f"({elapsed:.1f}s)")

        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({"case": case_name, "error": str(e)})

        with open(args.output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(metrics)

        # 立即打印结果
        dice = metrics.get("dice_3d", "N/A")
        hd95 = metrics.get("hd95_3d", "N/A")
        print(f"[{i+1}/{len(case_names)}] {case_name}  "
            f"Dice={dice}  HD95={hd95}")


if __name__ == "__main__":
    main()