import os
import numpy as np
import nibabel as nib
import pandas as pd
from evaluate.metrics.surface_distance import (
    compute_dice_coefficient,
    compute_surface_distances,
    compute_average_surface_distance,
    compute_robust_hausdorff
)
from openpyxl import load_workbook

def evaluate_case(gt_file, pred_file):
    gt_img = nib.load(gt_file)
    pred_img = nib.load(pred_file)
    spacing_mm = gt_img.header.get_zooms()[:3]
    gt_mask = gt_img.get_fdata() > 0.5
    pred_mask = pred_img.get_fdata() > 0.5

    surface_distances = compute_surface_distances(gt_mask, pred_mask, spacing_mm)

    dice = compute_dice_coefficient(gt_mask, pred_mask)
    asd_gt, asd_pred = compute_average_surface_distance(surface_distances)
    asd = (asd_gt + asd_pred) / 2
    hd95 = compute_robust_hausdorff(surface_distances, percent=95)

    return dice, asd, hd95

def process_folder(gt_root, pred_root, output_excel):
    assert os.path.exists(gt_root) and os.path.exists(pred_root)

    names = sorted(os.listdir(gt_root))
    excel_writer = None
    if os.path.exists(output_excel):
        excel_writer = pd.ExcelWriter(output_excel, engine='openpyxl', mode='a', if_sheet_exists='replace')
    else:
        excel_writer = pd.ExcelWriter(output_excel, engine='openpyxl')

    for name in names:
        gt_dir = os.path.join(gt_root, name)
        pred_dir = os.path.join(pred_root, name)
        if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
            continue

        metrics = {'Case': [], 'Dice': [], 'ASD': [], 'HD95': []}
        for file in sorted(os.listdir(gt_dir)):
            if not file.endswith(".nii") and not file.endswith(".nii.gz"):
                continue
            gt_file = os.path.join(gt_dir, file)
            pred_file = os.path.join(pred_dir, file)
            if not os.path.exists(pred_file):
                print(f"[Warning] Prediction file missing: {pred_file}")
                continue

            try:
                dice, asd, hd95 = evaluate_case(gt_file, pred_file)
                metrics['Case'].append(file)
                metrics['Dice'].append(round(dice, 4))
                metrics['ASD'].append(round(asd, 4))
                metrics['HD95'].append(round(hd95, 4))
            except Exception as e:
                print(f"[Error] Failed to evaluate {file}: {e}")

        df = pd.DataFrame(metrics)
        df.set_index('Case', inplace=True)
        df.to_excel(excel_writer, sheet_name=name)

    excel_writer.close()
    print(f"✅ Results saved to {output_excel}")

# 使用方式（替换路径为你实际的路径）：
if __name__ == '__main__':
    gt_root = '/your/path/to/gt'      # 包含 name 子文件夹的路径
    pred_root = '/your/path/to/pred'  # 包含 name 子文件夹的路径
    output_excel = 'metrics_results.xlsx'

    process_folder(gt_root, pred_root, output_excel)
