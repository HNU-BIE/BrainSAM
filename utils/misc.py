from utils.nifti_reader import NIFTI_READER as nii_reader
import os
import numpy as np
from PIL import Image
from evaluate.metrics.calculate_metric_on_folder import evaluate_case
from evaluate.metrics.calculate_metric_on_folder import compute_dice_coefficient,\
    compute_surface_distances, compute_robust_hausdorff
import cv2

def save_slice_as_png(nii_path, axis, output_dir):
    brain = nii_reader(nii_path)
    num_frames = brain.get_slices_num(axis)
    os.makedirs(output_dir, exist_ok=True)
    if len(os.listdir(output_dir)) < num_frames:
        for sli in range(0, num_frames):
            slice,_ = brain.get_slice_array(axis, sli)
            slice_img_normalized = np.interp(slice, (slice.min(), slice.max()), (0, 255)).astype(np.uint8)
            slice_pil_img = Image.fromarray(slice_img_normalized)
            slice_pil_img.save(f"{output_dir}/{sli}.png")
    return output_dir

def calculate_metrics_single_case(pred_mask, gt_mask, image_type='nii'):
    """
    计算单个 case 的指标
    :param pred_mask: nifti 文件路径 或 PNG 文件夹路径
    :param gt_mask: nifti 文件路径 或 PNG 文件夹路径
    :param image_type: 'nii' 或 'png'
    :return: dict
    """
    try:
        if image_type == "nii":
            dice, asd, hd95 = evaluate_case(pred_mask, gt_mask)
            return {"dice": round(dice, 4), "asd": round(asd, 4), "hd95": round(hd95, 4)}

        elif image_type == "png":
            gt_files = [f for f in os.listdir(gt_mask) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            gt_map = {}
            for f in gt_files:
                num = int(os.path.splitext(f)[0])  # "001.png" -> 1, "1.jpg" -> 1
                gt_map[num] = f
            
            
            pred_files = sorted([f for f in os.listdir(pred_mask) if f.endswith(".png")])

            pred_slices = []
            gt_slices = []
            dice_per_slice = []
            hd95_per_slice = []

            for f in pred_files:
                pred_path = os.path.join(pred_mask, f)
                num = int(os.path.splitext(f)[0])  # "1.png" -> 1
                if num not in gt_map:
                    print(f"[Warning] GT missing for frame {num}")
                    continue
                gt_path = os.path.join(gt_mask, gt_map[num])
                
                pred_2d = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) > 0
                gt_2d = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0

                pred_slices.append(pred_2d)
                gt_slices.append(gt_2d)

                # 逐切片指标（跳过空切片）
                if np.sum(pred_2d) == 0 and np.sum(gt_2d) == 0:
                    continue
                elif np.sum(pred_2d) == 0 or np.sum(gt_2d) == 0:
                    dice_per_slice.append(0.0)
                    hd95_per_slice.append(float('inf'))
                    continue

                # 扩维成 (1, H, W) 复用 3D 函数
                pred_3d = pred_2d[np.newaxis, :, :]
                gt_3d = gt_2d[np.newaxis, :, :]

                dice_per_slice.append(float(compute_dice_coefficient(gt_3d, pred_3d)))
                sd = compute_surface_distances(gt_3d, pred_3d, spacing_mm=(1.0, 1.0, 1.0))
                hd95_per_slice.append(float(compute_robust_hausdorff(sd, percent=95)))

            # 整体 3D 指标（堆叠所有切片）
            pred_vol = np.stack(pred_slices, axis=0).astype(bool)
            gt_vol = np.stack(gt_slices, axis=0).astype(bool)

            dice_3d = float(compute_dice_coefficient(gt_vol, pred_vol))
            sd_3d = compute_surface_distances(gt_vol, pred_vol, spacing_mm=(1.0, 1.0, 1.0))
            hd95_3d = float(compute_robust_hausdorff(sd_3d, percent=95))

            return {
                "dice_3d": round(dice_3d, 4),
                "hd95_3d": round(hd95_3d, 4),
                "dice_slice_mean": round(float(np.mean(dice_per_slice)), 4) if dice_per_slice else 0.0,
                "dice_slice_std": round(float(np.std(dice_per_slice)), 4) if dice_per_slice else 0.0,
                "hd95_slice_mean": round(float(np.mean(hd95_per_slice)), 4) if hd95_per_slice else float('inf'),
                "hd95_slice_std": round(float(np.std(hd95_per_slice)), 4) if hd95_per_slice else 0.0,
            }

        else:
            raise ValueError("Unsupported image type. Use 'nii' or 'png'.")

    except Exception as e:
        print(f"Error in calculate_metrics_single_case: {e}")
        return {"dice_3d": 0.0, "hd95_3d": float('inf')}

def get_prompts_points(gt_mask, frame_idx, imgae_type='nii', axis=0, num=1):
    if imgae_type == "nii":
        brain = nii_reader(gt_mask)
        slice_2d = brain.get_slice_array(axis, frame_idx)[0]
        mask = (slice_2d > 0).astype(np.uint8)

    elif imgae_type == "png":
        frame_names = [
            p for p in os.listdir(gt_mask)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        mask = cv2.imread(os.path.join(gt_mask, frame_names[frame_idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

    else:
        raise ValueError("Unsupported image type. Please use 'nii' or 'png'.")

    # 腐蚀 mask，去掉边缘区域，只保留内部
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eroded = cv2.erode(mask, kernel, iterations=1)

    # 如果腐蚀后为空（区域太小），逐步缩小腐蚀力度
    for k_size in [11, 7, 3]:
        if np.sum(eroded) > 0:
            break
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        eroded = cv2.erode(mask, kernel, iterations=1)

    # 如果所有腐蚀都为空，退回到原始 mask
    if np.sum(eroded) == 0:
        eroded = mask

    points = np.argwhere(eroded > 0)[:, ::-1].tolist()  # (row,col) -> (x,y)

    if num < len(points):
        indices = np.random.choice(len(points), num, replace=False)
        points = [points[i] for i in indices]

    labels = [1] * len(points)

    ptsList = [points]
    ptsTypeList = [labels]

    return ptsList, ptsTypeList

def get_bbox_from_mask(mask_2d, margin=10):
    H, W = mask_2d.shape
    yx = np.argwhere(mask_2d > 0)
    if len(yx) == 0:
        return [margin, margin, W - margin, H - margin]
    y0, x0 = yx.min(axis=0)
    y1, x1 = yx.max(axis=0)
    return [
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(W, x1 + margin),
        min(H, y1 + margin),
    ]

def get_bbox_from_gt(gt_nii_path, frame_idx, orient=2, margin=10):
    gt_reader = nii_reader(gt_nii_path)
    slice_2d, _ = gt_reader.get_slice_array(orient, frame_idx)
    mask = (slice_2d > 0).astype(np.uint8)
    return get_bbox_from_mask(mask, margin)

def get_bbox_auto(img_slice, margin=10):
    return get_bbox_from_mask((img_slice > 5).astype(np.uint8), margin)