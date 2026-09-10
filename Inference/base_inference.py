# %%
from sam2.build_sam import build_brainsam_predictor
import os
import time
import tempfile
import cv2
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage import measure
from utils.nifti_reader import NIFTI_READER as nii_reader
from utils.misc import get_bbox_from_gt

#===========================================================
# 帧写入/读取相关的可调参数（都在这里改，不用传参数）
#===========================================================
FRAME_READ_MAX_RETRIES = 3
FRAME_READ_RETRY_DELAY_SEC = 1.5

# prompt 标注图（画了 bbox/points 的那一张切片图）保存在 {save_dir}/log/ 下
SAVE_PROMPT_LOG_IMAGE = True
#===========================================================


def _frame_path(frame_dir, sli):
    return f"{frame_dir}/{sli}.png"


def _is_valid_frame_file(path):
    """校验一帧 png 是否完整可读：文件存在、非空、PIL 能正常打开并 load 像素数据。
    只 Image.open() 不够，PIL 有些截断文件 open() 不报错，load() 才会报错。"""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        return False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def _write_frame(brain, axis, sli, frame_dir):
    slice_, _ = brain.get_slice_array(axis, sli)
    slice_img_normalized = np.interp(slice_, (slice_.min(), slice_.max()), (0, 255)).astype(np.uint8)
    slice_pil_img = Image.fromarray(slice_img_normalized)
    path = _frame_path(frame_dir, sli)
    slice_pil_img.save(path)


def _write_all_frames(brain, axis, num_frames, frame_dir):
    """把 0..num_frames-1 每一帧都写进 frame_dir。frame_dir 现在总是一个全新的
    临时目录（调用方用 tempfile.TemporaryDirectory() 创建），不存在"之前跑过就跳过"
    这回事了，每次都是干净地从头写，天然不会有残留的半成品/损坏帧问题。"""
    for sli in range(num_frames):
        _write_frame(brain, axis, sli, frame_dir)


def create_inference_state(predictor, video_dir, frame_dir=None, prog_signal=None, axis=None,
                           modal_id=0, organ_ids=[0,0,0]):
    """
    video_dir: 原始输入，可以是 .nii/.nii.gz 文件路径，也可以是一个已经存好帧图片的文件夹。
    frame_dir: 实际写入/读取帧图片的目录，只有 video_dir 是 .nii/.nii.gz 时才需要。
               调用方应该传入一个 tempfile.TemporaryDirectory() 给的临时目录路径，
               用完由调用方负责清理（with 块退出时自动清理），这个函数本身不创建、
               也不清理任何持久化的 temp 文件夹。
    """
    if video_dir.endswith(('.nii', '.nii.gz')):
        if frame_dir is None:
            raise ValueError(
                "video_dir 是 .nii/.nii.gz 文件时必须传入 frame_dir"
                "（建议用 tempfile.TemporaryDirectory() 生成，参见 _run_single_axis）。"
            )
        rawFilePath = video_dir
        brain = nii_reader(rawFilePath)
        num_frames = brain.get_slices_num(axis)

        _write_all_frames(brain, axis, num_frames, frame_dir)

        inference_state = _init_state_with_retry(
            predictor, frame_dir, prog_signal, modal_id, organ_ids,
            # 读失败时怎么重新生成坏帧：交给这个回调，只重写读失败的那一帧
            rewrite_frame_fn=lambda sli: _write_frame(brain, axis, sli, frame_dir),
        )
        return inference_state, brain
    
    elif os.path.isdir(video_dir):
        # 这个分支输入本身就是一个持久化的图片文件夹（不是我们生成的临时帧），
        # 不涉及临时目录创建/清理，直接读。
        try:
            frame_names = [
                p for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            inference_state = _init_state_with_retry(
                predictor, video_dir, prog_signal, modal_id, organ_ids,
                rewrite_frame_fn=None,  # 这个分支没有原始 nii 数据源，坏帧没法重写，只能重试读取
            )
            return inference_state, None
        
        except Exception as e:
            raise RuntimeError(f"Failed to create inference state from directory: {video_dir}. Error: {e}")
    else:
        raise ValueError("input a unknow type, you can wait for the author to update.")


def _init_state_with_retry(predictor, video_dir, prog_signal, modal_id, organ_ids, rewrite_frame_fn):
    """
    调用 predictor.init_state()，把两类失败原因分开处理：

    1. 参数不兼容（这个 predictor 版本不接受 modal_id/organ_ids）——
       这是原来 except 兜底真正想解决的情况，保留同样的降级重试。
    2. 帧文件读取失败（cannot identify image file / 类似 I/O 错误）——
       这跟 modal_id/organ_ids 毫无关系，之前的代码会用"去掉 modal_id/organ_ids
       重试"这个无关操作去撞运气。现在改成：等一下、如果知道是哪一帧坏了就把那一帧
       重新从原始数据写一遍，再重试读取；重试仍然带着 modal_id/organ_ids，
       不会因为一次读取失败就悄悄丢掉这两个参数。
    """
    last_err = None
    for attempt in range(FRAME_READ_MAX_RETRIES + 1):
        try:
            return predictor.init_state(
                video_path=video_dir,
                prog_signal=prog_signal,
                modal_id=modal_id,
                organ_ids=organ_ids,
            )
        except TypeError as e:
            # 典型的"这个 predictor 不认识 modal_id/organ_ids 这两个关键字参数"的错误类型
            print(f"  [FALLBACK] {video_dir}: predictor.init_state 不支持 modal_id/organ_ids"
                  f"（{e}），改用不带这两个参数的调用")
            return predictor.init_state(video_path=video_dir, prog_signal=prog_signal)
        except Exception as e:
            last_err = e
            msg = str(e)
            bad_frame_idx = _extract_frame_index_from_error(msg)
            print(f"  [READ-RETRY {attempt+1}/{FRAME_READ_MAX_RETRIES}] {video_dir}: "
                  f"init_state 失败: {e}")
            if bad_frame_idx is not None and rewrite_frame_fn is not None:
                bad_path = _frame_path(video_dir, bad_frame_idx)
                print(f"    -> 定位到坏帧 {bad_path}，重新写入后再试")
                try:
                    rewrite_frame_fn(bad_frame_idx)
                except Exception as rewrite_err:
                    print(f"    -> 重写坏帧也失败了: {rewrite_err}")
            if attempt < FRAME_READ_MAX_RETRIES:
                time.sleep(FRAME_READ_RETRY_DELAY_SEC)
    # 重试用完了，如实抛出最后一次的错误，不再悄悄换一套参数掩盖问题
    raise last_err


def _extract_frame_index_from_error(msg: str):
    """从 'cannot identify image file .../123.png' 这类报错里把帧号抠出来，
    抠不出来就返回 None（比如目录级别的错误，不针对某一帧）。"""
    import re
    m = re.search(r"/(\d+)\.(?:png|jpg|jpeg)", msg, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


_AXIS_NAME = {0: "axi", 1: "cor", 2: "sag"}


def _save_prompt_log_image(brain, axis, nii_idx, video_dir, save_dir,
                            prompt_type, box=None, points=None, labels=None):
    """把交互切片（画了 prompt 的那一帧）单独存一张图到 {save_dir}/log/，
    图上标出 prompt 类型、坐标、slice idx，方便事后检查 prompt 打得对不对。
    只存这一张，不存整段视频的每一帧。"""
    if not SAVE_PROMPT_LOG_IMAGE:
        return

    log_dir = os.path.join(save_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    slice_, _ = brain.get_slice_array(axis, nii_idx)
    slice_norm = np.interp(slice_, (slice_.min(), slice_.max()), (0, 255)).astype(np.uint8)
    img_bgr = cv2.cvtColor(np.stack([slice_norm] * 3, axis=-1), cv2.COLOR_RGB2BGR)

    coord_text = ""
    if box:
        x0, y0, x1, y1 = [int(v) for v in box]
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)
        coord_text = f"box=({x0},{y0},{x1},{y1})"
    elif points is not None and len(points) > 0:
        labels_use = labels if labels is not None else [1] * len(points)
        for (x, y), lb in zip(points, labels_use):
            color = (0, 255, 0) if int(lb) == 1 else (0, 0, 255)  # 绿=正样本 红=负样本
            cv2.circle(img_bgr, (int(x), int(y)), 4, color, -1)
        pts_str = ",".join(f"({int(x)},{int(y)},{int(lb)})" for (x, y), lb in zip(points, labels_use))
        coord_text = f"pts={pts_str}"

    axis_name = _AXIS_NAME.get(axis, str(axis))
    label_lines = [
        f"prompt_type={prompt_type}",
        f"axis={axis_name}({axis})  slice_idx={nii_idx}",
        coord_text,
    ]
    y_text = 20
    for line in label_lines:
        if not line:
            continue
        cv2.putText(img_bgr, line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        y_text += 20

    basename = os.path.basename(video_dir)
    for ext in (".nii.gz", ".nii"):
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    out_name = f"{basename}_{axis_name}_slice{nii_idx:04d}.png"
    out_path = os.path.join(log_dir, out_name)
    cv2.imwrite(out_path, img_bgr)
    print(f"  [LOG] prompt 标注图 -> {out_path}")


@torch.no_grad()
def _Continuous_inference(predictor,
                          inference_state,
                          ptsList=None,
                          ptsTypeList=None,
                          box=None,
                          result=False,
                          nii_idx=59,
                          start_frame=0,
                          max_frame_num_to_track=None,
                          reverse=False,
                          prog_signal=None,
                        ) :

    """
    a generator function that continuously yields segmentation results for each frame in the video, starting from the frame we interact with (nii_idx).
    :param predictor: a predictor build by build_sam2_video_predictor.
    :param ptsList: 点prompt，[obj1_pts, obj2_pts, ...]，每个 objN_pts 是 shape (N,2) 的坐标数组。
    :param ptsTypeList: 对应每个物体每个点的标签，[obj1_labels, obj2_labels, ...]，
                         `1` 表示正样本点，`0` 表示负样本点。
    :param box: bbox prompt，单个物体的 [x0, y0, x1, y1]（不是"每个坐标算一个物体"）。
    :param video_dir: the target folder which contains a series of images(*.jpg, *.jpeg, *.png).
    :return: return a dictionary,like {frame0:{obj_id:mask},frame1:{obj_id:mask}...}

    points 和 box 是二选一的两种 prompt 方式，不要同时塞值进来：
    只要 ptsList 非空就走点 prompt；否则如果 box 非空就走 bbox prompt。
    """
    ann_frame_idx = nii_idx  # the frame index we interact with
    print("ann_frame_idx:", ann_frame_idx)

    use_points = ptsList is not None and len(ptsList) > 0
    use_box = (not use_points) and box is not None and len(box) > 0

    if not use_points and not use_box:
        raise ValueError("ptsList 和 box 都是空的，_Continuous_inference 至少需要一种 prompt。")

    # 物体数：点 prompt 是"传进来几个物体的点集就是几个物体"；
    # bbox prompt 一个 [x0,y0,x1,y1] 就是一个物体，不能拿坐标个数当物体数。
    n_objs = len(ptsList) if use_points else 1
    ann_obj_id = list(range(1, n_objs + 1))

    out_mask_logits = None
    for idx, obj_id in enumerate(ann_obj_id):
        if use_points:
            pts_i = np.asarray(ptsList[idx], dtype=np.float32)
            lbs_i = np.asarray(ptsTypeList[idx], dtype=np.int32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                points=pts_i,
                labels=lbs_i,
                box=None,
            )
            
        else:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                points=None,
                labels=None,
                box=box,
            )
            
    yield (out_mask_logits > 0.0).cpu().numpy()
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    # if max_frame_num_to_track==None:
    #     max_frame_num_to_track=inference_state["num_frames"]
        # print(f"max_frame_num_to_track set to num_frames:{max_frame_num_to_track}")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=ann_frame_idx,
            # max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
            prog_signal=prog_signal):
        yield out_frame_idx, out_obj_ids, out_mask_logits
    
    # 反向：nii_idx -> 第一帧
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=ann_frame_idx,
            # max_frame_num_to_track=max_frame_num_to_track,
            reverse=True,
            prog_signal=prog_signal):
        yield out_frame_idx, out_obj_ids, out_mask_logits
    
def seg_folder_inference(predictor, video_dir, ptsList, ptsTypeList, nii_idx=59, save_dir=None,modal_id=5, organ_ids=[1,2,15]):
    video_segments = {}

    inference, _ = create_inference_state(predictor, video_dir, prog_signal=None, axis=None,modal_id=5, organ_ids=[1,2,15])
    gen = _Continuous_inference(predictor=predictor,
                                inference_state=inference,
                                ptsList=ptsList,
                                ptsTypeList=ptsTypeList,
                                nii_idx=nii_idx,
                                box=[],
                                result=False,
                                )

    _ = next(gen)
    for out_frame_idx, out_obj_ids, out_mask_logits in gen:
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # 保存为图片序列
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for frame_idx, objs in video_segments.items():
            for obj_id, mask in objs.items():
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                mask_img = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_img).save(
                    # os.path.join(save_dir, f"{frame_idx}_obj{obj_id}.png")
                    os.path.join(save_dir, f"{frame_idx}.png")
                )

    return video_segments

def seg_nifti_inference(predictor, video_dir, ptsList, ptsTypeList,axis,save_dir,nii_idx=None,vote_threshold=2,auto_prompt = False,modal_id=0, organ_ids=[0,0,0],gt_path=None):
    
    if axis == 'all':
        axis_list = [0, 1, 2]
        auto_prompt = True
        if ptsList or ptsTypeList:
            raise ValueError(
                "axis='all' 时，不允许传入 ptsList/ptsTypeList，"
                "prompt 由内部 LPG 自动生成"
            )
    elif isinstance(axis, int) and axis in (0, 1, 2):
        axis_list = [axis]
        if not ptsList or not ptsTypeList:
            if not auto_prompt:
                raise ValueError(
                    f"您需要传入 ptsList 和 ptsTypeList,或者您可设置变量auto_prompt为 True 以允许自动提示."
                )
    else:
        raise ValueError(f"axis 必须是 0/1/2 或 'all'，收到: {axis}")
    
    # 跑每个 axis
    aligned_masks = []
    brain_ref = None
    
    for ax in axis_list:
        single_mask, brain = _run_single_axis(
            predictor=predictor,
            video_dir=video_dir,
            gt_path=gt_path,
            ptsList=ptsList,
            ptsTypeList=ptsTypeList,
            axis=ax,
            nii_idx=nii_idx,
            auto_prompt=auto_prompt,
            save_dir=save_dir,
            modal_id=modal_id,
            organ_ids=organ_ids
        )
        aligned_masks.append(single_mask)
        if brain_ref is None:
            brain_ref = brain
    
    # 融合
    if len(aligned_masks) > 1:
        vote = sum(m.astype(np.uint8) for m in aligned_masks)
        final_mask = (vote >= vote_threshold).astype(np.uint8)
    else:
        final_mask = aligned_masks[0]
    
    # 保存
    file_name = os.path.basename(video_dir)
    file_name = file_name.split('.')[0] + "_mask.nii.gz"
    save_path = os.path.join(save_dir, file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Directory created: {save_dir}")
    brain_ref.save_seg(final_mask, save_path)
    
    return save_path
    
    

def _run_single_axis(predictor, video_dir, ptsList, ptsTypeList, axis, nii_idx, auto_prompt,
                      save_dir, modal_id=0, organ_ids=[0,0,0], gt_path=None):
    stack_array = None
    video_segments = {}

    # 用真正的临时目录装帧图片：一进这个 with 块就是全新空目录，
    # 一出这个 with 块（不管正常结束还是中途报错）Python 自动把它删掉，
    # 不会在网络盘或本地盘上留下任何残留文件。
    with tempfile.TemporaryDirectory(prefix="brainsam_frames_") as frame_dir:
        inference, brain = create_inference_state(
            predictor,
            video_dir,
            frame_dir,
            prog_signal=None,
            axis=axis,
            modal_id=modal_id,
            organ_ids=organ_ids,
        )

        if nii_idx is None:
            raise ValueError("必须指定 nii_idx（交互的帧索引）。如果您希望程序自动决断，请将 nii_idx 设置为 'Autonomous'。")
        elif nii_idx == "Autonomous":
            num_frames = brain.get_slices_num(axis)
            nii_idx = num_frames // 2
            print(f"nii_idx set to Autonomous, automatically set to middle frame index: {nii_idx}")

        bbox = None
        ptsList_use = None
        ptsTypeList_use = None

        if auto_prompt==True:
            num_frames = brain.get_slices_num(axis)
            H_img, W_img = brain.get_slices_shape(axis)
            center_frame_idx = num_frames // 2
            points, labels, heatmap,backbone_out = predictor.lpeg_generate_points(inference, center_frame_idx)
            points_scaled = points.copy()
            points_scaled[:, 0] = points[:, 0] * W_img / 512  # x
            points_scaled[:, 1] = points[:, 1] * H_img / 512  # y

            pts = np.array(points_scaled)      # shape: (N, 2)
            lbs = np.array(labels) 
            ptsList_use = [pts]
            ptsTypeList_use = [lbs]
        elif auto_prompt=='pts':
            print("auto prompt: pts")
            gt_brain = nii_reader(gt_path)
            gt_slice, _ = gt_brain.get_slice_array(axis, nii_idx)
            gt_mask = (gt_slice > 0).astype(np.uint8)
            # 从 gt_mask 前景区域随机采样 3 个点作为正样本提示
            fg_coords = np.argwhere(gt_mask > 0)  # shape: (N, 2)，顺序为 (row, col)
            if len(fg_coords) < 3:
                raise ValueError(f"gt 前景像素不足 3 个（共 {len(fg_coords)} 个），无法采样提示点。")
            
            sampled = fg_coords[np.random.choice(len(fg_coords), size=3, replace=False)]
            pts = sampled[:, ::-1].astype(np.float32)   # (row,col) → (x,y)
            lbs = np.ones(3, dtype=np.int32)            # 全为正样本
            ptsList_use = [pts]
            ptsTypeList_use = [lbs]
            bbox = []
        elif auto_prompt == 'bbox':
            print("auto prompt: bbox")
            bbox = get_bbox_from_gt(gt_path, nii_idx, orient=axis, margin=10)
            ptsList_use = ptsList
            ptsTypeList_use = ptsTypeList
        else:
            ptsList_use = ptsList
            ptsTypeList_use = ptsTypeList
            bbox = []

        # 存一张标了 prompt 的切片图到 {save_dir}/log/，方便事后核查
        prompt_type_label = "bbox" if bbox else ("points" if ptsList_use else "none")
        _save_prompt_log_image(
            brain, axis, nii_idx, video_dir, save_dir,
            prompt_type=prompt_type_label,
            box=bbox if bbox else None,
            points=(ptsList_use[0] if ptsList_use else None),
            labels=(ptsTypeList_use[0] if ptsTypeList_use else None),
        )

        gen = _Continuous_inference(
            predictor=predictor,
            inference_state=inference,
            ptsList=ptsList_use,
            ptsTypeList=ptsTypeList_use,
            nii_idx=nii_idx,
            box=(bbox if bbox else None),
            result=False,
        )

        _ = next(gen)
        for out_frame_idx, out_obj_ids, out_mask_logits in gen:
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        # with 块结束 -> frame_dir 里的临时帧图片在这里被自动清理掉

    for out_frame_idx in video_segments.keys():
        mask_2d = None
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            if out_mask.shape[0] == 1:
                out_mask = out_mask.squeeze(0)
            mask_2d = out_mask
        stack_array = brain.align_to_me(axis, mask_2d, out_frame_idx, mask_array=stack_array)
    if stack_array is not None and stack_array.max() > 0:
        labels = measure.label(stack_array)
        print("保留最大联通域")
        stack_array = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(np.uint8)
    
    return stack_array, brain

if __name__ == "__main__":

    predictor = build_brainsam_predictor(
        config_file="/home/Dinglin/projects/BrianSAM/brainSAM/sam2/configs/sam2.1/brainsam2.1_hiera_b+.yaml",
            ckpt_path="/home/Dinglin/projects/BrianSAM/brainSAM/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+BrainSAM.yaml/checkpoints/checkpoint.pt",
            device="cuda:0")

    file_path = "/home/Dinglin/WuDL/Data_Brain/new_0724/raw/For_brainSAM_training/Optical_Monkey_52a1"
    inference,brain = create_inference_state(predictor, 
                                             video_dir=file_path,
                                             axis=0,
                                             
    )
    ptsList = [[[52, 120]]]
    ptsTypeList = [[[1]]]

    results = seg_folder_inference(predictor, file_path, ptsList, ptsTypeList,
                                   nii_idx=20,
                                   save_dir="/home/Dinglin/WuDL/output_masks")
    
    # %%
    import matplotlib.pyplot as plt

    results = seg_folder_inference(...)

    for frame_idx, objs in results.items():
        for obj_id, mask in objs.items():
            plt.imshow(mask.squeeze(), cmap='gray')
            plt.title(f"Frame {frame_idx} - Obj {obj_id}")
            plt.show()