import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from utils.nifti_reader import NIFTI_READER as nii_reader

# mpl.use('Agg')
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )



def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


@torch.no_grad()
def blockface_inference(predictor, ptsList, ptsTypeList,video_dir,result) :

    """
    :param predictor: a predictor build by build_sam2_video_predictor.
    :param ptsList: the point list.(e.g.[obj1,obj2,...],and obj1 like [[502, 602],[577,618]])
    :param ptsTypeList: the points types list,which must match the length of the points list.
                         for labels, `1` means positive click and `0` means negative click(e.g.[1,1])
    :param video_dir: the blockface folder.
    :return: return a dictionary,like {frame0:{obj_id:mask},frame1:{obj_id:mask}...}
    """
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # take a look the first video frame
    frame_idx = 80
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
    if result:plt.show()
    image=Image.open(os.path.join(video_dir, frame_names[frame_idx]))
    print(f"image size is{image.size}")
    width, height = image.size
    mask_in_image_size=np.zeros((height,width),dtype=np.uint8)
    inference_state = predictor.init_state(video_path=video_dir)

    # predictor.reset_state(inference_state)

    ann_frame_idx = 62  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)


    points = np.array(np.array(ptsList), dtype=np.float32)

    # for labels, `1` means positive click and `0` means negative click
    labels = np.array(np.array(ptsTypeList), np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))


    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    if result:plt.show()

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    # e.g.
    # video_segments=
    # {
    #     0: {  # 第 0 帧
    #         1: array([[False, True, False], [True, False, True]]),  # 目标对象 ID 为 1 的掩码
    #         2: array([[True, True, False], [False, True, False]]),  # 目标对象 ID 为 2 的掩码
    #     },
    #     1: {  # 第 1 帧
    #         1: array([[True, False, True], [False, True, False]]),  # 目标对象 ID 为 1 的掩码
    #         3: array([[False, False, True], [True, False, True]]),  # 目标对象 ID 为 3 的掩码
    #     },
    #     2: {  # 第 2 帧
    #         2: array([[True, True, False], [False, False, True]]),  # 目标对象 ID 为 2 的掩码
    #         4: array([[False, True, True], [True, True, False]]),  # 目标对象 ID 为 4 的掩码
    #     }
    # }
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,start_frame_idx=0):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }




    print(f"len of video_segments:{len(video_segments)}")
    # render the segmentation results every few frames
    vis_frame_stride =1
    plt.close("all")
    first_mask = next(iter(video_segments[0].values()))
    mask_height, mask_width =width,height
    num_frames = len(frame_names)
    stacked_mask_array = np.zeros((mask_height, mask_width, num_frames), dtype=np.uint8)
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        # plt.title(f"frame {out_frame_idx}")
        # plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # print(f"out_maskShape:{out_mask.shape}")
            # show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            # h,w=out_mask.shape[-2:]
            out_mask=out_mask.squeeze(0)
            # print(f"out_maskShape2:{out_mask.shape}")
            stacked_mask_array[:, :, out_frame_idx] += out_mask
            if result: plt.show()

    return stacked_mask_array

@torch.no_grad()
def new_inference(predictor,
                          inference_state,
                          ptsList=None,
                          ptsTypeList=None,
                          box=[],
                          result=False,
                          nii_idx=0,
                          max_frame_num_to_track=None,
                          reverse=False,
                        ) :
    ann_frame_idx = nii_idx  # the frame index we interact with
    ann_obj_id = []
    if len(ptsList) != 0:
        for i, _ in enumerate(ptsList, start=1):
            ann_obj_id.append(i)  # give a unique id to each object we interact with (it can be any integers)
    elif len(box) != 0:
        ann_obj_id = []
        for i, _ in enumerate(box, start=1):
            ann_obj_id.append(i)

    points = np.array(np.array(ptsList), dtype=np.float32)
    if len(box) != 0:
        box = np.array(box, dtype=np.float32)
    else:
        box = None
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array(np.array(ptsTypeList), np.int32)
    print(f"len of ids:{len(ann_obj_id)}")
    for i in range(len(ann_obj_id)):
        # print(f"box[i] is {box[i]}")
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id[i],
            points=points[i],
            labels=labels[i],
            box=box[i] if box is not None else None,
        )
        print(f"points:{points[i]},labels:{labels[i]}")
        print(f"box:{box if box is not None else None}")

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                    start_frame_idx=nii_idx,
                                                                                    ):
        yield out_frame_idx, out_obj_ids, out_mask_logits

def create_inference_state(predictor,video_dir,prog_signal=None,axis=None):
    if video_dir.endswith(('.nii', '.nii.gz')):
        rawFilePath = video_dir
        folder, filename = os.path.split(video_dir)
        filename = os.path.splitext(filename)[0]
        if axis == 0:
            video_dir = rf"{os.path.dirname(video_dir)}/temp/{filename}_axi"
        elif axis == 1:
            video_dir = rf"{os.path.dirname(video_dir)}/temp/{filename}_cor"
        elif axis == 2:
            video_dir = rf"{os.path.dirname(video_dir)}/temp/{filename}_sag"
        os.makedirs(video_dir, exist_ok=True)
        brain = nii_reader(rawFilePath)
        # mask_height, mask_width, mask_depth = get_dimensions(axis)
        print(f"axis:{axis}")
        num_frames = brain.get_slices_num(axis)
        # print(f"num_frames:{num_frames}")
        if len(os.listdir(video_dir)) < num_frames:
            print("re_split_frames")
            for sli in range(0, num_frames):
                slice,_ = brain.get_slice_array(axis, sli)
                slice_img_normalized = np.interp(slice, (slice.min(), slice.max()), (0, 255)).astype(np.uint8)
                slice_pil_img = Image.fromarray(slice_img_normalized)
                slice_pil_img.save(f"{video_dir}/{sli}.png")

            # frame_names = [
            #     p for p in os.listdir(video_dir)
            #     if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
            # ]
            # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir,prog_signal=prog_signal)
        return inference_state, brain
    elif os.path.isdir(video_dir):
        try:
            frame_names = [
                p for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            inference_state = predictor.init_state(video_path=video_dir,prog_signal=prog_signal,modal_id=5,      # ← 加这两行
                organ_ids=[1,2,15],)
            return inference_state
        except Exception as e:
            raise RuntimeError(f"Failed to create inference state from directory: {video_dir}. Error: {e}")
    else:
        raise ValueError("input a unknow type, you can wait for the author to update.")

@torch.no_grad()
def _Continuous_inference(predictor,
                          inference_state,
                          ptsList=None,
                          ptsTypeList=None,
                          box=[],
                          result=False,
                          nii_idx=59,
                          start_frame=0,
                          max_frame_num_to_track=None,
                          reverse=False,
                          prog_signal=None,
                        ) :

    """
    :param predictor: a predictor build by build_sam2_video_predictor.
    :param ptsList: the point list.(e.g.[obj1,obj2,...],and obj1 like [[502, 602],[577,618]])
    :param ptsTypeList: the points types list,which must match the length of the points list.
                         for labels, `1` means positive click and `0` means negative click(e.g.[1,1])
    :param video_dir: the blockface folder.
    :return: return a dictionary,like {frame0:{obj_id:mask},frame1:{obj_id:mask}...}
    """
    ann_frame_idx = nii_idx  # the frame index we interact with

    print("ann_frame_idx:",ann_frame_idx)
    ann_obj_id=[]
    if len(ptsList)!=0:
        for i,_ in enumerate(ptsList,start=1):
                ann_obj_id.append(i)  # give a unique id to each object we interact with (it can be any integers)
    elif len(box)!=0:
        ann_obj_id = []
        for i, _ in enumerate(box, start=1):
            ann_obj_id.append(i)  # give a unique id to each object we interact with (it can be any integers)

    points = np.array(ptsList, dtype=np.float32)
    if len(box) !=0:
        box = np.array(box, dtype=np.float32)
    else:
        box =None
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array(ptsTypeList, dtype=np.int32)
    out_mask_logits =None
    for i in range(len(ann_obj_id)):
        # print(f"box[i] is {box[i]}")
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id[i],
            points=points[i] if i<len(points) else None,
            labels=labels[i] if i<len(labels) else None,
            box=box[i] if box is not None else None,
        )
    yield (out_mask_logits > 0.0).cpu().numpy()
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    # if max_frame_num_to_track==None:
    #     max_frame_num_to_track=inference_state["num_frames"]
        # print(f"max_frame_num_to_track set to num_frames:{max_frame_num_to_track}")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                    start_frame_idx=start_frame,
                                                                                    max_frame_num_to_track=max_frame_num_to_track,
                                        reverse=reverse,
                                        prog_signal=prog_signal
                                                                                    ):
        # video_segments[out_frame_idx] = {
        #     out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        #     for i, out_obj_id in enumerate(out_obj_ids)
        # }
        yield out_frame_idx, out_obj_ids, out_mask_logits

def seg_nifti_inference(predictor, video_dir, ptsList, ptsTypeList,axis,save_dir,nii_idx=59):
    stack_array = None
    video_segments = {}

    inference, brain = create_inference_state(predictor,video_dir,prog_signal=None,axis=axis)
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

    # print(len(video_segments))

    for out_frame_idx in video_segments.keys():
        # num_items = len(self.video_segments[out_frame_idx])
        mask_2d = None
        # print(f"Number of items in self.video_segments[{out_frame_idx}]: {num_items}")
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            if out_mask.shape[0] == 1: out_mask = out_mask.squeeze(0)
            out_mask += out_mask
            mask_2d = out_mask
        stack_array = brain.align_to_me(axis, mask_2d, out_frame_idx, mask_array=stack_array)

    file_name = os.path.basename(video_dir)
    file_name = file_name.split('.')[0]+"_mask.nii.gz"
    save_path = os.path.join(save_dir, file_name)
    # print(f"Saving segmentation result to {save_path}")
    brain.save_seg(stack_array, save_path)
    return save_path

if __name__ == "__main__":
    from sam2.build_sam import build_brainsam_video_predictor

    v_sam2_model = build_brainsam_video_predictor(
        # config_file="configs/sam2.1/sam2.1_hiera_b+.yaml",
        config_file="/home/Dinglin/PycharmProjects/BrainSAM920/sam2/configs/sam2.1_training/sam2.1_hiera_b+BrainSAM.yaml",
        ckpt_path="/home/Dinglin/PycharmProjects/BrainSAM920/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+BrainSAM.yaml/checkpoints/checkpoint.pt",
        # ckpt_path="/home/Dinglin/WuDL/Data_Brain/new_0724/checkPoints/chk_blockface_b+.pt",
        device="cuda:0")
    folder_path="/home/Dinglin/WuDL/Data_Brain/new_0724/raw/For_brainSAM_training/Optical_Monkey_52a1"

    ptsList = [[[52, 120]]]
    ptsTypeList = [[[1]]]
    save_dir="/home/Dinglin/WuDL/Data_Brain/new_0724/temp/new_test0403"

    inference, brain = create_inference_state(v_sam2_model, folder_path, prog_signal=None, axis=1)
    brain.show_slice(1,120)
    # _Continuous_inference(v_sam2_model,
    #                       inference,
    #                       ptsList=[[[162, 59]]],
    #                       ptsTypeList=[[[1]]],
    #                       box=[],
    #                       result=True,
    #                       nii_idx=0,
    #                       start_frame=0,
    #                       max_frame_num_to_track=None,
    #                       reverse=False,
    #                       prog_signal=None,
    #                       )
    # folder_path="/home/lab/DockerProject/BrainSAM/assets/"
    _ = seg_nifti_inference(v_sam2_model,folder_path,ptsList,ptsTypeList,1,save_dir,nii_idx=120)

