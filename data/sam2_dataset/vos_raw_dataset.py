import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import pandas as pd

import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from data.sam2_dataset.vos_segment_loader import (
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
)
from data.data_info import *

@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)

class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()

class BrainRawDataset(VOSRawDataset):
    """
     Dataset where the annotation in the format of png and video files
     """

    def __init__(
            self,
            img_folder,
            gt_folder,
            file_list_txt=None,
            excluded_videos_list_txt=None,
            sample_rate=1,
            is_palette=True,
            single_object_mode=False,
            truncate_video=-1,
            frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        # if video_name != "T1_HumanCor_slim25648":
        #     return None, None,None
        ##
        task = video_name
        # print("video_name:",video_name)
        modal = task.split("_")[0]
        organ = ('').join(task.split("_")[1:-1])
        # print(video_name,modal,organ)
        # T1_rabbit_brain_extract_anatomical
        # map modal and organ to idx
        modal = modal_map[modal_dict[modal]]
        # print(f"modal:=>{modal}")
        organ.strip('0123456789')
        # print(f"organ:=>{organ}")
        for k,v in brain_level_1_dict.items():
            if organ in v:
                brain_level_1 = brain_level_1_map[k]
                # print(f"brain_level_1:=>{brain_level_1}")
                break
        for k,v in brain_level_2_dict.items():
            if organ in v:
                brain_level_2 = brain_level_2_map[k]
                # print(f"brain_level_2:=>{brain_level_2}")
                break
        task = task.rsplit('_',1)[0]
        brain_level_3 = task_idx[task]
        # print(f"brain_level_3:=>{brain_level_3}")
        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.png")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)

        return video, segment_loader,(modal,brain_level_1,brain_level_2,brain_level_3)

    def __len__(self):
        return len(self.video_names)
