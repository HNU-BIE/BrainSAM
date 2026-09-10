from dataclasses import dataclass
from typing import List, Optional, Tuple,Union
import torch
from PIL import Image as PILImage
from tensordict import tensorclass



@dataclass
class Frame:
    data: Union[torch.Tensor,PILImage.Image]
    objects: List[object]
@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]
    Prior: Tuple = None