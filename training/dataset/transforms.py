"""Compatibility layer for SAM2 training transforms.

This module re-exports the existing VideoDatapoint transforms from
data.sam2_dataset.transforms and adds a small function-based medical
augmentation toolkit that can be plugged into ComposeAPI directly.
"""

from __future__ import annotations

import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image as PILImage

from data.sam2_dataset.transforms import *  # noqa: F401,F403
from training.utils.data_utils import VideoDatapoint


def _is_pil_image(image) -> bool:
    return isinstance(image, PILImage.Image)


def _image_to_array(image) -> np.ndarray:
    if _is_pil_image(image):
        return np.asarray(image)
    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
        if array.ndim == 3:
            array = np.transpose(array, (1, 2, 0))
        return array
    return np.asarray(image)


def _array_to_image(array: np.ndarray, template):
    array = np.clip(array, 0.0, 1.0)
    if _is_pil_image(template):
        if array.ndim == 2:
            return PILImage.fromarray((array * 255.0).astype(np.uint8), mode="L")
        if array.shape[-1] == 1:
            return PILImage.fromarray((array[..., 0] * 255.0).astype(np.uint8), mode="L")
        return PILImage.fromarray((array * 255.0).astype(np.uint8))
    if torch.is_tensor(template):
        tensor = torch.from_numpy(array)
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        return tensor.to(dtype=template.dtype, device=template.device)
    return array


def _apply_to_frames(datapoint: VideoDatapoint, fn: Callable) -> VideoDatapoint:
    for frame in datapoint.frames:
        frame.data = fn(frame.data)
    return datapoint


def make_medical_window_normalizer(
    modality: str = "ct",
    window_level: Optional[Tuple[int, int]] = None,
) -> Callable:
    """Create a transform that normalizes medical images before ToTensorAPI.

    The returned transform keeps masks untouched and only changes frame.data.
    For CT it applies windowing; for other modalities it uses robust percentile
    clipping.
    """

    window_level = window_level or (400, 40)

    def transform(datapoint: VideoDatapoint, **kwargs):
        def normalize_image(image):
            array = _image_to_array(image).astype(np.float32)

            if modality == "ct":
                window_width, window_center = window_level
                min_value = window_center - window_width / 2.0
                max_value = window_center + window_width / 2.0
                array = np.clip(array, min_value, max_value)
                array = (array - min_value) / (max_value - min_value + 1e-6)
            elif modality == "mri":
                low, high = np.percentile(array, [2, 98])
                array = np.clip(array, low, high)
                array = (array - low) / (high - low + 1e-6)
            elif modality == "xray":
                max_value = array.max() if array.max() > 255 else 255.0
                array = array / (max_value + 1e-6)
            elif modality == "ultrasound":
                low, high = np.percentile(array, [5, 95])
                array = np.clip(array, low, high)
                array = (array - low) / (high - low + 1e-6)
            elif modality == "pathology":
                if array.max() > 1:
                    array = array / 255.0
            else:
                low, high = np.percentile(array, [0.5, 99.5])
                array = np.clip(array, low, high)
                array = (array - low) / (high - low + 1e-6)

            return _array_to_image(array, image)

        return _apply_to_frames(datapoint, normalize_image)

    return transform


def make_medical_random_brightness(
    brightness_range: Tuple[float, float] = (0.85, 1.15),
    p: float = 0.15,
) -> Callable:
    def transform(datapoint: VideoDatapoint, **kwargs):
        if random.random() > p:
            return datapoint

        def augment(image):
            return F.adjust_brightness(image, random.uniform(*brightness_range))

        return _apply_to_frames(datapoint, augment)

    return transform


def make_medical_random_contrast(
    contrast_range: Tuple[float, float] = (0.85, 1.15),
    p: float = 0.15,
) -> Callable:
    def transform(datapoint: VideoDatapoint, **kwargs):
        if random.random() > p:
            return datapoint

        def augment(image):
            return F.adjust_contrast(image, random.uniform(*contrast_range))

        return _apply_to_frames(datapoint, augment)

    return transform


def make_medical_random_gamma(
    gamma_range: Tuple[float, float] = (0.9, 1.1),
    p: float = 0.15,
) -> Callable:
    def transform(datapoint: VideoDatapoint, **kwargs):
        if random.random() > p:
            return datapoint

        def augment(image):
            return F.adjust_gamma(image, gamma=random.uniform(*gamma_range), gain=1.0)

        return _apply_to_frames(datapoint, augment)

    return transform


def make_medical_random_gaussian_noise(
    noise_std: float = 0.02,
    p: float = 0.1,
) -> Callable:
    def transform(datapoint: VideoDatapoint, **kwargs):
        if random.random() > p:
            return datapoint

        def augment(image):
            array = _image_to_array(image).astype(np.float32)
            if array.max() > 1.0:
                array = array / 255.0
            noise = np.random.normal(0.0, noise_std, size=array.shape).astype(np.float32)
            array = np.clip(array + noise, 0.0, 1.0)
            return _array_to_image(array, image)

        return _apply_to_frames(datapoint, augment)

    return transform


def make_medical_random_gaussian_blur(
    kernel_size: int = 3,
    sigma: float = 0.8,
    p: float = 0.1,
) -> Callable:
    def transform(datapoint: VideoDatapoint, **kwargs):
        if random.random() > p:
            return datapoint

        def augment(image):
            return F.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        return _apply_to_frames(datapoint, augment)

    return transform


def make_sam2_medical_augmentor(
    modality: str = "ct",
    window_level: Optional[Tuple[int, int]] = None,
    augmentation_strength: float = 0.5,
    organ_type: str = "generic",
) -> Callable:
    """Return a ComposeAPI-friendly callable for conservative medical augmentation.

    This only affects image data. Spatial transforms that must keep masks aligned
    are already handled by the existing RandomHorizontalFlip / RandomAffine
    classes in data.sam2_dataset.transforms.
    """

    organ_configs = {
        "ultrasound": (0.10, 0.10, 0.10, 0.30, 0.15),
        "ct_lung": (0.10, 0.10, 0.10, 0.05, 0.05),
        "ct_abdomen": (0.10, 0.10, 0.15, 0.10, 0.10),
        "mri_brain": (0.15, 0.15, 0.20, 0.10, 0.10),
        "xray": (0.10, 0.10, 0.10, 0.10, 0.10),
        "pathology": (0.20, 0.20, 0.15, 0.05, 0.10),
        "generic": (0.15, 0.15, 0.15, 0.10, 0.10),
    }

    brightness_p, contrast_p, gamma_p, noise_p, blur_p = organ_configs.get(
        organ_type, organ_configs["generic"]
    )

    transforms = [
        make_medical_window_normalizer(modality=modality, window_level=window_level),
        make_medical_random_brightness(
            brightness_range=(0.85, 1.15), p=brightness_p * augmentation_strength
        ),
        make_medical_random_contrast(
            contrast_range=(0.85, 1.15), p=contrast_p * augmentation_strength
        ),
        make_medical_random_gamma(gamma_range=(0.9, 1.1), p=gamma_p * augmentation_strength),
        make_medical_random_gaussian_noise(noise_std=0.02, p=noise_p * augmentation_strength),
        make_medical_random_gaussian_blur(kernel_size=3, sigma=0.8, p=blur_p * augmentation_strength),
    ]

    def transform(datapoint: VideoDatapoint, **kwargs):
        for item in transforms:
            datapoint = item(datapoint, **kwargs)
        return datapoint

    return transform


__all__ = [
    "ComposeAPI",
    "RandomAffine",
    "RandomHorizontalFlip",
    "RandomMosaicVideoAPI",
    "RandomGrayscale",
    "RandomResizeAPI",
    "ToTensorAPI",
    "NormalizeAPI",
    "ColorJitter",
    "make_medical_window_normalizer",
    "make_medical_random_brightness",
    "make_medical_random_contrast",
    "make_medical_random_gamma",
    "make_medical_random_gaussian_noise",
    "make_medical_random_gaussian_blur",
    "make_sam2_medical_augmentor",
]
