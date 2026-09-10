# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects

def load_balance_loss(gate_weights_list, num_experts):
    total = 0.0
    for gate_w in gate_weights_list:
        frac = gate_w.mean(dim=0)  # [E]
        total += (frac * num_experts - 1).pow(2).mean()
    return total / len(gate_weights_list)

def make_boundary_gt(mask_gt:torch.Tensor, kernel_size:int=3):
    """
    用形态学 dilation - erosion 生成 boundary GT。
    
    Args:
        mask_gt: [N, 1, H, W] 二值 mask(0/1 float)
        kernel_size: 边界宽度,3 ≈ 1 像素,5 ≈ 2 像素
    
    Returns:
        boundary: [N, 1, H, W] 边界带,1=边界像素,0=非边界
    """
    pad = kernel_size // 2
    # max-pool == dilation
    dilated = F.max_pool2d(mask_gt, kernel_size, stride=1, padding=pad)
    # -max-pool(-x) == erosion
    eroded = -F.max_pool2d(-mask_gt, kernel_size, stride=1, padding=pad) #把图像取反，取最大值后再取反就得到了腐蚀的效果
    boundary = (dilated - eroded).clamp(0, 1)
    return boundary

def boundary_loss(
    boundary_logits: torch.Tensor,
    boundary_gt: torch.Tensor,
    num_objects,
    use_dice: bool = True,
):
    """
    Boundary loss = BCE(logits) + (optional) Dice。
    边界像素稀疏,BCE 容易被背景主导,加 dice 平衡。
    
    Args:
        boundary_logits: [N, 1, H, W]
        boundary_gt:     [N, 1, H, W],binary
        num_objects:     batch 内 object 数(用于归一化)
        use_dice:        是否加 dice 项
    
    Returns:
        scalar loss(已经除过 num_objects)
    """
    # resize logits 到 GT 尺寸(SAM2 的 high_res 通常就是 image_size,但 boundary head 输出是 1/4 分辨率)
    if boundary_logits.shape[-2:] != boundary_gt.shape[-2:]:
        boundary_logits = F.interpolate(
            boundary_logits,
            size=boundary_gt.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    
    n_pos = boundary_gt.sum()
    n_neg = boundary_gt.numel() - n_pos
    pos_weight = (n_neg / n_pos.clamp(min=1)).clamp(max=50)  # 防止过大

    # BCE
    loss_bce = F.binary_cross_entropy_with_logits(
        boundary_logits, boundary_gt, reduction="none", pos_weight=pos_weight
    )
    # [N, 1, H, W] → mean over spatial, sum over N
    loss_bce = loss_bce.flatten(1).mean(-1).sum() / num_objects
    
    if not use_dice:
        return loss_bce
    
    # Dice on boundary
    prob = boundary_logits.sigmoid().flatten(1)
    gt_flat = boundary_gt.flatten(1)
    numerator = 2 * (prob * gt_flat).sum(-1)
    denominator = prob.sum(-1) + gt_flat.sum(-1)
    loss_dice = (1 - (numerator + 1) / (denominator + 1)).sum() / num_objects
    
    return loss_bce + loss_dice

class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        lb_loss_weight=0.0, #*#
        lpeg_loss_weight=0.0, #*#
        use_boundary_loss=False, #*#
        boundary_kernel_size=3, #*#
        boundary_use_dice=True
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
            lb_loss_weight: weight for load balance loss on MoE gates #*#
            lpeg_loss_weight: weight for LPEG loss #*#
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.lb_loss_weight = lb_loss_weight
        
        self.lpeg_loss_weight = lpeg_loss_weight
        self.lpeg_criterion = LPEGLoss(loss_weight=1.0) if lpeg_loss_weight > 0 else None

        self.use_boundary_loss = use_boundary_loss
        self.boundary_kernel_size = boundary_kernel_size
        self.boundary_use_dice = boundary_use_dice

        if self.use_boundary_loss and "loss_boundary" not in self.weight_dict:
            self.weight_dict["loss_boundary"] = 0.5  # 默认权重

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor,expert_acts=None, lpeg_info=None):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v
        if expert_acts and self.lb_loss_weight > 0:
            lb = load_balance_loss(expert_acts, num_experts=4)
            losses["loss_lb"] = lb
            losses[CORE_LOSS_KEY] = losses[CORE_LOSS_KEY] + self.lb_loss_weight * lb
        
        #*#
        if lpeg_info is not None and self.lpeg_loss_weight > 0:
            lpeg_heatmaps, gt_masks = lpeg_info
            lpeg_losses = []
            for t, hmap in lpeg_heatmaps.items():
                if t in gt_masks:
                    gt_resized = F.interpolate(
                        gt_masks[t].float(), size=hmap.shape[2:],
                        mode="bilinear", align_corners=False
                    )
                    gt_resized = (gt_resized > 0.5).float()
                    lpeg_losses.append(
                        F.binary_cross_entropy_with_logits(hmap, gt_resized)
                    )
            if lpeg_losses:
                lpeg_loss = torch.stack(lpeg_losses).mean()
                losses["loss_lpeg"] = lpeg_loss
                losses[CORE_LOSS_KEY] = losses[CORE_LOSS_KEY] + self.lpeg_loss_weight * lpeg_loss

            #*#

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        boundary_logits_list = outputs.get("multistep_pred_boundary_logits", None)

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        if self.use_boundary_loss and boundary_logits_list is not None:
            losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0, "loss_boundary": 0}
        else:
            losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        if self.use_boundary_loss and boundary_logits_list is not None:
            boundary_gt = make_boundary_gt(target_masks, self.boundary_kernel_size)
        else:
            boundary_gt = None
    
        for step_idx, (src_masks, ious, object_score_logits) in enumerate(zip(
            src_masks_list, ious_list, object_score_logits_list
        )):
            b_logits = (boundary_logits_list[step_idx]
                    if boundary_logits_list is not None else None)

            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits,
                boundary_logits=b_logits, boundary_gt=boundary_gt
            )

        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits,boundary_logits=None,
        boundary_gt=None
    ):
        target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        if boundary_logits is not None and boundary_gt is not None:

            l_b = boundary_loss(
                boundary_logits, boundary_gt, num_objects,
                use_dice=self.boundary_use_dice,
            )
            losses["loss_boundary"] += l_b

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

        

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss

class LPEGLoss(nn.Module):
    """
    LPEG 热力图的监督损失。
    用 GT mask 的距离变换作为监督信号：
    mask 内部离边界越远的点，热力图值应该越高。

    Args:
        loss_weight: 损失权重
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, heatmap: torch.Tensor, gt_mask: torch.Tensor):
        """
        Args:
            heatmap: [B, 1, H_feat, W_feat] LPEG 输出的热力图
            gt_mask: [B, 1, H_img, W_img]   GT 分割 mask

        Returns:
            loss: 标量
        """
        # 将 GT mask 下采样到特征图尺寸
        gt_resized = F.interpolate(
            gt_mask.float(),
            size=heatmap.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        gt_resized = (gt_resized > 0.5).float()

        # 用 BCE loss 监督热力图
        loss = F.binary_cross_entropy_with_logits(heatmap, gt_resized)

        return self.loss_weight * loss
