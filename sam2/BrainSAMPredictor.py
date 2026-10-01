from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_info import *

def get_index(map_idx):
    idx = map_idx
    for i in range(list(map_idx.items())[-1][0]):
        if i not in idx:
            idx[i] = 0
    idx = dict(sorted(idx.items(), key=lambda item: item[0]))
    return torch.LongTensor(list(idx.values()))

class BrainSAMPredictor(SAM2VideoPredictor):
    def __init__(self,
                 use_boundary_prior=False, 
                 register_buffers_for_prior_idx=False, 
                 embedding_dim=64,
                  pos=None,
                  lpeg=None,
                    **kwargs):
        print(f"'lpeg' in kwargs: {'lpeg' in kwargs}")
        super().__init__(**kwargs)  
        
        # embedding register
        if pos:
            self.pos = pos
        else:
            self.pos = list(range(len(self.image_encoder.trunk.blocks)))
        if register_buffers_for_prior_idx:
            print("Registering buffers for prior indices...")
            modal_index = get_index(modal_map_idx)
            brain_index_1 = get_index(brain_level_1_map_idx)
            brain_index_2 = get_index(brain_level_2_map_idx)
            brain_index_3 = get_index(brain_level_3_map_idx)

            self.image_encoder.register_buffer('modal_index', modal_index, persistent=True)
            self.image_encoder.register_buffer('brain_index_1', brain_index_1, persistent=True)
            self.image_encoder.register_buffer('brain_index_2', brain_index_2, persistent=True)
            self.image_encoder.register_buffer('brain_index_3',brain_index_3, persistent=True)

            modal_embed = nn.Embedding(len(modal_index), embedding_dim)
            organ_embed_0 = nn.Embedding(1, embedding_dim)
            organ_embed_1 = nn.Embedding(len(brain_level_1_map_idx), embedding_dim)
            organ_embed_2 = nn.Embedding(len(brain_level_2_map_idx), embedding_dim)
            organ_embed_3 = nn.Embedding(len(brain_level_3_map_idx), embedding_dim)  
            organ_embed_4 = nn.Embedding(len(task_list) + 1, embedding_dim)  # tasks

            nn.init.normal_(modal_embed.weight, mean=0.0, std=0.02)
            nn.init.zeros_(organ_embed_0.weight)
            nn.init.normal_(organ_embed_1.weight, mean=0.0, std=0.02)
            nn.init.normal_(organ_embed_2.weight, mean=0.0, std=0.02)
            nn.init.normal_(organ_embed_3.weight, mean=0.0, std=0.02)
            nn.init.normal_(organ_embed_4.weight,  mean=0.0, std=0.02)

            self.image_encoder.modal_embed = modal_embed
            self.image_encoder.organ_embed = nn.ModuleList([
                organ_embed_0, organ_embed_1, organ_embed_2, organ_embed_3,
                organ_embed_4
            ])

        self.lpeg = lpeg
        self.use_boundary_prior = use_boundary_prior
        if  self.use_boundary_prior:
            print("BrainSAMPredictor initialized with boundary_prior enabled.")

    def forward_image(self, img_batch, modal=None, organ=None):
        backbone_out = self.image_encoder(img_batch, modal, organ)
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1])
        return backbone_out

    @torch.inference_mode()
    def init_state(self, video_path, prog_signal=None,
                   offload_video_to_cpu=False, offload_state_to_cpu=False,
                   async_loading_frames=False,
                   modal_id=0, organ_ids=None):
        

        if organ_ids is None:
            organ_ids = [0, 0, 0, 0]

        inference_state = super().init_state(
            video_path=video_path,
            prog_signal=prog_signal,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            modal_id=modal_id, organ_ids=organ_ids
        )
        
        device = inference_state["device"]

        if self.use_boundary_prior and hasattr(self.image_encoder, 'organ_embed'):
            #create Prior Embedding Tree
            modal_idx = self.image_encoder.modal_index[
                torch.tensor([modal_id], device=device)]                 # [1]
            o1_idx = self.image_encoder.brain_index_1[
                torch.tensor([organ_ids[0]], device=device)]             # [1]
            o2_idx = self.image_encoder.brain_index_2[
                torch.tensor([organ_ids[1]], device=device)]             # [1]
            o3_idx = self.image_encoder.brain_index_3[
                            torch.tensor([organ_ids[2]], device=device)]             # [1]
            task_idx = torch.tensor([organ_ids[3]], device=device)        # [1]

            m_e  = self.image_encoder.modal_embed(modal_idx)              # [1, 64]
            o1_e = self.image_encoder.organ_embed[1](o1_idx)              # [1, 64]
            o2_e = self.image_encoder.organ_embed[2](o2_idx)              # [1, 64]
            o3_e = self.image_encoder.organ_embed[3](o3_idx)              # [1, 64]
            task_e = self.image_encoder.organ_embed[4](task_idx)            # [1, 64]

            # print(f"[DEBUG] modal_idx:{modal_idx.item()} o1_idx:{o1_idx.item()} o2_idx:{o2_idx.item()}")
            # print(f"[DEBUG] o1_e[:3]:{self.image_encoder.organ_embed[1](o1_idx)[0,:3].tolist()}")
            # print(f"[DEBUG] o2_e[:3]:{self.image_encoder.organ_embed[2](o2_idx)[0,:3].tolist()}")
            # print(f"[DEBUG] o3_e[:3]:{self.image_encoder.organ_embed[3](task_idx)[0,:3].tolist()}")
            boundary_prior = torch.cat([m_e, o1_e, o2_e, o3_e, task_e], dim=-1)   # [1, 320]
            inference_state["boundary_prior"] = boundary_prior
        
        else:
            inference_state["boundary_prior"] = None
          

        return inference_state

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        device = inference_state["device"]

        if hasattr(self.image_encoder, 'modal_embed') and hasattr(self.image_encoder, 'organ_embed'):
            modal_id = inference_state.get("modal_id", 0)
            organ_ids = inference_state.get("organ_ids", [0, 0, 0, 0])

            modal_idx = self.image_encoder.modal_index[torch.tensor([modal_id], device=device)]
            modal_embed = self.image_encoder.modal_embed(modal_idx)  # [1, 64]

            o1_idx = self.image_encoder.brain_index_1[torch.tensor([organ_ids[0]], device=device)]
            o2_idx = self.image_encoder.brain_index_2[torch.tensor([organ_ids[1]], device=device)]
            o3_idx = self.image_encoder.brain_index_3[torch.tensor([organ_ids[2]], device=device)]
            task_idx = torch.tensor([organ_ids[3]], device=device)

            o1_embed = self.image_encoder.organ_embed[1](o1_idx)    # [1, 64]
            o2_embed = self.image_encoder.organ_embed[2](o2_idx)    # [1, 64]
            o3_embed = self.image_encoder.organ_embed[3](o3_idx)    # [1, 64]
            task_embed = self.image_encoder.organ_embed[4](task_idx) # [1, 64]
            zero_embed = self.image_encoder.organ_embed[0](torch.zeros(1, dtype=torch.long, device=device))

            organ_embed = [zero_embed, o1_embed, o2_embed, o3_embed, task_embed]

            # Store into inference_state for use by subsequent frames
            inference_state["modal_embed"] = modal_embed
            inference_state["organ_embed"] = organ_embed

        if backbone_out is None:
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
        
 
            modal = inference_state.get("modal_embed", None)
            organ = inference_state.get("organ_embed", None)
            # print(f"[DEBUG] _get_image_feature modal_embed: {modal.shape if modal is not None else None}, organ_embed: {[e.shape for e in organ] if organ is not None else None}")
     

            backbone_out = self.forward_image(image, modal=modal, organ=organ)
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(batch_size, -1, -1, -1)

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        if inference_state["storage_device"].type == "cpu":
            features = tuple(f.to(torch.bfloat16) for f in features)
        return features
    
    def _run_single_frame_inference(self, inference_state, output_dict, frame_idx,
                                 batch_size, is_init_cond_frame, point_inputs,
                                 mask_inputs, reverse, run_mem_encoder,
                                 prev_sam_mask_logits=None):
        
        boundary_prior = inference_state.get("boundary_prior", None)
        
        
        if boundary_prior is not None and boundary_prior.size(0) != batch_size:
            boundary_prior = boundary_prior.expand(batch_size, -1).contiguous()

        (_, _, current_vision_feats, current_vision_pos_embeds, feat_sizes) = \
            self._get_image_feature(inference_state, frame_idx, batch_size)

        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            boundary_prior=boundary_prior,    
        )

        
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(pred_masks_gpu, self.fill_hole_area)
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu

    @torch.no_grad()
    def lpeg_generate_points(self, inference_state, frame_idx):
        """
        用 LPEG 自动生成 prompt 点
        
        Args:
            inference_state: init_state 返回的状态
            frame_idx: 帧索引
            
        Returns:
            points: np.ndarray [K, 2] (x, y)
            labels: np.ndarray [K] (全1=前景)
        """
        assert self.lpeg is not None, "LPEG not loaded"
        
        device = inference_state["device"]
        image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
        
        # get modal/organ embed
        modal = inference_state.get("modal_embed", None)
        organ = inference_state.get("organ_embed", None)
        
        # get FPN feature from Froward backbone 
        backbone_out = self.forward_image(image, modal=modal, organ=organ)
        fpn_feat = backbone_out["backbone_fpn"][-1]
        
        # generate points
        with torch.no_grad():
            point_inputs, heatmap = self.lpeg.forward_with_heatmap(fpn_feat)
        
        points = point_inputs["point_coords"][0].cpu().numpy()   # [K, 2]
        labels = point_inputs["point_labels"][0].cpu().numpy()    # [K]
        # heatmap_np = torch.sigmoid(heatmap[0, 0]).cpu().numpy()  # [H, W]

         # interpolate heatmap to oringin size H×W
        heatmap_full = F.interpolate(
            heatmap,
            size=image.shape[-2:],  
            mode="bilinear",
            align_corners=False,
        )
        heatmap_np = torch.sigmoid(heatmap_full[0, 0]).cpu().numpy()
        return points, labels, heatmap_np,backbone_out