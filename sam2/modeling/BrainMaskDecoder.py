from typing import List, Optional, Tuple, Type
import torch
from torch import nn

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam2_utils import  MLP

class BrainMaskDecoder(MaskDecoder):
    """
    继承 SAM2 原 MaskDecoder，新增:
      - boundary_token: 专门负责边界预测的可学习 token
      - boundary_head: 把 boundary_token 输出 + upscaled features 变成软边界 alpha
      - boundary_refine: 【MODIFIED】现在额外接收 raw feat_s0（backbone直出、
        未经 upscaling 路径稀释的高分辨率特征），与 boundary_coarse 拼接后
        再做精修，让边界判别本身就能利用原始高频细节，而不仅是在最后的
        fusion 阶段才补救。
      - boundary_fusion: 将 mask_logits + boundary_logits + 高分辨率特征
        做空间可变的残差融合（zero-init，训练初期等价于不融合）

    SAM2 原有的 mask 输出路径完全保留，融合模块以残差形式介入，
    不破坏预训练能力。
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        # === BrainSAM ===
        use_boundary_head: bool = True,
        use_boundary_prior: bool = True,
        prior_dim: int = 320,
        use_boundary_fusion: bool = True,
        fusion_hidden_dim: int = 32,
        high_res_feat_dim: int = 32,
        use_raw_feat_in_boundary_refine: bool = True,
        zero_init_new_refine_channels: bool = True,
        # === BrainSAM ===
    ) -> None:  
        super().__init__(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=num_multimask_outputs,
            activation=activation,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
            use_high_res_features=use_high_res_features,
            iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
            dynamic_multimask_via_stability=dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta=dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh=dynamic_multimask_stability_thresh,
            pred_obj_scores=pred_obj_scores,
            pred_obj_scores_mlp=pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
        )

        self.use_boundary_head = use_boundary_head
        self.use_boundary_prior = use_boundary_prior
        self.prior_dim = prior_dim
        self._debug_boundary_alpha = None

        self.use_boundary_fusion = use_boundary_fusion and use_boundary_head
        self.use_high_res_features = use_high_res_features
        self.high_res_feat_dim = high_res_feat_dim

        self.use_raw_feat_in_boundary_refine = (
            use_raw_feat_in_boundary_refine and use_high_res_features
        )

        if self.use_boundary_head:
            self.boundary_token = nn.Embedding(1, transformer_dim)

            head_dim = transformer_dim // 8   # SAM2 默认 256 // 8 = 32

            self.boundary_hypernet = MLP(
                transformer_dim, transformer_dim, head_dim, 3
            )

            boundary_refine_in_ch = 1
            if self.use_raw_feat_in_boundary_refine:
                boundary_refine_in_ch += high_res_feat_dim

            self.boundary_refine = nn.Sequential(
                nn.Conv2d(boundary_refine_in_ch, head_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(head_dim, 1, kernel_size=3, padding=1),
            )

            if zero_init_new_refine_channels and self.use_raw_feat_in_boundary_refine:
                with torch.no_grad():
                    # weight shape: [head_dim, boundary_refine_in_ch, 3, 3]
                    self.boundary_refine[0].weight[:, 1:, :, :].zero_()


            if self.use_boundary_prior:
                self.boundary_film = nn.Linear(prior_dim, 2*transformer_dim)
                nn.init.normal_(self.boundary_film.weight, std=0.02)
                nn.init.zeros_(self.boundary_film.bias)

            if self.use_boundary_fusion:
                # 融合模块输入通道 = 1 (mask_logit) + 1 (boundary_logit) [+ high_res_feat_dim]
                fusion_in_ch = 2
                if self.use_high_res_features:
                    fusion_in_ch += high_res_feat_dim

                self.boundary_fusion = nn.Sequential(
                    nn.Conv2d(fusion_in_ch, fusion_hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(fusion_hidden_dim, fusion_hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(fusion_hidden_dim, 1, kernel_size=3, padding=1),
                )
                nn.init.zeros_(self.boundary_fusion[-1].weight)
                nn.init.zeros_(self.boundary_fusion[-1].bias)

            self._init_boundary_token()
            print("use BrainMaskDecoder: boundary_token initialized from mask_tokens mean.")
            print(f"use BrainMaskDecoder: boundary_refine in_ch={boundary_refine_in_ch} "
                  f"(use_raw_feat_in_boundary_refine={self.use_raw_feat_in_boundary_refine}).")
            if self.use_boundary_fusion:
                print(f"use BrainMaskDecoder: boundary_fusion enabled "
                      f"(high_res={self.use_high_res_features}, in_ch={fusion_in_ch}).")


    @torch.no_grad()
    def _init_boundary_token(self):
        """从 SAM2 预训练 mask_tokens 的均值初始化 boundary_token"""
        init_vec = self.mask_tokens.weight.mean(dim=0, keepdim=True)
        noise = torch.randn_like(init_vec) * 0.01
        self.boundary_token.weight.copy_(init_vec + noise)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        prior_embedding: Optional[torch.Tensor] = None, #*# boundary prior
    ):

        (masks, iou_pred, mask_tokens_out, object_score_logits,
        boundary_logits, boundary_alpha) = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            prior_embedding=prior_embedding
        )

        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]

        if self.use_boundary_head:
            return (masks, iou_pred, sam_tokens_out, object_score_logits,
                    boundary_logits, boundary_alpha)
        else:
            return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        prior_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
 
        s = 0

        if self.pred_obj_scores:
            tokens_list = [
                self.obj_score_token.weight,
                self.iou_token.weight,
                self.mask_tokens.weight,
            ]
            s = 1

        else:
            tokens_list = [self.iou_token.weight, self.mask_tokens.weight]


        output_tokens = torch.cat(tokens_list, dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )

        if self.use_boundary_head:
            B = sparse_prompt_embeddings.size(0)
            bt = self.boundary_token.weight.expand(B, -1)        # [B, dim]
            if self.use_boundary_prior and prior_embedding is not None:
                gamma, beta = self.boundary_film(prior_embedding).chunk(2, dim=-1)

                bt_before = bt.clone()
                bt = bt * (1 + gamma) + beta

                self._film_gamma_mean = gamma.abs().mean().item()
                self._film_beta_mean  = beta.abs().mean().item()
                self._film_delta_bt   = (bt - bt_before).abs().mean().item()

            bt = bt.unsqueeze(1)                                      # [B, 1, dim]
            output_tokens = torch.cat([output_tokens, bt], dim=1)  # [B, n_base+1, dim]

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)

        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        if self.use_boundary_head:
            boundary_token_out = hs[:, s + 1 + self.num_mask_tokens, :]   # [B, transformer_dim]

        src = src.transpose(1, 2).view(b, c, h, w)


        feat_s0_for_fusion = None

        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)
            feat_s0_for_fusion = feat_s0

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b_, c_, h_, w_ = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b_, c_, h_ * w_)).view(b_, -1, h_, w_)

        if self.use_boundary_head:
           boundary_kernel = self.boundary_hypernet(boundary_token_out)  # [B, c_]
           boundary_kernel = boundary_kernel.unsqueeze(1)  #[B, 1, c_]

           boundary_coarse = (
               boundary_kernel @ upscaled_embedding.view(b_, c_, h_ * w_)
                ).view(b_, 1, h_, w_)                                          # [B, 1, h_, w_]

           # >>> NEW: boundary_refine 之前，拼接 raw feat_s0
           # boundary_coarse 是从 upscaled_embedding（已被 mask 分支同款的
           # 语义特征稀释过）里算出来的粗略边界；这里把 backbone 直出、
           # 未经稀释的 feat_s0_for_fusion 拼进去，让精修阶段能直接看到
           # 原始高频细节，而不是只能依赖已经损失过细节的粗略估计。
           if self.use_raw_feat_in_boundary_refine and feat_s0_for_fusion is not None:
               assert feat_s0_for_fusion.shape[-2:] == boundary_coarse.shape[-2:], (
                   f"分辨率不匹配: feat_s0 {feat_s0_for_fusion.shape} "
                   f"vs boundary_coarse {boundary_coarse.shape}"
               )
               refine_in = torch.cat([boundary_coarse, feat_s0_for_fusion], dim=1)
           else:
               refine_in = boundary_coarse

           boundary_logits = self.boundary_refine(refine_in)
           # <<< END NEW
           boundary_alpha = torch.sigmoid(boundary_logits)

           # 空间可变残差融合（原有逻辑，不变）
           # 对每个 mask 通道（num_mask_tokens 个，通常 4 个：1 个 single + 3 个 multi）
           # 分别和 boundary_logits (+ 高分辨率特征) 拼接，过融合卷积得到残差，加回原 mask
           if self.use_boundary_fusion:
               fused_masks_list = []
               for i in range(self.num_mask_tokens):
                   mask_i = masks[:, i:i+1, :, :]                # [B, 1, h_, w_]
                   fuse_in = [mask_i, boundary_logits]
                   if self.use_high_res_features and feat_s0_for_fusion is not None:
                       # feat_s0_for_fusion 分辨率需与 mask_i / boundary_logits 一致
                       # (SAM2 官方 use_high_res_features=True 时 upscaled_embedding
                       #  与 feat_s0 分辨率天然对齐，不需要额外插值)
                       fuse_in.append(feat_s0_for_fusion)
                   fuse_in = torch.cat(fuse_in, dim=1)           # [B, 2(+high_res_feat_dim), h_, w_]
                   residual = self.boundary_fusion(fuse_in)      # [B, 1, h_, w_]
                   fused_masks_list.append(mask_i + residual)
               masks = torch.cat(fused_masks_list, dim=1)        # [B, num_mask_tokens, h_, w_]

        else:
            boundary_alpha = None
            boundary_logits = None

        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits, boundary_logits, boundary_alpha