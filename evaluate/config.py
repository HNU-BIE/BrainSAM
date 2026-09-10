from types import SimpleNamespace

__all__ = ["get_args", "EXPERIMENTS"]

# EXPERIMENTS = [
# dict(modal_id=0, organ_ids=[2, 3, 6],     species="human",  save_dir="/path/to/your/data/BrainSAM_output/Exp0_re/human",  sheet="human_exp0_re"),
# dict(modal_id=0, organ_ids=[1, 2, 17],    species="monkey", save_dir="/path/to/your/data/BrainSAM_output/Exp0_re/monkey", sheet="Monkey_exp0_re"),
# dict(modal_id=1, organ_ids=[0, 0, 13],    species="ben",    save_dir="/path/to/your/data/BrainSAM_output/Exp0_re/ben",    sheet="ben_exp0_re"),
# dict(modal_id=1, organ_ids=[1, 4, 10],    species="rabbit", save_dir="/path/to/your/data/BrainSAM_output/Exp0_re/rabbit", sheet="rabbit_exp0_re"),
# ]
EXPERIMENTS = [
# ===== human_other 系列，按实际模态分组（已排除 T1：qin_t1/ixi_t1/fsm_t1/asl_t1） =====
    # T2 组（qin_t2 + ixi_t2 + fsm_t2，sheet 相同，save_dir 各自独立）
    dict(modal_id=1, organ_ids=[2, 3, 6], species="human_other_qin_t2", save_dir="/path/to/your/data/BrainSAM_output/human_other/qin_t2", sheet="human_other_t2_exp0_re"),
    dict(modal_id=1, organ_ids=[2, 3, 6], species="human_other_ixi_t2", save_dir="/path/to/your/data/BrainSAM_output/human_other/ixi_t2", sheet="human_other_t2_exp0_re"),
    dict(modal_id=1, organ_ids=[2, 3, 6], species="human_other_fsm_t2", save_dir="/path/to/your/data/BrainSAM_output/human_other/fsm_t2", sheet="human_other_t2_exp0_re"),

    # PD 组（ixi_pd + fsm_pd）
    dict(modal_id=1, organ_ids=[2, 3, 6], species="human_other_ixi_pd", save_dir="/path/to/your/data/BrainSAM_output/human_other/ixi_pd", sheet="human_other_pd_exp0_re"),
    dict(modal_id=1, organ_ids=[2, 3, 6], species="human_other_fsm_pd", save_dir="/path/to/your/data/BrainSAM_output/human_other/fsm_pd", sheet="human_other_pd_exp0_re"),

    # ASL/EPI（单独一个）
    dict(modal_id=1, organ_ids=[2, 3, 6], species="human_other_asl_epi", save_dir="/path/to/your/data/BrainSAM_output/human_other/asl_epi", sheet="human_other_asl_epi_exp0_re"),

    # MRA（单独一个）
    dict(modal_id=1, organ_ids=[2, 3, 6], species="human_other_ixi_mra", save_dir="/path/to/your/data/BrainSAM_output/human_other/ixi_mra", sheet="human_other_mra_exp0_re"),

    # DWI（单独一个）
    dict(modal_id=1, organ_ids=[2, 3, 6], species="human_other_ixi_dwi", save_dir="/path/to/your/data/BrainSAM_output/human_other/ixi_dwi", sheet="human_other_dwi_exp0_re"),

    # qT1（单独一个）
    dict(modal_id=1, organ_ids=[2, 3, 6], species="human_other_fsm_qt1", save_dir="/path/to/your/data/BrainSAM_output/human_other/fsm_qt1", sheet="human_other_qt1_exp0_re"),

    # FLAIR（单独一个）
    dict(modal_id=1, organ_ids=[2, 3, 6], species="human_other_qin_flair", save_dir="/path/to/your/data/BrainSAM_output/human_other/qin_flair", sheet="human_other_flair_exp0_re"),
]
def get_args(exp_idx: int = 0):
    """Return config for BrainSAM batch evaluation."""

    args = SimpleNamespace()



    # ------------------------
    # Data
    # ------------------------
    args.config_json  = "/path/to/your/data/raw/config/config.json"

    # ------------------------
    # Model
    # ------------------------
    args.model_cfg = (
        "./sam2/configs/sam2.1/"
        "brainsam2.1_hiera_b+.yaml"
    )
    # args.model_cfg = (
    #     "./sam2/configs/sam2.1/"
    #     "sam2.1_hiera_b+_512.yaml"
    # )

    # EXP0
    # args.ckpt = (
    #     "./sam2_logs/configs/sam2.1_training/"
    #     "sam2.1_hiera_b+BrainSAM.yaml/checkpoints/checkpoint.pt"
    # )
    # EXP0_re
    args.ckpt = (
        "./work_dir/checkpoint_prior_boudary_and_MoE.pt"
    )
    # EXP1
    # args.ckpt = (
    #     "./work_dir/checkpoint_with_prior_boundary_module.pt"
    # )
    #EXP3
    # args.ckpt = (
    #     "./work_dir/checkpoint_full_finetuing.pt"
    # )
    args.device = "cuda:0"

    # ------------------------
    # Inference
    # ------------------------
    args.axis        = 1  # 0,1,2 分别对应 轴向,冠状面,矢状面

    args.auto_prompt = 'bbox'  #可以填True或'pts'或False，True会使用轻量提示生成器，'pts'会在mask上随机取点；False则不使用点提示

    # ------------------------
    # Output
    # -----------------------
    #moneky:[0,1,2,17],human:[0,2,3,6],mouse:[1,0,0,13]
    exp = EXPERIMENTS[exp_idx]
    args.modal_id  = exp["modal_id"]
    args.organ_ids = exp["organ_ids"]
    args.species   = exp["species"]
    args.save_dir  = exp["save_dir"]
    args.sheet     = exp["sheet"]
    args.xlsx      = "/path/to/your/data/BrainSAM_output/human_other/results.xlsx"
    args.log_dir  = None   # None → save_dir

    # ------------------------
    # Runtime
    # ------------------------
    args.save_interval = 1   # 每处理多少条保存一次 checkpoint

    return args


if __name__ == "__main__":
    for i, exp in enumerate(EXPERIMENTS):
        print(f"[{i}] {exp}")