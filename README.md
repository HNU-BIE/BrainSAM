# BrainSAM
<<<<<<< HEAD
=======

基于 [SAM 2](https://github.com/facebookresearch/sam2)（Meta, Apache-2.0）二次开发的跨物种、跨模态脑结构分割模型。核心改动包括模态/器官先验嵌入（Embedding Tree）、Hiera backbone 深层 block 上的 Mixture-of-Adapters（MoA）路由，以及边界优化监督模块。项目同时提供训练/评测脚本、一个 PyQt5 桌面标注工具（`app.py`）和一个 Web 版标注工具（`Brain_web/`）。

> 本仓库基于 Meta SAM 2 的 Apache-2.0 协议代码修改而来，见根目录 `LICENSE` 与 `NOTICE`。

## ⚠️ 使用前必读：部分入口脚本依赖未包含在本仓库中的模块

以下文件在代码里 `import` 了 `PNG_inference`、`Continuous_Inference`（或 `inference.base_inference` / `inference.click_correction`），但这几个模块**当前没有出现在本仓库的任何目录里**：

- `app.py`（PyQt5 桌面版入口）
- `Brain_web/server.py`（Web 版后端）
- `evaluate/evaluate.py`、`evaluate/evaluate_brainSAM.py`、`evaluate/correction_evaluate.py`（批量推理/评测脚本）

也就是说，这几个脚本目前 `import` 就会失败，无法直接运行。在推到公开仓库前，需要确认：

1. 这些模块是否本来就应该一起上传（如果是，把 `PNG_inference.py`、`Continuous_Inference.py` 和 `inference/` 这个包一起加进仓库）；
2. 还是这些模块暂时是私有/未整理好的代码，故意不公开——如果是这种情况，建议在这里注明清楚，让使用者知道只有 `training/`（训练流水线）和 `sam2/`（核心模型代码）、`metrics/`（评测指标）是可以独立运行的，其余几个入口脚本仅供参考。

`app.py` 第 64 行加载的 UI 文件 `roi-lynx_v9211307.ui` 在 `ui/` 目录里也不存在（目前只有 `v2`~`v5`），这一处按你的要求先保留原样，需要你自己确认后修正。

## 目录结构

```
BrainSAM/
├── app.py              # PyQt5 桌面标注工具入口（依赖上述未包含的模块）
├── Brain_web/          # Web 版标注工具（FastAPI），见其自带 README
├── data/               # 数据集封装（BrainSAM 自己的 + 移植自 SAM2 的 sam2_dataset/）
├── evaluate/           # 批量推理 + 指标计算脚本
├── metrics/            # Dice / Surface Dice / Surface Distance 等评测指标
├── sam2/               # 核心模型代码（基于 Meta SAM2 修改，含新增模块）
├── training/           # 训练入口、loss、优化器、trainer
├── ui/                 # Qt Designer 的 .ui / .qss 资源
├── utils/              # NIfTI 读取等工具函数
├── work_dir/           # 模型权重存放处（不提交进仓库，见 .gitignore）
├── requirements.txt    # 项目依赖
├── setup.py
├── LICENSE             # Apache-2.0（因基于 SAM2 修改而来）
└── NOTICE              # 第三方代码来源与修改说明
```

## 环境安装

```bash
conda create -n brainsam python=3.10
conda activate brainsam
pip install -r requirements.txt
# 或者：pip install -e .   （setup.py 里声明了 core / web / gui / train 几组 extras）
```

`sam2/csrc/connected_components.cu` 是一段 CUDA 扩展源码；如果你需要用到它对应的算子，需要自行编写/补全构建它的 `setup.py` 步骤（当前 `setup.py` 未包含 CUDA 扩展的编译配置）。

## 训练

训练入口是 `training/train.py`，基于 Hydra 读取 `sam2/configs/sam2.1_training/` 下的 yaml 配置（默认用的是 `sam2.1_hiera_b+BrainSAM.yaml`）。配置文件里 `img_folder` / `gt_folder` / `file_list_txt` 等数据路径原来写死指向作者本机路径，现已替换成 `/path/to/your/...` 占位符，**运行前请先改成你自己的数据路径**；`checkpoint.*.checkpoint_path` 和几个 `file_list_txt`（`training/assets/*.txt`）已经改成仓库内的相对路径，可以直接用。

```bash
python training/train.py --config sam2.1_training/sam2.1_hiera_b+BrainSAM.yaml
```

（具体的 config 路径写法请以 `training/train.py` 里 hydra 的 config-module 搜索路径为准，这里未做实际训练验证，建议先用 `--help` 或直接读一遍该文件确认。）

## 评测

`evaluate/config.py` 里的 `EXPERIMENTS` 列表和 `get_args()` 是研究阶段的实验配置，路径已替换成占位符，实际使用前需要按自己的实验重新填写。`evaluate/evaluate.py` 是一个独立的、基于 argparse 的批量推理+指标脚本（但同样依赖上面提到的缺失的 `inference` 模块）。

## Web 标注工具（Brain_web/）

见 `Brain_web/README.md`，里面已经写清楚了安装、运行方式和已知的未验证之处。

## 依赖来源与协议

`sam2/` 目录下大部分文件的版权头仍保留 Meta 的原始声明，因为这些文件是基于 [SAM 2](https://github.com/facebookresearch/sam2)（Apache-2.0）修改而来。完整协议见 `LICENSE`，修改说明见 `NOTICE`——上传前请把 `NOTICE` 里的 `[TODO: your name / lab / organization]` 替换成实际的作者/机构信息。
>>>>>>> 7b6dcc5 (init)
