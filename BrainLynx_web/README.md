# BrainSAM-LYNX Web

`roi-lynx.py` (PyQt5)的Web版骨架。目标场景:你/组内少数人在服务器上跑,内网访问。

已实现:
- 加载NIfTI文件、按 axis 切换、slider/滚轮翻页
- 点提示(左键前景/右键背景)+ 框提示,单帧SAM2推理,实时mask叠加
- 多对象(Object)管理
- Mask 传播(propagate),经WebSocket流式返回,前端实时刷新
- 保存为 NIfTI mask

**没有实现**(原PyQt程序里有,但这次先不做,按你选的范围裁剪掉了):
双向同时传播(to_two_sides)、JSON批处理(json_load/tree_start/Multi_gen_mask)、
config文件同步(sync_to_config_file)、Mask Generator自动模式、擦除工具、
图片文件夹(非NIfTI)模式。这些如果之后要加,在 `server.py` 里补对应路由、
在 `app.js` 里补交互即可,架构上是兼容的。

## 目录结构

```
brainsam_web/
  server.py          # FastAPI后端,唯一入口,配置写在文件顶部(没有CLI参数)
  requirements.txt
  static/
    index.html
    app.js
    style.css
```

## 安装 & 运行

```bash
conda activate BrainSAM        # 用你现有的环境,别新建
cd /path/to/brainsam_web
pip install -r requirements.txt

# 把 server.py 顶部的 CONFIG 改成你机器上的真实路径:
#   PROJECT_ROOT  -> roi-lynx.py 所在目录(能 import PNG_inference / Continuous_Inference / utils.nifti_reader 的地方)
#   SAM2_CONFIG / SAM2_CKPT -> 和原PyQt程序里写的一致
#   HOST/PORT     -> 内网跑的话 HOST="0.0.0.0" 就行,组内用 http://服务器IP:8000 访问

python server.py
```

浏览器打开 `http://<服务器IP>:8000`。

## 关键假设(和原模块对接的地方)

`server.py` 顶部的docstring里写了每个复用函数的假设签名,是照抄
`MainWindow` 里的调用方式(`PNG_inference.PNG_inference(...)`、
`CI.create_inference_state(...)`、`CI._Continuous_inference(...)`、
`nii_reader(...).get_slice_array/get_slices_num/align_to_me/save_seg`)。
如果你实际的模块签名和这个不完全一样(比如返回值多一个东西、参数顺序不同),
只需要改 `server.py` 里对应的几处调用,不用动前端。

多留意这几个我没法验证的地方(没有你的模块源码、也没有GPU环境跑通测试):

1. `PNG_inference.PNG_inference` 返回的 `masks` 的具体shape——我按原代码里
   `masks.squeeze(0) if masks.shape[0]!=1 else masks` 这段逻辑猜的
   `(num_obj, 1, H, W)`,在 `normalize_masks()` 里处理,如果实际不是这个
   形状,改这个函数就行。
2. `axis` 的语义(0/1/2 对应 冠状/轴位/矢状 还是别的)——原PyQt代码里两处
   注释互相矛盾(`NIFTIAxis`按钮组 vs `tree_start`里的注释),我保留成
   不透明的整数,前端按钮先写 "Axis 0/1/2",你加载完数据后确认一下哪个
   对应哪个切面,需要的话把 `index.html` 里按钮文字改成实际含义。
3. `CI.create_inference_state` 的返回值——原代码有的地方按元组解包
   (`self.inference_state, _ = ...`),有的地方直接赋值,我在
   `propagate_ws` 里做了兼容处理(`isinstance(..., tuple)`判断),但没法
   100%确认。
4. 会话(Session)状态目前存在内存里,重启服务会丢失,没做持久化——内网
   少数人用的场景应该够用;真要多人同时长时间标注不同文件,后面可以加个
   简单的sqlite落盘。

## 我没做的验证

这个容器里没有 GPU、没有你的 `sam2`/`PNG_inference`/`Continuous_Inference`/
`utils.nifti_reader` 这几个模块,所以我**只做了 Python/JS 语法检查**,没法
实际跑通端到端流程。建议你部署后先用一个小的 NIfTI 文件跑一遍:加载 →
点一个前景点看mask出不出来 → 设置起止帧跑一次propagate → 保存,任何一步
报错都大概率是上面第1-3条里提到的签名不匹配,把报错贴给我我再针对性改。
