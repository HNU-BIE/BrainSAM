"""
BrainSAM-LYNX Web Server
=========================
Web replacement for the PyQt5 ROI-LYNX tool. Reuses the existing project
modules (PNG_inference, Continuous_Inference, utils.nifti_reader, sam2)
exactly the way MainWindow (roi-lynx.py) called them, so it should drop
into the same conda env / project folder without touching those modules.

ASSUMPTIONS (adjust the small "ADAPTER" functions near the bottom if your
actual module signatures differ slightly from what MainWindow used):
  - PNG_inference.PNG_inference(model, image, ptsList, ptsTypeList, boxList)
      -> (masks, extra)   masks: ndarray, shape (num_obj, 1, H, W) or (1,H,W)
  - CI.create_inference_state(predictor, video_dir, prog_signal=None, axis=int)
      -> inference_state (or (inference_state, extra))
  - CI._Continuous_inference(predictor, inference_state=..., ptsList=...,
        ptsTypeList=..., box=..., nii_idx=int, start_frame=int,
        max_frame_num_to_track=int, reverse=bool, prog_signal=None)
      -> generator yielding (out_frame_idx, out_obj_ids, out_mask_logits)
  - nii_reader(file_path) with .get_slices_num(axis), .get_slice_array(axis, idx),
      .get_slices_shape(axis), .align_to_me(axis, mask_2d, idx, mask_array=None),
      .save_seg(stack_array, file_path)

No CLI args — edit the CONFIG block below and run:  python server.py
"""

import base64
import io
import os
import uuid
import logging
from copy import deepcopy
from typing import Optional

import numpy as np
import nibabel as nib
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("brainsam_web")

# ============================== CONFIG ======================================
HOST = "0.0.0.0"
PORT = 8000

CUDA_DEVICE = "cuda:0"                       # single A6000
PROJECT_ROOT = "/path/to/your/BrainSAM"   # so `import PNG_inference` etc. work
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT = "/path/to/your/checkpoints/sam2.1_hiera_base_plus.pt"

DEFAULT_AXIS = 1          # matches original NIFTIAxis default
MASK_ALPHA_DEFAULT = 128  # 0-255, opacity baked in ONLY as a fallback; the
                           # frontend composites opacity live so this is unused
                           # by default, kept for completeness.
# =============================================================================

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, PROJECT_ROOT)

# --- project-specific imports (must live in PROJECT_ROOT) -------------------
import PNG_inference                              # noqa: E402
import inference.Continuous_Inference as CI                  # noqa: E402
from sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402
from utils.nifti_reader import NIFTI_READER as nii_reader          # noqa: E402

app = FastAPI(title="BrainSAM-LYNX ")

# ============================== MODEL (singleton, loaded once) ==============
log.info("Loading SAM2 image model...")
SAM2_MODEL = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=CUDA_DEVICE)
log.info("Loading SAM2 video predictor...")
V_SAM2_MODEL = build_sam2_video_predictor(config_file=SAM2_CONFIG, ckpt_path=SAM2_CKPT, device=CUDA_DEVICE)
log.info("Models loaded.")


# ============================== SESSION STATE ================================
class Session:
    """One browser tab == one Session. Mirrors the mutable state that used
    to live directly on MainWindow (self.axis_click, self.boxList, etc.)."""

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.source_path: Optional[str] = None
        self.file_type: Optional[int] = None      # 1 = nifti, 2 = image folder
        self.nii_img = None
        self.axis: int = DEFAULT_AXIS
        self.axis_len: int = 0
        self.current_idx: int = 0
        self.current_slice_uint8: Optional[np.ndarray] = None  # normalized, fed to PNG_inference

        # prompts, indexed per-object like the original
        self.axis_click: list[list[list[float]]] = [[]]
        self.type_click: list[list[int]] = [[]]
        self.boxList: list[Optional[list[float]]] = [None]
        self.obj_num: int = 0

        # propagate state
        self.inference_state = None
        self.start_page: Optional[int] = None
        self.end_page: Optional[int] = None

        # results: {frame_idx: {obj_id: np.ndarray[H,W] bool}}
        self.video_segments: dict[int, dict[int, np.ndarray]] = {}
        self.last_mask_path: Optional[str] = None   # set after a successful /api/save

    def ensure_obj_slots(self, obj_id: int):
        while len(self.axis_click) <= obj_id:
            self.axis_click.append([])
            self.type_click.append([])
            self.boxList.append(None)


SESSIONS: dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    s = SESSIONS.get(session_id)
    if s is None:
        raise ValueError(f"Unknown session_id {session_id}")
    return s


# ============================== HELPERS ======================================
def normalize_slice_uint8(slice_img: np.ndarray) -> np.ndarray:
    """Raw NIfTI slice (float64) -> normalized uint8, matching the original
    show_slice() normalization. This is the array that must be fed to
    PNG_inference (mirrors self.slice_img_p in the PyQt version) — SAM2's
    image encoder is float32 and chokes on raw float64 voxel values."""
    lo, hi = np.min(slice_img), np.max(slice_img)
    if hi - lo < 1e-6:
        return np.zeros_like(slice_img, dtype=np.uint8)
    return ((slice_img - lo) / (hi - lo) * 255).astype(np.uint8)


def slice_to_png_b64(slice_img_uint8: np.ndarray) -> str:
    """uint8 slice -> base64 PNG."""
    buf = io.BytesIO()
    Image.fromarray(slice_img_uint8, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def mask_to_png_b64(mask: np.ndarray) -> str:
    """Binary mask -> base64 grayscale PNG (0/255). Frontend colors + composites."""
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(m, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def normalize_masks(masks: np.ndarray) -> list[np.ndarray]:
    """Reproduce the squeeze logic scattered through the original mouse handlers:
    returns a list of 2D uint8 masks, one per object, in object order."""
    masks = np.asarray(masks)
    if masks.ndim == 4:            # (num_obj, 1, H, W)
        out = []
        for i in range(masks.shape[0]):
            m = masks[i]
            if m.shape[0] == 1:
                m = m.squeeze(0)
            out.append(m.astype(np.uint8))
        return out
    if masks.ndim == 3:            # (1, H, W) single object
        return [masks.squeeze(0).astype(np.uint8)]
    if masks.ndim == 2:            # (H, W)
        return [masks.astype(np.uint8)]
    raise ValueError(f"Unexpected mask shape {masks.shape}")


def run_point_box_inference(s: Session) -> dict[int, np.ndarray]:
    """Equivalent of PNG_inference.PNG_inference call in mouse_press/mouse_release."""
    if s.file_type == 1:
        if s.current_slice_uint8 is None:
            raise ValueError("no slice loaded yet")
        image = s.current_slice_uint8
    else:
        raise ValueError("Image-folder mode not wired up in this endpoint")

    masks, _ = PNG_inference.PNG_inference(
        SAM2_MODEL, image, s.axis_click, s.type_click, s.boxList
    )
    mask_list = normalize_masks(masks)
    result = {}
    for obj_id, m in enumerate(mask_list):
        result[obj_id] = m
    # stash into video_segments so save_mask / re-render pick it up
    s.video_segments[s.current_idx] = {
        obj_id: (m > 0).astype(np.uint8) for obj_id, m in result.items()
    }
    return result


def save_segmentation_nifti(mask_path: str, orig_path: str, output_path: str) -> str:
    """
    读取 mask 和原图，把mask应用到原图上 (原图 * 二值mask)，
    保存为nifti，affine/header跟原图保持一致。

    参数:
        mask_path      mask的nifti路径 (二值，>0视为前景)
        orig_path      原始灰度图的nifti路径
        output_path    输出文件路径 (比如 xxx_seg.nii.gz)
    """
    orig_img = nib.load(orig_path)
    mask_img = nib.load(mask_path)

    orig_data = orig_img.get_fdata()
    mask_data = mask_img.get_fdata()

    if orig_data.shape != mask_data.shape:
        raise ValueError(
            f"orig和mask形状不一致: orig={orig_data.shape}, mask={mask_data.shape}. "
            f"两者需要在同一空间下才能相乘。"
        )

    mask_bin = (mask_data > 0.5).astype(orig_data.dtype)
    seg_data = orig_data * mask_bin

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    nib.save(nib.Nifti1Image(seg_data, orig_img.affine, orig_img.header), output_path)
    # print(f"已保存: {output_path}")
    return output_path


# ============================== API MODELS ===================================
class LoadRequest(BaseModel):
    file_path: str
    axis: int = DEFAULT_AXIS


class PointRequest(BaseModel):
    session_id: str
    obj_id: int
    x: float
    y: float
    label: int   # 1 = positive (foreground), 0 = negative (background)


class BoxRequest(BaseModel):
    session_id: str
    obj_id: int
    x1: float
    y1: float
    x2: float
    y2: float


class ClearRequest(BaseModel):
    session_id: str
    obj_id: Optional[int] = None   # None = clear all objects


class SliceRequest(BaseModel):
    session_id: str
    idx: int


class SaveRequest(BaseModel):
    session_id: str
    file_path: str
    force: bool = False   # if True, fill any frame with no mask as an all-zero mask (like the original "F-save")


class SaveSegmentationRequest(BaseModel):
    session_id: str
    output_path: str
    mask_path: Optional[str] = None   # defaults to the path from the last /api/save call


class InitPropagateRequest(BaseModel):
    session_id: str


# ============================== ROUTES: SESSION / LOAD ========================
@app.post("/api/session")
def create_session():
    s = Session()
    SESSIONS[s.id] = s
    return {"session_id": s.id}


@app.get("/api/list_nifti")
def list_nifti(dir_path: str):
    if not os.path.isdir(dir_path):
        return JSONResponse({"error": f"not a directory: {dir_path}"}, status_code=404)

    results = []
    MAX_RESULTS = 2000
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith((".nii", ".nii.gz")):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, dir_path)
                results.append({"path": full, "rel": rel})
                if len(results) >= MAX_RESULTS:
                    break
        if len(results) >= MAX_RESULTS:
            break
    results.sort(key=lambda x: x["rel"])
    return {"files": results, "count": len(results), "truncated": len(results) >= MAX_RESULTS}


@app.post("/api/load")
def load_file(req: LoadRequest, session_id: str):
    s = get_session(session_id)
    if not os.path.exists(req.file_path):
        return JSONResponse({"error": f"path not found: {req.file_path}"}, status_code=404)

    s.source_path = req.file_path
    s.axis = req.axis
    s.current_idx = 0
    s.video_segments = {}
    s.inference_state = None
    s.axis_click = [[]]
    s.type_click = [[]]
    s.boxList = [None]
    s.obj_num = 0

    if req.file_path.lower().endswith((".nii", ".nii.gz")):
        s.file_type = 1
        s.nii_img = nii_reader(req.file_path)
        s.axis_len = s.nii_img.get_slices_num(s.axis)
    else:
        return JSONResponse({"error": "only NIfTI files are supported by this endpoint"}, status_code=400)

    slice_img = s.nii_img.get_slice_array(s.axis, s.current_idx)[0]
    s.current_slice_uint8 = normalize_slice_uint8(slice_img)
    return {
        "axis_len": s.axis_len,
        "shape": list(s.nii_img.get_slices_shape(s.axis)),
        "slice_png": slice_to_png_b64(s.current_slice_uint8),
        "idx": s.current_idx,
    }


@app.post("/api/set_axis")
def set_axis(session_id: str, axis: int):
    s = get_session(session_id)
    s.axis = axis
    s.current_idx = 0
    s.axis_len = s.nii_img.get_slices_num(axis)
    slice_img = s.nii_img.get_slice_array(s.axis, s.current_idx)[0]
    s.current_slice_uint8 = normalize_slice_uint8(slice_img)
    return {"axis_len": s.axis_len, "slice_png": slice_to_png_b64(s.current_slice_uint8), "idx": s.current_idx}


@app.post("/api/slice")
def get_slice(req: SliceRequest):
    s = get_session(req.session_id)
    s.current_idx = max(0, min(req.idx, s.axis_len - 1))
    slice_img = s.nii_img.get_slice_array(s.axis, s.current_idx)[0]
    s.current_slice_uint8 = normalize_slice_uint8(slice_img)
    resp = {"idx": s.current_idx, "slice_png": slice_to_png_b64(s.current_slice_uint8), "masks": {}}
    # replay any existing masks for this frame
    if s.current_idx in s.video_segments:
        resp["masks"] = {
            str(obj_id): mask_to_png_b64(m) for obj_id, m in s.video_segments[s.current_idx].items()
        }
    return resp


# ============================== ROUTES: PROMPTS ===============================
@app.post("/api/point")
def add_point(req: PointRequest):
    s = get_session(req.session_id)
    s.ensure_obj_slots(req.obj_id)
    s.axis_click[req.obj_id].append([req.x, req.y])
    s.type_click[req.obj_id].append(req.label)
    try:
        masks = run_point_box_inference(s)
    except Exception as e:
        log.exception("inference failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"masks": {str(k): mask_to_png_b64(v) for k, v in masks.items()}}


@app.post("/api/box")
def set_box(req: BoxRequest):
    s = get_session(req.session_id)
    s.ensure_obj_slots(req.obj_id)
    x1, x2 = sorted([req.x1, req.x2])
    y1, y2 = sorted([req.y1, req.y2])
    s.boxList[req.obj_id] = [x1, y1, x2, y2]
    try:
        masks = run_point_box_inference(s)
    except Exception as e:
        log.exception("inference failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"masks": {str(k): mask_to_png_b64(v) for k, v in masks.items()}}


@app.post("/api/clear_prompts")
def clear_prompts(req: ClearRequest):
    s = get_session(req.session_id)
    if req.obj_id is None:
        s.axis_click = [[]]
        s.type_click = [[]]
        s.boxList = [None]
        s.obj_num = 0
    else:
        s.ensure_obj_slots(req.obj_id)
        s.axis_click[req.obj_id] = []
        s.type_click[req.obj_id] = []
        s.boxList[req.obj_id] = None
    return {"ok": True}


@app.post("/api/set_start_page")
def set_start_page(session_id: str):
    s = get_session(session_id)
    s.start_page = s.current_idx
    return {"start_page": s.start_page}


@app.post("/api/set_end_page")
def set_end_page(session_id: str):
    s = get_session(session_id)
    s.end_page = s.current_idx
    return {"end_page": s.end_page}


# ============================== ROUTES: PROPAGATE (WebSocket) ================
@app.websocket("/ws/propagate/{session_id}")
async def propagate_ws(ws: WebSocket, session_id: str):
    await ws.accept()
    try:
        s = get_session(session_id)
    except ValueError as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return

    try:
        cfg = await ws.receive_json()
        start_frame = int(cfg.get("start_frame", s.start_page if s.start_page is not None else s.current_idx))
        end_frame = int(cfg.get("end_frame", s.end_page if s.end_page is not None else s.axis_len - 1))
        reverse = bool(end_frame < start_frame) if cfg.get("reverse") is None else bool(cfg["reverse"])
        max_frames = abs(end_frame - start_frame)

        await ws.send_json({"type": "status", "message": "computing embedding..."})

        def prog(v):
            # best-effort progress callback; ignore if the loop isn't around
            pass

        if s.inference_state is None:
            s.inference_state = CI.create_inference_state(
                V_SAM2_MODEL, s.source_path, prog_signal=prog, axis=s.axis
            )
            if isinstance(s.inference_state, tuple):
                s.inference_state = s.inference_state[0]

        await ws.send_json({"type": "status", "message": "propagating..."})

        gen = CI._Continuous_inference(
            V_SAM2_MODEL,
            inference_state=s.inference_state,
            ptsList=s.axis_click,
            ptsTypeList=s.type_click,
            box=s.boxList,
            nii_idx=start_frame,
            start_frame=start_frame,
            max_frame_num_to_track=max_frames,
            reverse=reverse,
        )

        # _Continuous_inference's FIRST yield is just the raw mask array for the
        # start frame (from add_new_points_or_box), not a (frame_idx, obj_ids,
        # logits) triple — mirrors the `next(gen)` call the original PyQt code
        # always did before iterating. Consume it and push it as frame start_frame.
        first_mask_logits = next(gen)
        first_masks_np = np.asarray(first_mask_logits)
        first_frame_masks = {}
        if first_masks_np.ndim == 4:       # (num_obj, 1, H, W)
            for i in range(first_masks_np.shape[0]):
                m = first_masks_np[i]
                if m.shape[0] == 1:
                    m = m.squeeze(0)
                first_frame_masks[i + 1] = m.astype(np.uint8)  # obj ids are 1-based, see ann_obj_id
        elif first_masks_np.ndim == 3:      # (1, H, W)
            first_frame_masks[1] = first_masks_np.squeeze(0).astype(np.uint8)
        elif first_masks_np.ndim == 2:      # (H, W)
            first_frame_masks[1] = first_masks_np.astype(np.uint8)
        s.video_segments[start_frame] = first_frame_masks
        await ws.send_json({
            "type": "frame",
            "frame_idx": start_frame,
            "masks": {str(k): mask_to_png_b64(v) for k, v in first_frame_masks.items()},
        })

        for out_frame_idx, out_obj_ids, out_mask_logits in gen:
            frame_masks = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                m = (out_mask_logits[i] > 0.0).cpu().numpy()
                if m.ndim == 3:
                    m = m.squeeze(0)
                frame_masks[out_obj_id] = m.astype(np.uint8)
            s.video_segments[out_frame_idx] = frame_masks

            await ws.send_json({
                "type": "frame",
                "frame_idx": out_frame_idx,
                "masks": {str(k): mask_to_png_b64(v) for k, v in frame_masks.items()},
            })

        await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        log.info("client disconnected during propagate")
    except Exception as e:
        log.exception("propagate failed")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ============================== ROUTES: SAVE ==================================
@app.post("/api/save")
def save_mask(req: SaveRequest):
    s = get_session(req.session_id)
    if s.nii_img is None:
        return JSONResponse({"error": "no file loaded"}, status_code=400)

    stack_array = None
    missing = []
    filled = []
    for idx in range(s.axis_len):
        if idx not in s.video_segments:
            if req.force:
                # mirrors the original "F-save" (Force-save) behavior: create
                # an empty mask for any frame that has no annotation at all,
                # rather than blocking the save.
                mask_2d = np.zeros(s.nii_img.get_slices_shape(s.axis), dtype=np.uint8)
                filled.append(idx)
            else:
                missing.append(idx)
                continue
        else:
            mask_2d = np.zeros(s.nii_img.get_slices_shape(s.axis), dtype=np.uint8)
            for obj_id, m in s.video_segments[idx].items():
                mask_2d |= (m != 0).astype(np.uint8)
        stack_array = s.nii_img.align_to_me(s.axis, mask_2d, idx, mask_array=stack_array)

    if missing and not req.force:
        return JSONResponse(
            {"warning": f"{len(missing)} frame(s) have no mask and were skipped: {missing}. "
                        f"Tick 'force save' to fill them as empty and save anyway.",
             "missing": missing,
             "saved": False},
            status_code=200,
        )

    s.nii_img.save_seg(stack_array, req.file_path)
    s.last_mask_path = req.file_path
    resp = {"saved": True, "path": req.file_path}
    if filled:
        resp["filled_empty"] = filled
        resp["warning"] = f"{len(filled)} frame(s) had no mask and were saved as empty: {filled}"
    return resp


@app.post("/api/save_segmentation")
def save_segmentation(req: SaveSegmentationRequest):
    s = get_session(req.session_id)
    if s.source_path is None:
        return JSONResponse({"error": "no file loaded"}, status_code=400)
    mask_path = req.mask_path or s.last_mask_path
    if not mask_path:
        return JSONResponse(
            {"error": "no mask file to use — save the mask first (or pass mask_path explicitly)"},
            status_code=400,
        )
    if not os.path.exists(mask_path):
        return JSONResponse({"error": f"mask file not found: {mask_path}"}, status_code=404)
    try:
        out = save_segmentation_nifti(mask_path, s.source_path, req.output_path)
    except Exception as e:
        log.exception("save_segmentation failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"saved": True, "path": out, "mask_path": mask_path}


# ============================== STATIC FRONTEND ================================
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)