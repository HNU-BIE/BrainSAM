import os
# Must be set before importing torch / cv2 / numpy / SimpleITK (or anything else
# that links OpenMP) — otherwise Windows raises "OMP: Error #15: Initializing
# libiomp5md.dll, but found libiomp5md.dll already initialized." This has to be
# the very first thing in the file; moving it below those imports (as happened
# once already) silently brings the crash back.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import json
import re
from copy import deepcopy

import cv2
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem,
    QFileDialog, QHBoxLayout, QWidget, QLabel, QTextEdit
)
from PyQt5.QtWidgets import QGraphicsEllipseItem, QButtonGroup
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QPainter, QImage, QColor, QPen, QBrush, QTransform, QStandardItemModel, \
    QStandardItem
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QModelIndex, QObject, \
    QRectF
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch


try:
    import Inference.Continuous_Inference as CI
except ImportError as e:
    print(f"[BrainSAM] Inference.Continuous_Inference 不可用，视频/传播推理相关功能将无法使用: {e}")
    CI = None
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from utils.nifti_reader import NIFTI_READER as nii_reader
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
mask_colors=[
    [173,216,230,1],
    [245,245,220,1],
    [255,228,225,1],
    [189,252,201,1],
]

def np2pixmap(np_img):
    height, width, channel = np_img.shape
    if channel == 3:
        bytesPerLine = 3 * width
        qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    else:
        bytesPerLine = 4 * width
        qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qImg)

@staticmethod
def parse_coords(s:str):

    try:
        inner = s[1:-1]
        s = inner.split(',')
        if len(s)==2:
            x,y = inner.split(",")
            return float(x),float(y)
        else:
            x1,y1,x2,y2 = inner.split(",")
            return float(x1),float(y1),float(x2),float(y2)
    except:
        return None,None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(r"ui/index.ui", self)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setStyleSheet("background-color: black;")
        self.graphicsView_hovered = False
        self.graphicsView.setRenderHint(QPainter.Antialiasing)
        self.graphicsView.setRenderHint(QPainter.SmoothPixmapTransform)

        self.checkbox_auto_Embedding.setChecked(False)

        #prompt group
        self.buttongroup = QButtonGroup()
        self.buttongroup.addButton(self.radioButton_Pts, 1)
        self.buttongroup.addButton(self.radioButton_Box, 2)
        self.buttongroup.addButton(self.radioButton_None, 3)
        self.buttongroup.buttonClicked[int].connect(self.Model)

        #orientation group
        self.buttongroup2 = QButtonGroup()
        self.buttongroup2.addButton(self.ButtonSag, 0)
        self.buttongroup2.addButton(self.ButtonCor, 1)
        self.buttongroup2.addButton(self.ButtonAxi, 2)
        self.buttongroup2.buttonClicked[int].connect(self.NIFTIAxis)

        #magnify group
        self.buttongroup3 = QButtonGroup()
        self.buttongroup3.addButton(self.Button_Magnify_1x, 1)
        self.buttongroup3.addButton(self.Button_Magnify_2x, 2)
        self.buttongroup3.addButton(self.Button_Magnify_4x, 4)
        self.buttongroup3.addButton(self.Button_Magnify_05x, 5)
        self.buttongroup3.buttonClicked[int].connect(self.NIFTIMagnify)

        #progressBar
        self.bar_embedding_calculate.hide()
        self.bar_embedding_calculate.setValue(0)

        #nifti file initial
        self.nii_img = None
        self.nii_size = None
        self.slice_img_p = None
        self.slice_img = None
        self.axis = None
        self.axis_nii_len = None
        self.current_slice_idx = 0
        self.reverse = False

        #image folder initial
        self.image_files = []
        self.current_index = 0
        self.folder_path = None
        self.image_length = 0

        #image initial
        self.image = None

        #jason file initial
        self.cfg = None
        self.conf_types = None
        self.conf_points = None
        self.obj_num = 0
        self.cfg_Tree = QStandardItemModel()
        self.treeView_file_list.setModel(self.cfg_Tree)
        self.cfg_Tree_pt = QStandardItemModel(0, 3)
        self.cfg_Tree_pt.setHorizontalHeaderLabels(["num", "Type", "coord"])
        self.treeView_pt.setModel(self.cfg_Tree_pt)
        self.treeView_pt.selectionModel().selectionChanged.connect(self.select_point_use_circle)
        self.treeView_pt.setFocusPolicy(Qt.NoFocus)

        # SAM Prompt
        self.modelSelect = None
        self.half_point_size = 5  # radius of bbox starting and ending points
        self.point_size = self.half_point_size * 2
        self.is_mouse_down = False
        self.end_point = None
        self.start_pos = (None, None)
        self.end_pos = (None, None)
        self.rect = None
        self.rect_temp = None
        self.nii_data = None
        self.input_point = None
        self.input_label = None
        self.axis_click = []
        self.type_click = []
        self.ptsList = [[]]
        self.ptsTypeList = []
        self.boxList = []
        self.prompt_Preserve = []
        self.continue_inference = False
        self._last_circle = {}
        self.point_set = {}
        self.box_set = {}

        # embedding
        self.re_inintial_embedding = False
        self.inference_state = {}

        # mask 参数
        self.mask_preserve = []
        self.mask_temp = []
        self.mask_group = {}
        self.video_segments = {}
        self.start_page = None
        self.end_page = None
        self.mask_item = None
        self.maskOpacity = 0.5
        self.magnify = 1

        self.maskOpacity_temp = None
        self.is_force = False
        self.re_inintial = True
        self.waiting_for_confirmation= False

        self.transpose_orders = [
            (2, 0, 1),  # 原始顺序
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (0, 1, 2),
            (2, 1, 0)
        ]
        self.flip_states = [
            None,
            0,
            1,
            2
        ]
        self.transpose_orders_index = None
        self.current_flip_idx = 0
        self.folder_path_folder = None
        self.img_folder_path=None
        # Tools初始化
        self.is_erasing = False
        self.checkBox_Sync_to_Config_File.setEnabled(False)

        #Mask Generator
        self.Button_MG_prompt.setCheckable(True)  # MG--->mask generator
        self.Button_MG_prompt.setAutoRepeat(False)
        self.Button_MG_prompt.clicked.connect(self._MG_prompt)

        #page1
        self.Button_SourceFiles.clicked.connect(lambda: self.load_source_file())
        self.lineEdit_SourceFiles.returnPressed.connect(self.on_source_lineedit_enter)
        self.Button_SourceFiles_folder.clicked.connect(self.load_images_from_folder)
        self.Slider_SelectFiles.valueChanged.connect(self.update_image_from_slider)
        self.Button_conf.clicked.connect(self.json_load)
        self.Button_loadMask.clicked.connect(lambda res: self.load_mask_file(self.nii_path, self.axis))
        self.treeView_file_list.doubleClicked.connect(self.tree_doubleClicked)
        self.Button_pro_files.clicked.connect(self.tree_start)

        #page2
        self.Slider_mask_opacity.valueChanged.connect(self.set_mask_opacity)
        self.CheckBox_mask_opacity.toggled.connect(self.Slider_mask_opacity.setEnabled)
        self.CheckBox_mask_opacity.stateChanged.connect(self.show_and_hide_mask)
        self.Slider_mask_opacity.setValue(int(0.5 * 10))
        self.Slider_mask_opacity.setRange(0, 10)

        #page3
        self.CLB_start_page.clicked.connect(self.go_to_start_page)
        self.CLB_end_page.clicked.connect(self.go_to_end_page)
        self.CLB_skip_page.clicked.connect(self.go_to_any_page)
        self.Button_start_page.clicked.connect(self.set_start_page)
        self.Button_end_page.clicked.connect(self.set_end_page)
        self.Button_TryIt.clicked.connect(self.propagateMask)
        self.ButtonPropagate.clicked.connect(self.on_button_propagate_clicked)
        self.Button_Stop.clicked.connect(self._thread_stop)
        self.Button_Continute.clicked.connect(self._thread_continue)
        self.Button_Save_generation.clicked.connect(self.save_mask)
        # self.Button_change_trans.clicked.connect(self.change_transpose_order)
        # self.Button_change_flip.clicked.connect(self.change_flip_state)

        #status bar
        statusWidget = QWidget()
        layout_statuBar = QHBoxLayout()
        layout_statuBar.setContentsMargins(0, 0, 0, 0)
        self.label_model = QLabel('Prompt: <b>Not allowed</b>')
        self.label_embedding = QLabel('Status: <b>No file</b>')
        self.label_device = QLabel(f"Device: <b>{self._pick_device()}</b>")
        self.label_embedding.setTextFormat(Qt.RichText)
        self.label_embedding.setText('Status: <b>No file</b>')
        layout_statuBar.addWidget(QLabel(" | "))
        layout_statuBar.addWidget(self.label_model)
        layout_statuBar.addWidget(QLabel(" | "))
        layout_statuBar.addWidget(self.label_embedding)
        layout_statuBar.addWidget(QLabel(" | "))
        layout_statuBar.addWidget(self.label_device)
        layout_statuBar.addWidget(QLabel(" | "))
        self.statusBar.showMessage("Load the image file to start.Support NIFTI, single picture or a set of pictures.")
        statusWidget.setLayout(layout_statuBar)
        self.statusBar.addPermanentWidget(statusWidget)
        self.setStyleSheet(open('ui/statusbar_styles.qss', encoding='utf-8').read())

        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release
        self.file_type = None#1:nifti,2:image folder,3:single image
        self.thread = None



        # build sam —— 不在启动时自动加载权重。改成手动:在"Video:"/"PNG:"两行里选好
        # checkpoint 路径，再点 init_v / init_p 才真正 build 对应的模型；不点就是 None。
        self.sam2_model = None
        self.v_sam2_model = None
        self.Button_choose_v_ck_path.clicked.connect(self.choose_v_ckpt_path)
        self.Button_choose_i_ck_path.clicked.connect(self.choose_p_ckpt_path)
        self.Button_init_v.clicked.connect(self.init_video_predictor)
        self.Button_init_p.clicked.connect(self.init_image_predictor)

        #initial a file with assets/MRI_brain.nii.gz
        self.load_source_file("./assets/human_T1.nii.gz")
        # self.Slider_SelectFiles.setValue(127)
        self.comboBox.currentTextChanged.connect(self.switch_other_obj)
        layout_statuBar = QHBoxLayout()
        layout_statuBar.setContentsMargins(0, 0, 0, 0)

    def _default_ckpt_dir(self):
        """选择checkpoint文件对话框的默认目录：工作目录下的 work_dir。"""
        work_dir = os.path.join(os.getcwd(), "work_dir")
        os.makedirs(work_dir, exist_ok=True)
        return work_dir

    def _pick_device(self):
        """有 CUDA 就用 cuda:0，没有就退回 cpu——不要不管硬件死写 'cuda:0'。"""
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _set_device_label(self, device: str):
        """更新状态栏最下方的 Device 标签，让它反映实际在用的设备。"""
        if hasattr(self, "label_device"):
            self.label_device.setText(f"Device: <b>{device}</b>")

    def choose_v_ckpt_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频预测器(Video) Checkpoint",
            self._default_ckpt_dir(),
            "Checkpoint Files (*.pt *.pth);;All Files (*)",
        )
        if path:
            self.lineEdit_v_ckp_path.setText(path)

    def choose_p_ckpt_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择PNG/图像预测器 Checkpoint",
            self._default_ckpt_dir(),
            "Checkpoint Files (*.pt *.pth);;All Files (*)",
        )
        if path:
            self.lineEdit_p_ckp_path.setText(path)

    def init_video_predictor(self):
        """点击 init_v：用 lineEdit_v_ckp_path 里的路径手动构建视频预测器；不点就一直是 None。"""
        ckpt_path = self.lineEdit_v_ckp_path.text().strip()
        if not ckpt_path or not os.path.isfile(ckpt_path):
            self.speaker(f"[Video] checkpoint 路径无效，未构建: {ckpt_path}")
            return
        device = self._pick_device()
        try:
            self.v_sam2_model = build_sam2_video_predictor(
                config_file="configs/sam2.1/sam2.1_hiera_b+.yaml",
                ckpt_path=ckpt_path,
                device=device,
            )
            self._set_device_label(device)
            self.speaker(f"[Video] 模型加载成功: {ckpt_path} (device={device})")
        except Exception as e:
            self.v_sam2_model = None
            self.speaker(f"[Video] 模型加载失败: {e}")
            print(f"[BrainSAM] init_video_predictor failed: {e}")

    def init_image_predictor(self):
        """点击 init_p：用 lineEdit_p_ckp_path 里的路径手动构建PNG/图像预测器；不点就一直是 None。"""
        ckpt_path = self.lineEdit_p_ckp_path.text().strip()
        if not ckpt_path or not os.path.isfile(ckpt_path):
            self.speaker(f"[PNG] checkpoint 路径无效，未构建: {ckpt_path}")
            return
        device = self._pick_device()
        try:
            self.sam2_model = build_sam2(
                "configs/sam2.1/sam2.1_hiera_b+.yaml",
                ckpt_path,
                device=device,
            )
            self._set_device_label(device)
            self.speaker(f"[PNG] 模型加载成功: {ckpt_path} (device={device})")
        except Exception as e:
            self.sam2_model = None
            self.speaker(f"[PNG] 模型加载失败: {e}")
            print(f"[BrainSAM] init_image_predictor failed: {e}")

    def state_reset(self):
        self.graphicsView.setTransform(QTransform())
        self.mask_preserve = []
        self.mask_temp = []
        self.mak_group = {}
        self.video_segments = {}
        self.nii_img= None
        self.start_page = None
        self.end_page = None
        self.image = None
        self.slice_img = None
        self.nii_size = None
        self.end_point = None
        self.start_pos = (None, None)
        self.end_pos = (None, None)
        self.rect = None
        self.nii_data = None
        self.input_point = None
        self.input_label = None
        self.axis_click = []
        self.axis_click_v = []
        self.type_click = []
        self.ptsList = [[]]
        self.ptsTypeList = []
        self.boxList = []
        self.prompt_Preserve = []
        self.axis = None
        self.image_files = []
        self.current_index = 0
        self.current_slice_idx = 0
        self.folder_path = None
        self.image_length = 0
        self.reverse = False
        self.axis_nii_len = None
        self.re_inintial=True
        self.transpose_orders_index = None
        self.current_flip_idx = 0
        self.re_inintial_embedding = True
        self.inference_state={}
        self.folder_path_folder = None
        self.img_folder_path = None

    def json_load(self):
        json_path,_ = QFileDialog.getOpenFileName(self, "BRAINSAM-chose a config file for batch processing", "","JSON files (*.json)")
        if not json_path or not os.path.isfile(json_path):
            return
        with open(json_path,encoding='utf-8') as jf:
            self.cfg = json.load(jf)
        base_dir = self.cfg.get("file_base_path","")
        if not base_dir or  not os.path.isdir(base_dir):
            return

        self.cfg_Tree.clear()
        root = self.cfg_Tree.invisibleRootItem()
        self.cfg_Tree.setHorizontalHeaderLabels([f"{self.cfg.get('task','Unknow Task')}"])

        file_item = QStandardItem("Files")
        root.appendRow(file_item)
        if os.path.isdir(base_dir):
            for root_dir,_,files in os.walk(base_dir):
                for f in files:
                    if f.lower().endswith(".nii.gz"):
                            file=QStandardItem(os.path.join(root_dir, f))
                            file.setCheckable(True)
                            file_item.appendRow(file)

        #load axis from config
        axis = self.cfg.get("axis", None)
        axis_item = QStandardItem("Axis")
        root.appendRow(axis_item)
        if axis is not None:
            axis_item.appendRow(QStandardItem(axis))
        else:
            txt = "null"
            axis_item.appendRow(QStandardItem(txt))
        #load prompt from config
        self.conf_points = self.cfg.get("point_list", None)
        self.conf_types = self.cfg.get("type_list", None)
        points_item = QStandardItem("Points")
        root.appendRow(points_item)
        if self.conf_points is not None:
            self.cfg_Tree_pt.removeRows(0, self.cfg_Tree_pt.rowCount())
            for obj_idx, (obj_coords, obj_type) in enumerate(zip(self.conf_points, self.conf_types), start=1):
                obj_item = QStandardItem(f"Object-{obj_idx}")
                points_item.appendRow(obj_item)
                for coord, tp in zip(obj_coords, obj_type):
                    label = "Positive" if tp == 1 else "Negative"
                    txt = f"{label} ({coord[0]:.3f},{coord[1]:.3f})"
                    obj_item.appendRow(QStandardItem(txt))
        else:
            txt="null"
            points_item.appendRow(QStandardItem(txt))
        #加载提示框组
        box_item = QStandardItem("Boxes")
        root.appendRow(box_item)
        box = self.cfg.get("box_list", None)
        if box is not None:
            for obj_idx, obj_coords in enumerate(box, start=1):
                obj_item = QStandardItem(f"Object-{obj_idx}")
                box_item.appendRow(obj_item)
                txt = f"({obj_coords[0]:.3f},{obj_coords[1]:.3f},{obj_coords[2]:.3f},{obj_coords[3]:.3f})"
                obj_item.appendRow(QStandardItem(txt))
        else:
            txt = "null"
            box_item.appendRow(QStandardItem(txt))



        # self.load_prompt_from_config()
        self.treeView_file_list.expandAll()
        self.checkBox_Sync_to_Config_File.setEnabled(True)

    def switch_other_obj(self,str):
        self.cfg_Tree_pt.removeRows(0, self.cfg_Tree_pt.rowCount())
        m = re.search(r'\d+$', str)
        obj_num = int(m.group())
        self.obj_num = obj_num - 1
        #保证提示组够长
        if len(self.axis_click) <= obj_num:
            self.axis_click.extend([[] for _ in range(obj_num - len(self.axis_click))])
            self.type_click.extend([[] for _ in range(obj_num - len(self.type_click))])
        if len(self.boxList) <= self.obj_num:
            self.boxList.extend([None for _ in range(self.obj_num - len(self.boxList) + 1)])
        #取得配置文件的专有item组
        root = self.cfg_Tree.invisibleRootItem()
        points_item = None
        box_item = None
        for r in range(root.rowCount()):
            item = root.child(r)
            if item.text() == "Points":
                points_item = item
            if item.text() == "Boxes":
                box_item = item
        if self.conf_points:
            obj_item = None
            obj_item_box = None
            for r in range(points_item.rowCount()):
                item = points_item.child(r)
                if item.text() == f"Object-{obj_num}":
                    obj_item = item
            for r in range(box_item.rowCount()):
                item = box_item.child(r)
                if item.text() == f"Object-{obj_num}":
                    obj_item_box = item
            #寻找坐标并添加到self.cfg_Tree_pt
            pattern = re.compile(r'\((\d+\.?\d*),(\d+\.?\d*)\)')
            if obj_item.rowCount() != 0:
                for r in range(obj_item.rowCount()):
                    child = obj_item.child(r)
                    m = pattern.search(child.text())
                    obj_x, objy = (float(m.group(1)), float(m.group(2)))
                    tp = re.match(r'^(\w+)',child.text()).group(1)
                    if tp =='Positive':
                        tp = 1
                    elif tp =='Negative':
                        tp = 0
                    ax_click = [obj_x, objy ]
                    # 修改提示组列表
                    if ax_click not in self.axis_click[obj_num - 1]:
                        self.axis_click[obj_num - 1].append(ax_click)
                        self.type_click[obj_num - 1].append(tp)
                        print("提示组被修改！", self.axis_click)
                        print(self.type_click)
                    self.add_point(ax_click,tp)

            if obj_item_box.rowCount() != 0:
                for r in range(box_item.rowCount()):
                    item = box_item.child(r)
                    obj_name = box_item.child(r).text()
                    obj_name = tuple(obj_name)
                    coord = None
                    if int(obj_name[-1]) == obj_num:
                        for c in range(item.columnCount()):
                            coord = item.child(c).text()
                            coord =[float(c) for c in coord.strip("()").split(",")]
                            rect = self.scene.addRect(
                                coord[0], coord[1], coord[2] - coord[0], coord[3] - coord[1], pen=QPen(QColor("green"))
                            )
                            for c in list(self.box_set.keys()):
                                if c[0] == self.obj_num:
                                    self.scene.removeItem(self.box_set.pop(c))
                            self.box_set[(self.obj_num, coord[0], coord[1], coord[2], coord[3])] = rect
                            self.listView_pt('box',tuple(coord))

                            if len(self.boxList) <= self.obj_num:
                                self.boxList.extend([None for _ in range(self.obj_num - len(self.boxList) + 1)])
                            self.boxList[self.obj_num] = coord
        #添加图形
        coord_list=[]
        for row in range(self.cfg_Tree_pt.rowCount()):
            coord = self.cfg_Tree_pt.item(row, 2)
            label = self.cfg_Tree_pt.item(row, 1).text().split(" ")[0]
            x, y,*rest = coord.text().strip("()").split(',')
            if not rest:
                x, y = float(x), float(y)
                coord_list.append([x,y])
        if len(self.axis_click)>self.obj_num:
            for i,c in enumerate(self.axis_click[self.obj_num],start=0):
                if c not in coord_list:
                    tp=self.type_click[self.obj_num][i]
                    self.add_point(c, tp)
        if len(self.boxList)>self.obj_num:
            if self.boxList[self.obj_num] not in coord_list and self.boxList[self.obj_num] is not None:
                self.listView_pt('box',self.boxList[self.obj_num])

    @pyqtSlot(QModelIndex)
    def tree_doubleClicked(self,idx):
        item = self.cfg_Tree.itemFromIndex(idx)
        text = item.text()
        self.load_source_file(file_path=text)
    def collect_points_box(self):
        root = self.cfg_Tree.invisibleRootItem()
        points_node = None
        boxex_node = None
        for r in range(root.rowCount()):
            item = root.child(r)
            if item.text() == "Points":
                points_node = item
            if item.text() == "Box":
                boxex_node = item

        point_out = []
        type_list_out =[]
        if points_node :
            for obj_row in range(points_node.rowCount()):
                obj_item = points_node.child(obj_row)
                coords = []
                types = []
                for pt_row in range(obj_item.rowCount()):
                    pt_item = obj_item.child(pt_row)
                    txt = pt_item.text()
                    clean = txt.split('(')[-1].split(')')[0]
                    x_str,y_str = clean.split(',')
                    coords.append([float(x_str),float(y_str)])
                    tp = 1 if txt.startswith("Positive") else 0
                    types.append(tp)
                point_out.append(coords)
                type_list_out.append(types)

        box_out = []
        if boxex_node :
            for obj_row in range(boxex_node.rowCount()):
                obj_item = boxex_node.child(obj_row)
                boxes = []
                for box_row in range(obj_item.rowCount()):
                    box_item = obj_item.child(box_row)
                    coords = box_item.text().strip("()").split(',')
                    boxes.append([float(c) for c in coords])
                box_out.append(boxes)
        for sublist in point_out:
            if len(sublist)==0:
                point_out.remove(sublist)
        for sublist in type_list_out:
            if len(sublist)==0:
                type_list_out.remove(sublist)
        for sublist in box_out:
            if len(sublist)==0:
                box_out.remove(sublist)
        if len(box_out)==0:box_out=None
        if len(point_out)==0:
            point_out=None
            type_list_out=None
        print(point_out)
        print(type_list_out)

        return point_out,type_list_out,box_out


    def tree_start(self):
        checked_item = []
        file_item = self.cfg_Tree.item(0)
        for row in range(file_item.rowCount()):
            item = file_item.child(row)
            if item.checkState() == Qt.Checked:
                checked_item.append(item.text())
        point_out, type_list_out, box_out =self.collect_points_box()

        root = self.cfg_Tree.invisibleRootItem()
        axis_node = None
        for r in range(root.rowCount()):
            item = root.child(r)
            if item.text() == "Axis":
                axis_node = item.child(0)
                print("find axis")
        self.create_temp_text_window_in_scene()
        self.append_text_in_temp_window("config file:")
        self.append_text_in_temp_window(f"[Find {len(checked_item)} checked NIFTI files]: ")
        for i in range(len(checked_item)):
            self.append_text_in_temp_window(f"  {checked_item[i]}")
        #id: 0: coronal, 1: axial, 2: sagittal
        if axis_node.text() == "cor":
            self.axis = 0
            self.append_text_in_temp_window(f"[Orientation]: coronal.")
        elif axis_node.text() == "axi":
            self.axis = 1
            self.append_text_in_temp_window(f"[Orientation]: transverse.")
        elif axis_node.text() == "sag":
            self.axis = 2
            self.append_text_in_temp_window(f"[Orientation]: sagittal.")

        self.append_text_in_temp_window(f"[Prompt]")
        self.append_text_in_temp_window(f"  point: {point_out} .")
        self.append_text_in_temp_window(f"  type: {type_list_out} .")
        self.append_text_in_temp_window(f"  box: {box_out} .")

        self.multi_mask_gen = Multi_gen_mask(self.v_sam2_model,checked_item,self.axis,pt=point_out,
                                             ptTp=type_list_out,box=box_out)
        self.multi_mask_gen.signals.seg_dict.connect(self.thread_seg_reslut)
        self.multi_mask_gen.signals.progress.connect(lambda res: self.receive_and_append_in_temp_window(res,checked_item,len(checked_item)))
        self.multi_mask_gen.start()

    def thread_seg_reslut(self,video_dir,res):
        print("video_dir:",video_dir)
        self.nii_img = nii_reader(video_dir)
        self.video_segments = res
        filename = os.path.split(video_dir)[1].split('.')[0] + "_mask" + '.nii.gz'
        file_path = os.path.join(os.path.splitext(video_dir)[0],filename)
        print("file_path:",file_path)
        self.save_nifti(file_path)

    def listView_pt(self,type,coord):
        row = self.cfg_Tree_pt.rowCount()
        if type=="box":
            for r in range(self.cfg_Tree_pt.rowCount()-1):
                label = self.cfg_Tree_pt.item(r, 1).text()
                if label == "box":
                    self.cfg_Tree_pt.removeRow(r)
                    break
        items = [
            QStandardItem(f"{row+1}"),
             QStandardItem(f"{type}"),
             QStandardItem(f"{coord}")
        ]
        for item in items:
            item.setEditable(False)
        self.cfg_Tree_pt.appendRow(items)

    def select_point_use_circle(self,selected,deselected):
        coords = set()
        for idx in self.treeView_pt.selectionModel().selectedIndexes():
            if idx.column() == 2: #取坐标列
                coord_str =idx.data(Qt.DisplayRole)
                coords.add(parse_coords(coord_str))
        for coord in list(self._last_circle.keys()):
            if coord not in coords:
                self.scene.removeItem(self._last_circle.pop(coord))
        for c in coords:
            if c not in self._last_circle:
                self.add_circle(c)

    def sync_to_config_file(self,str,root):
        """
        sync the current point box to config file without save to the jason file.
        :param str:
        :param root:
        :return:
        """
        m =re.search(r'\d+$',str)
        obj_num = int(m.group())
        obj_item = None
        obj_item_box = None
        self.obj_num = obj_num-1
        for i in range(root.rowCount()):
            point_item = root.child(i)
            if point_item.text() == "Points":
                for j in range(point_item.rowCount()):
                    obj_item1 = point_item.child(j)
                    if obj_item1.text() == f"Object-{obj_num}":
                        obj_item = obj_item1
                        break
                continue
            if point_item.text() == "Boxes":
                for j in range(point_item.rowCount()):
                    obj_item1 = point_item.child(j)
                    if obj_item1.text() == f"Object-{obj_num}":
                        obj_item_box = obj_item1
                        break
                break
        if obj_item is None:
            self.speaker(f"obj{m} is not exist,add this object in the config file first.The file tag provide a convenient way to add.")
            return
        temp_treeA_existing = set()

        pattern = re.compile(r'\((\d+\.?\d*),(\d+\.?\d*)\)')
        for r in  range(obj_item.rowCount()):
            child = obj_item.child(r)
            m = pattern.search(child.text())
            if m:
                temp_treeA_existing.add((float(m.group(1)),float(m.group(2))))
        for r in range(obj_item_box.rowCount()):
            child = obj_item_box.child(r)
            li = [float(x) for x in child.text().strip("()").split(',')]
            temp_treeA_existing.add((li[0],li[1],li[2],li[3]))

        for row in range(self.cfg_Tree_pt.rowCount()):
            coord = self.cfg_Tree_pt.item(row,2)
            label = self.cfg_Tree_pt.item(row,1).text().split(" ")[0]
            if label == "box":
                x1, y1 ,x2,y2 = coord.text().strip("()").strip("[]").split(',')
                x1,y1,x2,y2 = map(float,(x1,y1,x2,y2))
                if (x1,y1,x2,y2) not in temp_treeA_existing:
                    text = f"({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f})"
                    if obj_item_box.rowCount()==0:
                        obj_item_box.appendRow(QStandardItem(text))
                    else:
                        obj_item_box.child(0).setText(text)
            if not coord:
                continue
            try:
                print("ccoo:",coord.text())
                x,y = coord.text().strip("()").split(',')
                x,y = float(x),float(y)
            except ValueError:
                print(coord)
                continue
            if (x,y) not in temp_treeA_existing:
                text = f"{label}({x:.3f},{y:.3f})"
                obj_item.appendRow(QStandardItem(text))

    def sync_to_config_file_del(self,str,coord,root):
        m =re.search(r'\d+$',str)
        obj_num = int(m.group())
        obj_item = None
        obj_item_box = None
        for i in range(root.rowCount()):
            point_item = root.child(i)
            if point_item.text() == "Points":
                for j in range(point_item.rowCount()):
                    obj_item1 = point_item.child(j)
                    if obj_item1.text() == f"Object-{obj_num}":
                        obj_item = obj_item1
                        break
            if point_item.text() == "Boxes":
                for j in range(point_item.rowCount()):
                    obj_item1 = point_item.child(j)
                    if obj_item1.text() == f"Object-{obj_num}":
                        obj_item_box = obj_item1
                        break

                break
        if obj_item is None:
            self.speaker(f"obj{m} is not exist,add this object in the config file first.The file tag provide a convenient way to add.")
            return
        x,y,*rest =coord
        x,y = float(x),float(y)

        #remove matching point prompt
        pattern = re.compile(r'\((\d+\.?\d*),(\d+\.?\d*)\)')
        for r in range(obj_item.rowCount()):
            child = obj_item.child(r)
            m = pattern.search(child.text())
            obj_x,objy = (float(m.group(1)), float(m.group(2)))
            if obj_x ==x and objy ==y:
                obj_item.removeRow(r)
                break
        #remove matching box prompt
        try:
            li=obj_item_box.child(0).text().strip("()").split(",")
            if float(li[0])==x and float(li[1])==y:
                obj_item_box.removeRow(0)
        except:
            print("something wrong in sync box delete")
            pass
    def upadate_radio_lock_by_filepath(self):
        if self.source_path is None:
            self.radioButton_None.setChecked(True)
            self.radioButton_Pts.setEnabled(False)
            self.radioButton_Box.setEnabled(False)
            if self.Button_MG_prompt.isChecked():
                self.Button_MG_prompt.setEnabled(False)
            self.Button_MG_prompt.setEnabled(False)
        else:
            self.radioButton_None.setChecked(True)
            self.radioButton_Pts.setEnabled(True)
            self.radioButton_Box.setEnabled(True)
            self.Button_MG_prompt.setEnabled(True)

    def add_point(self, ax_click, type_click):
        x, y = ax_click
        point_size = 4
        point = QGraphicsEllipseItem(x - point_size, y - point_size, point_size * 2, point_size * 2)
        point.setZValue(1000)
        if type_click == 1:  # 绿点
            point.setPen(QPen(QColor("green")))
            point.setBrush(QBrush(QColor("green")))
            self.listView_pt(f"Positive point", f"{x, y}")
        else:  # 红点
            point.setPen(QPen(QColor("red")))
            point.setBrush(QBrush(QColor("red")))
            self.listView_pt(f"Negtive point", f"{x, y}")
        if (x, y) not in self.point_set:
            self.point_set[(x, y)] = point
            self.scene.addItem(point)
            self.scene.update()
            print("add point")

        if self.checkBox_Sync_to_Config_File.isChecked():
            self.sync_to_config_file(self.comboBox.currentText(), self.cfg_Tree.invisibleRootItem())

    def add_circle(self, ax_click):
        """
        add a circle to the scene for point selection
        """
        center_x, center_y,*rest= ax_click
        outer = None
        color = QColor(255,128,0)
        if rest:
            x2,y2 = rest
            outer = self.scene.addRect(
                center_x, center_y, x2 - center_x, y2 - center_y, pen=QPen(color)
            )
        else:
            outer_r =5
            line_width = 4
            outer  = QGraphicsEllipseItem(
                QRectF(center_x-outer_r,
                       center_y-outer_r,
                       outer_r*2,
                       outer_r*2,
                )
            )
            outer.setPen(QPen(QColor(color), line_width))
            outer.setBrush(QBrush(Qt.NoBrush))
        self._last_circle[ax_click] = outer
        self.scene.addItem(outer)
        self.scene.update()

    def del_item_and_coord(self,):
        print(f"ax before",self.axis_click)
        print(f"tp before",self.type_click)
        #find current obj num
        m = re.search(r'\d+$', self.comboBox.currentText())
        m = int(m.group())
        coords = set()
        # a dict to store the type of each coord
        type_dict = {}
        #the rows that have been selected
        rows = sorted({idx.row() for idx in self.treeView_pt.selectionModel().selectedIndexes()},reverse=True)
        for idx in self.treeView_pt.selectionModel().selectedIndexes():
            # print("当前选中行：",idx.row)
            if idx.column() == 2:  # 取坐标列
                type_clumn_idx =idx.sibling(idx.row(),1)
                type_clumn_data = type_clumn_idx.data()
                if type_clumn_data == "Positive point":
                    type_clumn_data = 1
                else:
                    type_clumn_data = 0
                coord_str = idx.data(Qt.DisplayRole)
                c= parse_coords(coord_str)
                coords.add(c)
                #store every coord's type
                type_dict[c] = type_clumn_data
        for c in coords:
            if len(c)==2: #处理提示组-点
                #remove point from scene
                self.scene.removeItem(self.point_set.pop(c))
                x,y =c
                #remove coord from the list for sam-prompt
                self.axis_click[m-1].remove([int(x/self.magnify),int(y/self.magnify)])
                if not self.axis_click[m-1] and len(self.axis_click)>m :
                    if not self.axis_click[m]:
                        del self.axis_click[m-1]
                elif not self.axis_click[m-1] and len(self.axis_click)==m:
                    del self.axis_click[m-1]
                self.type_click[m-1].remove(type_dict[c])
                if not self.type_click[m-1] and len(self.axis_click)>m :
                    if not self.type_click[m]:
                        del self.type_click[m-1]
                elif not self.type_click[m-1] and len(self.type_click)==m:
                    del self.type_click[m-1]
                if self.checkBox_Sync_to_Config_File.isChecked():
                    self.sync_to_config_file_del(self.comboBox.currentText(),c, self.cfg_Tree.invisibleRootItem())
            else:#处理提示组-box
                x1, y1, x2, y2 = c
                c=(self.obj_num,x1, y1, x2, y2 )
                try:
                    if c[0] == self.obj_num:
                        self.scene.removeItem(self.box_set.pop(c))
                    self.boxList.remove([float(x1/self.magnify), float(y1/self.magnify) ,float(x2/self.magnify), float(y2/self.magnify)])
                    if self.checkBox_Sync_to_Config_File.isChecked():
                        self.sync_to_config_file_del(self.comboBox.currentText(), (x1, y1, x2, y2), self.cfg_Tree.invisibleRootItem())
                except KeyError:
                    pass
        for r in rows:
            self.cfg_Tree_pt.removeRow(r)

    def clear_treeView_pt(self):
        self.cfg_Tree_pt.removeRows(0, self.cfg_Tree_pt.rowCount())
        self.boxList = []
        self.type_click = []
        self.box_set = {}
        self._last_circle = {}
        self.axis_click = []
        self.mask_group = {}

    def receive_and_append_in_temp_window(self, num, checked_items, total=0, ):
        last_digit = num % 10
        rest = num // 10
        if last_digit == 0:
            self.append_text_in_temp_window(f"\nTotal {total}, now processing {rest + 1}:{checked_items[rest]} ")
            self.append_text_in_temp_window(f"calculating embedding")
        elif last_digit == 2:
            self.append_text_in_temp_window(f"propogate ...")
        elif last_digit == 3:
            self.append_text_in_temp_window(f"Successfully saving to ...")
        elif last_digit == 4:
            self.append_text_in_temp_window(f"\nComplated.")

    def create_temp_text_window_in_scene(self):
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("background-color: black; color: white;")
        proxy = self.scene.addWidget(self.text_edit)
        proxy.setPos(0, 0)
        scene_rect = self.graphicsView.viewport().rect()
        proxy.resize(scene_rect.width(), scene_rect.height())
        self.scene.setSceneRect(0, 0, self.scene.width(), self.scene.height())
        self.graphicsView.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def remove_temp_text_window_in_scene(self):
        if hasattr(self, 'text_edit'):
            self.scene.removeItem(self.text_edit)
            del self.text_edit
        self.graphicsView.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

    def append_text_in_temp_window(self, text):
        self.text_edit.append(text)

    def NIFTIAxis(self, id):
        """
        id: 0: coronal, 1: axial, 2: sagittal
        """
        # self.state_reset()
        if self.nii_img is None:
            return
        self.axis = id
        self.axis_nii_len = self.nii_img.get_slices_num(id)
        self.image_length = self.axis_nii_len
        temp_idx = self.current_slice_idx
        if self.current_slice_idx > self.axis_nii_len - 1:
            temp_idx = self.axis_nii_len - 1
        self.Slider_SelectFiles.setRange(0, self.axis_nii_len - 1)
        self.Slider_SelectFiles.setValue(temp_idx)
        if self.checkbox_auto_Embedding.isChecked():
            self.thread_cal_embedding = calcute_embedding_thread(self.v_sam2_model, self.folder_path,
                                                                 axis=self.axis)
            self.thread_cal_embedding.prog_signal.connect(self.bar_embedding_calculate.setValue)
            self.thread_cal_embedding.prog_show_and_hide.connect(self.progress_bar_cal)
            self.thread_cal_embedding.finished.connect(self.recieve_infrence_state)
            self.thread_cal_embedding.start()

        self.show_slice(temp_idx)  # Show the first slice for the selected axis

    def NIFTIMagnify(self, id):
        """
        Magnify the nifti image.
        id: 1:1x, 2: 2x, 4: 4x, 0.5: 0.5x
        """
        # if self.nii_img is None:
        #     return
        self.magnify = id if id != 5 else 0.5
        self.show_slice(self.current_slice_idx)
        self.show_image(self.current_index)

    def change_transpose_order(self):
        if self.transpose_orders_index is None:
            self.transpose_orders_index = 0
        self.transpose_orders_index = (self.transpose_orders_index + 1) % len(self.transpose_orders)
        current_order = self.transpose_orders[self.transpose_orders_index]

        self.speaker(f"change Transpose Order to: {current_order}")

    def get_transpose_order(self):
        current_order = self.transpose_orders[self.transpose_orders_index]
        if current_order == None:
            current_order = (0, 1, 2)
        return current_order

    def change_flip_state(self):
        self.current_flip_idx = (self.current_flip_idx + 1) % len(self.flip_states)
        current_flip_state = self.flip_states[self.current_flip_idx]

        if current_flip_state is None:
            self.speaker("No Flip")
        else:
            self.speaker("Flip State changed to: {self.current_flip_idx}")

    def get_flip_state(self):
        # 返回当前的翻转状态
        return self.flip_states[self.current_flip_idx]

    def update_view(self, pixmap_item=None):
        """更新视图，根据图片大小决定是否显示滚动条，并居中显示图片"""
        if pixmap_item:
            # 获取图片的大小
            pixmap_rect = pixmap_item.boundingRect()
            pixmap_width = pixmap_rect.width()
            pixmap_height = pixmap_rect.height()

            # 获取视图的大小
            view_rect = self.graphicsView.viewport().rect()
            view_width = view_rect.width()
            view_height = view_rect.height()

            # 根据图片和视图的大小设置滚动条策略
            if pixmap_width <= view_width:
                self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            else:
                self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

            if pixmap_height <= view_height:
                self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            else:
                self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

            # 将图片居中显示
            self.graphicsView.centerOn(pixmap_item.boundingRect().center())
            self.scene.setSceneRect(pixmap_item.boundingRect())

    def load_source_file(self, file_path=None):
        if file_path is None:
            file_path = QFileDialog.getOpenFileName(self, "Select File",
                                                    self.folder_path_folder if self.folder_path_folder is not None else "",
                                                    "NIfTI Files (*.nii *.nii.gz);;Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                    # options=QFileDialog.DontUseNativeDialog
                                                    )[0]
        self.source_path = file_path
        self.source_path_folder, _ = os.path.split(file_path)
        self.lineEdit_SourceFiles.setText(file_path)
        if file_path:
            # 如果是个NIfTI，加载它
            self.state_reset()
            # 重新设置原点为左下脚
            # transform = QTransform()
            # transform.scale(1, -1)
            # self.graphicsView.setTransform(transform)
            if file_path.lower().endswith(('.nii', '.nii.gz')):
                self.load_nifti_file(file_path)
                self.file_type = 1
                self.nii_path = file_path
                folder, file_name = os.path.split(self.nii_path)
                self.lineEdit_TargetFiles.setText(folder)
                file_name = file_name.split(".")[0]
                self.nii_name = file_name
                to_save_file_name = file_name + "_mask" + ".nii.gz"
                self.Rename.setText(to_save_file_name)
            else:
                # 如果是个图片，展示它
                self.state_reset()
                self.load_image(file_path)
                self.file_type = 3
                self.img_path = file_path
                folder, file_name = os.path.split(self.img_path)
                self.lineEdit_TargetFiles.setText(folder)
                file_name = file_name.split(".")[0]
                to_save_file_name = file_name + "_mask" + ".png"
                self.Rename.setText(to_save_file_name)

    def on_source_lineedit_enter(self):
        text = self.lineEdit_SourceFiles.text().strip()
        if text:  # 当文本非空时
            self.load_source_file(text)

    def load_images_from_folder(self):

        try:
            folder_path = QFileDialog.getExistingDirectory(self, "Select File")
            if folder_path:  # 确保选择了文件夹
                self.state_reset()
                self.file_type = 2
                self.lineEdit_TargetFiles.setText(folder_path)
                self.lineEdit_SourceFiles.setText(folder_path)
                self.image_files = [f for f in os.listdir(folder_path) if
                                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                self.image_files.sort(key=lambda p: int(os.path.splitext(p)[0]))
                self.current_index = 0
                self.Slider_SelectFiles.setRange(0, len(self.image_files) - 1)
                self.Slider_SelectFiles.setValue(self.current_index)
            self.img_folder_path = self.lineEdit_SourceFiles.text()
            self.img_folder_path_name = os.path.split(self.img_folder_path)[-1]
            self.image_length = len(os.listdir(folder_path))
            self.Label_CurentInfo.setText(
                f"Total-->{self.image_length},Current Image Number: " + str(self.current_index))
            if not folder_path or not os.path.isdir(folder_path):
                return

            self.show_image(self.current_index)
            self.upadate_radio_lock_by_filepath()
        except Exception as e:
            print(e)

    def load_mask_file(self, file_path, orient):
        self.video_segments = {}
        try:
            file_name = os.path.split(file_path)[-1]
            mask_name = file_name.split('.')[0] + "_mask" + '.nii.gz'
            masked_file = os.path.join(os.path.split(file_path)[0], mask_name)
            print(f"masked_file:{masked_file}")
            print(f"orient:{orient}")
            self.nii_mask = nii_reader(masked_file)
            n_slices = self.nii_mask.get_slices_num(orient)
            vol = self.nii_mask.data.astype(np.float32)
            labels = np.unique(vol)
            labels = labels[labels != 0]  # 排除背景标签0
            id_map = {lbl: int(lbl) for lbl in labels}
            for idx in range(n_slices):
                mask, _ = self.nii_mask.get_slice_array(orient, idx)
                mask_dict = {}
                for lbl in labels:
                    mask_dict[id_map[lbl]] = (mask == lbl).astype(np.uint8)
                self.video_segments[idx] = mask_dict
                print(type(self.video_segments), len(self.video_segments))
        except Exception as e:
            print(f"Error loading Mask file: {e}")
            self.Label_CurentInfo.setText(f"Failed to load Mask file:{e}")

    def load_nifti_file(self, file_path):
        try:
            # self.upadate_radio_lock_by_filepath()
            self.nii_img = nii_reader(file_path)
            self.nii_img_for_preserve_nifti = sitk.ReadImage(file_path)
            if self.axis is None:
                self.axis = 1  #
            initial_slice_num = self.nii_img.get_slices_num(self.axis)
            self.axis_nii_len = self.nii_img.get_slices_num(self.axis)
            self.image_length = initial_slice_num
            self.Slider_SelectFiles.setRange(1, initial_slice_num)
            self.Slider_SelectFiles.setValue(self.current_slice_idx)

            self.show_slice(self.current_slice_idx)
            file_name = os.path.split(file_path)[-1]
            mask_name = file_name.split('.')[0] + "_mask" + '.nii.gz'
            self.Rename.setText(f"{mask_name}")
            masked_file = os.path.join(os.path.split(file_path)[0], mask_name)
            if os.path.exists(masked_file):
                self.Button_loadMask.setEnabled(True)
            else:
                self.Button_loadMask.setEnabled(False)
            self.Label_CurentInfo.setText(f"Loaded NIfTI file: {file_path}")
            self.textBrowser.append(f"Loaded NIfTI file: {file_path}")
            # self.statusBar.clearMessage()
            self.statusBar.showMessage(f"Loaded NIfTI file: {file_path}")
            self.label_embedding.setText(
                'Status: <span style="color:#C78E00; font-weight:700;">Embedding empty</span>')
            self.label_embedding.setTextFormat(Qt.RichText)
            if self.checkbox_auto_Embedding.isChecked():
                self.bar_embedding_calculate
                self.thread_cal_embedding = calcute_embedding_thread(self.v_sam2_model, file_path, axis=1)
                self.thread_cal_embedding.prog_signal.connect(self.bar_embedding_calculate.setValue)
                self.thread_cal_embedding.prog_show_and_hide.connect(self.progress_bar_cal)
                self.thread_cal_embedding.finished.connect(self.recieve_infrence_state)
                self.thread_cal_embedding.start()
        except Exception as e:
            print(f"Error loading NIfTI file: {e}")
            self.Label_CurentInfo.setText("Failed to load NIfTI file.")

    def load_image(self, file_path):
        self.image = file_path
        try:
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                print(f"Failed to load image: {file_path}")
                return
            self.scene.clear()
            self.scene.addItem(QGraphicsPixmapItem(pixmap))
            self.Label_CurentInfo.setText(f"Loaded image: {file_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            self.Label_CurentInfo.setText("Failed to load image.")

    def show_image(self, index):
        """
        a function to show image from folder.
        """
        if not self.image_files:
            return  # No images to display
        image_file = self.image_files[index]
        image_path = os.path.join(self.img_folder_path, image_file)
        bgr=cv2.imread(image_path)
        if bgr is None:
            print(f"Failed to load image: {image_path}")
            return
        interp = cv2.INTER_AREA if self.magnify <= 1 else cv2.INTER_CUBIC
        resized =cv2.resize(bgr, (0,0), fx=self.magnify, fy=self.magnify,interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        qimg= QImage(rgb.data, width, height, width * channel, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if pixmap.isNull():
            print(f"Failed to load image: {image_path}")
            return
        self.scene.clear()
        item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.update_view(item)
        if not self.continue_inference and self.video_segments:
            for out_obj_id, out_mask in self.video_segments[self.current_index].items():
                self.show_mask(out_mask)
                if self.checkBox_save_segment.isChecked():
                    folder_path = self.lineEdit_TargetFiles.text()
                    image_path = os.path.join(folder_path, self.image_files[self.current_index])
                    self.save_segmentation(out_mask, image_path)

        self.Label_CurentInfo.setText(f'Image: {index + 1} / {len(self.image_files)}')

    def show_mask(self, mask, idx=0):
        random_color = False
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = idx
            color = np.array([*cmap(cmap_idx)[:3], 1])
        # color = np.array(mask_colors[idx])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = np.clip(mask_image * 255, 0, 255).astype(np.uint8)
        if self.magnify >= 1:
            mask_image = mask_image.repeat(self.magnify, axis=0).repeat(self.magnify, axis=1)
        else:
            mask_image = cv2.resize(mask_image, (0,0),fx=self.magnify,fy=self.magnify,interpolation=cv2.INTER_AREA)
        if self.magnify != 1:
            self.slice_img = self.slice_img.repeat(self.magnify, axis=0).repeat(self.magnify, axis=1)
        else:
            self.slice_img = self.slice_img_p
        if self.slice_img is not None:
            # Scale the pixmap to fit the QGraphicsView's size# Get the current size of the graphics view
            mask_image = np2pixmap(mask_image)
            # mask_image = mask_image.scaled(view_width, view_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            mask_image = np2pixmap(mask_image)
        try:
            if idx in list(self.mask_group.keys()):
                self.scene.removeItem(self.mask_group.pop(idx))
        except:
            pass
        mask_item = QGraphicsPixmapItem(mask_image)
        mask_item.setOpacity(self.maskOpacity)
        self.mask_group[idx] = mask_item
        self.scene.addItem(mask_item)

    def show_slice(self, slice_index):
        if self.axis == None:
            self.axis = 1
        if self.nii_img is None:
            return
        slice_img, self.nii_slice_transpose = self.nii_img.get_slice_array(self.axis, slice_index)
        # Normalize the slice data to [0, 255] range for displaying
        slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img)) * 255
        self.slice_img = slice_img.astype(np.uint8)
        self.slice_img_p = self.slice_img
        if self.magnify != 1 and self.magnify !=5:
            self.slice_img = self.slice_img.repeat(self.magnify, axis=0).repeat(self.magnify, axis=1)
        elif self.magnify == 0.5:

            self.slice_img = cv2.resize(self.slice_img, (0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
        else:
            self.slice_img = self.slice_img_p
            # step = 2
            # self.slice_img = self.slice_img[::step, ::step]
        # Convert the NumPy array (slice_img) to QPixmap
        height, width = self.slice_img.shape
        # print(f"slice shape:{height,width}")
        bytes_per_line = width  # Each line of the image corresponds to the width in bytes
        qimage = QImage(self.slice_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Convert the QImage to QPixmap
        pixmap = QPixmap.fromImage(qimage)

        # Clear the scene before adding the new image
        self.scene.clear()
        item = QGraphicsPixmapItem(pixmap)

        # item.setPos(0, -pixmap.height())
        self.scene.addItem(item)
        self.update_view(item)
        if self.current_slice_idx in list(self.video_segments.keys()):
            masks = self.video_segments[self.current_slice_idx]
            for i, mask in enumerate(masks.values()):
                self.show_mask(mask, i)

        self.Label_CurentInfo.setText(
            f" {slice_index + 1} of {self.axis_nii_len}, HxW:{self.nii_img.get_slices_shape(self.axis)}"
        )

    def go_to_start_page(self):
        if self.start_page is not None:
            self.Slider_SelectFiles.setValue(self.start_page)
        else:
            pass

    def go_to_end_page(self):
        if self.end_page is not None:
            self.Slider_SelectFiles.setValue(self.end_page)
        else:
            pass

    def go_to_any_page(self):
        # 使用 1-based 页码显示，符合用户直觉
        idx = self.Slider_SelectFiles.value()
        last_idx = self.axis_nii_len - 1
        mid_idx = last_idx // 2

        # 转换 0-based 索引为 1-based 页码用于显示
        page_1 = 1
        page_mid = mid_idx + 1
        page_last = self.axis_nii_len

        if idx == 0:
            # 0 → last
            target = last_idx
            btn_text = f"to Page {page_mid}"  # 下次去中间

        elif idx == last_idx:
            # last → mid
            target = mid_idx
            btn_text = f"to Page {page_1}"  # 下次去开头

        elif idx == mid_idx:
            # mid → 0（专门处理中间页，保持循环）
            target = 0
            btn_text = f"to Page {page_last}"  # 下次去结尾

        else:
            # 其他任意页 → 先回 0（或你想直接去 mid？）
            target = 0
            btn_text = f"to Page {page_last}"  # 统一到循环中

        # 强制更新，防止信号丢失
        self.Slider_SelectFiles.blockSignals(True)
        self.Slider_SelectFiles.setValue(target)
        self.Slider_SelectFiles.blockSignals(False)

        # 手动触发更新
        self.CLB_skip_page.setText(btn_text)

    def on_button_propagate_clicked(self):
        self.continue_inference = True

    def mouse_press(self, ev):
        # if self.nii_img is None:
        #     inv_transform, _ = self.graphicsView.transform().inverted()
        #     visual_pos = inv_transform.map(ev.scenePos())
        #     x, y = visual_pos.x(), visual_pos.y()
        # else:
        x, y = ev.scenePos().x(), ev.scenePos().y()
        if self.modelSelect == 1:  # boundding Box
            self.is_mouse_down = True
            self.start_pos = x, y
            print(self.start_pos)
        elif self.modelSelect == 2:
            if ev.button() == Qt.LeftButton:#green dot
                ax_click = [x, y]
                if len(self.axis_click) <= self.obj_num:
                    self.axis_click.extend([[] for _ in range(self.obj_num - len(self.axis_click) + 1)])
                    self.type_click.extend([[] for _ in range(self.obj_num - len(self.type_click) + 1)])
                self.add_point(ax_click, 1)
                ax_click = [int(x / self.magnify), int(y / self.magnify)]
                self.axis_click_v.append(ax_click)
                self.ptsList.append(self.axis_click)
                self.axis_click[self.obj_num].append(ax_click)
                self.type_click[self.obj_num].append(1)
                self.prompt_Preserve.append(self.current_index)
                self.speaker(f"point：{self.axis_click_v},label:{self.type_click}")
            else:#red dot
                ax_click = [x, y]
                if len(self.axis_click) <= self.obj_num:
                    self.axis_click.extend([[] for _ in range(self.obj_num - len(self.axis_click) + 1)])
                    self.type_click.extend([[] for _ in range(self.obj_num - len(self.type_click) + 1)])
                # self.axis_click[self.obj_num].append(ax_click)
                self.add_point(ax_click, 2)
                ax_click = [int(x / self.magnify), int(y / self.magnify)]
                self.axis_click_v.append(ax_click)
                self.axis_click[self.obj_num].append(ax_click)
                self.type_click[self.obj_num].append(0)
                self.ptsTypeList.append(self.type_click)
                self.prompt_Preserve.append(self.current_index)
                # 在窗口中用红色的点标识出来
                # print(f"point：{self.axis_click_v}")
                # print(f"label:{self.type_click}")
                # print(f"Done.Add background point ：{ax_click}")
                self.speaker(f"point：{self.axis_click_v},label:{self.type_click}")

            try:
                if self.file_type  == 2 :
                    image = os.path.join(self.img_folder_path, self.image_files[self.current_index])
                elif self.file_type  == 1:
                    image = self.slice_img_p
                elif self.file_type == 3:
                    image = self.image
                masks, _ = PNG_inference.PNG_inference(
                    self.sam2_model,
                    image,
                    self.axis_click,
                    self.type_click,
                    self.boxList,
                )
                self.mask = masks.squeeze(0).astype(np.uint8) if masks.shape[0] != 1 else masks.astype(
                    np.uint8)
                if self.slice_img is not None:
                    self.speaker("single mask was written in a file,which is prepared to save as a nifti.")
                    for i, mask in enumerate(masks):
                        mask_to_nii = mask.squeeze(0).astype(np.uint8) if masks.shape[0] != 1 else mask.astype(
                            np.uint8)
                        self.video_segments[self.current_slice_idx] = {i: mask_to_nii}
                if masks.shape[0] != 1:
                    for i, mask in enumerate(masks, start=0):
                        self.show_mask(mask.squeeze(0), i)

                else:
                    self.show_mask(masks)

                # if self.CheckBox_SavePNG.isChecked():
                #     self.save_png(mask, image)
            except Exception as e:
                print(f"error in mouse press: {e}")
        else:
            return

    def mouse_move(self, ev):
        try:
            if self.modelSelect == 1 and self.is_mouse_down:
                # x, y = int(ev.scenePos().x()/self.magnify),int( ev.scenePos().y()/self.magnify)
                x, y = ev.scenePos().x(), ev.scenePos().y()
                if self.end_point is not None:
                    self.scene.removeItem(self.end_point)
                self.end_point = self.scene.addEllipse(
                    x - self.half_point_size,
                    y - self.half_point_size,
                    self.point_size,
                    self.point_size,
                    pen=QPen(QColor("red")),
                    brush=QBrush(QColor("red")),
                )

                if self.rect_temp is not None:
                    self.scene.removeItem(self.rect_temp)
                sx, sy = self.start_pos
                xmin = min(x, sx)
                xmax = max(x, sx)
                ymin = min(y, sy)
                ymax = max(y, sy)
                self.rect_temp = self.scene.addRect(
                    xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"))
                )
            if self.is_erasing:
                if self.is_mouse_down:  # 只有在擦除模式下才进行擦除
                    x, y = ev.scenePos().x(), ev.scenePos().y()
                    print(f"eraser!{x},{y}")
                    self.erase_area(x, y)
                    print("ok")
            else:
                return
        except Exception as e:
            print(e)

    def mouse_release(self, ev):
        if self.nii_img is None:
            inv_transform, _ = self.graphicsView.transform().inverted()
            visual_pos = inv_transform.map(ev.scenePos())
            x, y = visual_pos.x(), visual_pos.y()
        else:
            x, y = ev.scenePos().x(), ev.scenePos().y()
        if self.modelSelect == 1 and self.is_erasing == False:
            self.end_pos = float(x / self.magnify), float(y / self.magnify)
            sx_real, sy_real = self.start_pos
            sx, sy = float(sx_real / self.magnify), float(sy_real / self.magnify)
            xmin = min(x, sx_real)
            xmax = max(x, sx_real)
            ymin = min(y, sy_real)
            ymax = max(y, sy_real)
            if self.rect_temp is not None:
                self.scene.removeItem(self.rect_temp)
                self.scene.removeItem(self.end_point)

            rect = self.scene.addRect(
                xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("green"))
            )
            for c in list(self.box_set.keys()):
                if c[0] == self.obj_num:
                    self.scene.removeItem(self.box_set.pop(c))
            self.box_set[(self.obj_num, sx_real, sy_real, x, y)] = rect
            self.is_mouse_down = False
            if len(self.boxList) <= self.obj_num:
                self.boxList.extend([None for _ in range(self.obj_num - len(self.boxList) + 1)])
            self.boxList[self.obj_num] = list((sx, sy) + self.end_pos)
            print(f"boxList:{self.boxList}")
            print(f"ax_click:{self.axis_click}")
            print(f"type_click:{self.type_click}")
            self.listView_pt("box", f"{sx_real, sy_real, x, y}")
            if self.checkBox_Sync_to_Config_File.isChecked():
                self.sync_to_config_file(self.comboBox.currentText(), self.model.invisibleRootItem())
            try:
                if self.img_folder_path is not None :
                    image = os.path.join(self.img_folder_path, self.image_files[self.current_index])
                elif self.slice_img_p is not None:
                    image = self.slice_img_p
                elif self.image is not None:
                    image = self.image
                masks, _ = PNG_inference.PNG_inference(
                    self.sam2_model,
                    image,
                    self.axis_click,
                    self.type_click,
                    self.boxList
                )
                self.video_segments[self.current_slice_idx] = {}
                for i, mask in enumerate(masks):
                    mask_to_nii = mask.squeeze(0).astype(np.uint8) if masks.shape[0] != 1 else mask.astype(np.uint8)
                    mask_to_nii = (mask_to_nii > 0) * (i + 1)
                    mask_to_nii = mask_to_nii.astype(np.uint8)
                    self.video_segments[self.current_slice_idx][i] = mask_to_nii

                if masks.shape[0] != 1:
                    for i, mask in enumerate(masks, start=0):
                        print("mask shape is", mask.shape)
                        print(i)
                        self.show_mask(mask.squeeze(0), i)
                else:
                    self.show_mask(masks)
            except Exception as e:
                print(f"error occur in generating mask: {e}")
        else:
            return

    def update_image_from_slider(self):
        try:
            #if file is image sequence
            if self.file_type==2:
                self.current_index = self.Slider_SelectFiles.value()
                self.clear_treeView_pt()
                self.show_image(self.current_index)
                # print("update from current_idx")
            else:
                self.current_slice_idx = self.Slider_SelectFiles.value()
                self.clear_treeView_pt()
                self.show_slice(self.current_slice_idx)
                # print("update from current_slice")
        except Exception as e:
            self.speaker(f"update_image_from_slider Error:{e}")

    def set_start_page(self):
        if len(self.image_files) != 0:
            self.start_page = self.current_index
        else:
            self.start_page = self.current_slice_idx
        self.CLB_start_page.setText(str(self.start_page))

    def set_end_page(self):
        if len(self.image_files) != 0:
            self.end_page = self.current_index
        else:
            self.end_page = self.current_slice_idx
        self.CLB_end_page.setText(str(self.end_page))

    def set_mask_opacity(self):
        self.maskOpacity = self.Slider_mask_opacity.value() / 10
        if len(self.mask_group) == 0:
            return
        for mask_item in self.mask_group.values():
            if mask_item is not None:
                mask_item.setOpacity(self.maskOpacity)

    def show_and_hide_mask(self, state):
        if state == Qt.Checked:
            if self.maskOpacity_temp is not None:
                self.maskOpacity = self.maskOpacity_temp
            if len(self.mask_group) == 0:
                return
            for mask_item in self.mask_group.values():
                if mask_item is not None:
                    mask_item.setOpacity(self.maskOpacity)
        if state == Qt.Unchecked:
            self.maskOpacity_temp = self.maskOpacity
            self.maskOpacity = 0
            if len(self.mask_group) == 0:
                return
            for mask_item in self.mask_group.values():
                if mask_item is not None:
                    mask_item.setOpacity(self.maskOpacity)
    def save_mask(self):
        if self.file_type == 1:
            self.save_nifti()
        elif self.file_type == 2:
            self.save_png(None, None)
        elif self.file_type == 3:
            pass
            # self.save_png()
        if self.checkBox_save_segmentation.isChecked():
            if self.file_type == 1:
                self.save_segmentation()
            elif self.file_type == 2:
                self.save_png()
            elif self.file_type == 3:
                self.save_segmentation(self.mask, self.image)
    def save_nifti(self,file_path=None):
        try:
            miss_frames = []
            for num in range(0, self.axis_nii_len):
                if num not in self.video_segments:
                    if self.is_force:
                        empty_mask = np.zeros(self.nii_img.get_slices_shape(self.axis), dtype=np.uint8)
                        self.video_segments[num] = {1: empty_mask}
                        continue
                    miss_frames.append(num)
            if len(miss_frames) != 0:
                self.speaker(f"there are some empty frames,please check:{miss_frames}")
                self.speaker("if you press F-save(Force-save),i will create empty mask for all frames left.")
                self.Button_Save_generation.setText("F-save")
                self.is_force = True
                return

            stack_array = None
            for out_frame_idx in self.video_segments.keys():
                # num_items = len(self.video_segments[out_frame_idx])
                mask_2d = np.zeros(self.nii_img.get_slices_shape(self.axis), dtype=np.uint8)
                # print(f"Number of items in self.video_segments[{out_frame_idx}]: {num_items}")
                for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                    if out_mask.shape[0] == 1: out_mask = out_mask.squeeze(0)
                    # out_mask += out_mask
                    mask_2d |= (out_mask !=0).astype(np.uint8)
                stack_array = self.nii_img.align_to_me(self.axis, mask_2d, out_frame_idx, mask_array=stack_array)
            if file_path is None:
                file_path = os.path.join(self.lineEdit_TargetFiles.text(), self.Rename.text())
            self.nii_img.save_seg(stack_array, file_path)
            self.statusBar.showMessage(f"save mask file: {file_path}.")
        except Exception as e:
            self.speaker(f"Error in save_nifti: {e}")

    def save_png(self, mask, image_path):
        file_path = self.lineEdit_TargetFiles.text()
        file_path = os.path.join(file_path, self.nii_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if isinstance(image_path, str):
            _, image_name = os.path.split(image_path)
            file_name = image_name.split(".")[0] + self.Rename.text() + ".png"
        else:
            file_name = str(self.current_slice_idx) + ".png"
        if self.checkBox_SaveOriginPNG.isChecked():
            raw_folder_path = os.path.join(file_path, "raw")
            os.makedirs(raw_folder_path, exist_ok=True)
        mask_folder_path = os.path.join(file_path, "mask")
        os.makedirs(mask_folder_path, exist_ok=True)

        color = np.array([255, 255, 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = np.clip(mask_image * 255, 0, 255).astype(np.uint8)

        mask_image_pil = Image.fromarray(mask_image)
        if self.checkBox_SaveOriginPNG.isChecked():
            file_path = os.path.join(mask_folder_path, file_name)
            mask_image_pil.save(file_path)
            image_pil = Image.fromarray(self.slice_img)
            img_file_path = os.path.join(raw_folder_path, file_name)
            image_pil.save(img_file_path)
        else:
            mask_image_pil.save(os.path.join(mask_folder_path, file_name))
        self.speaker("mask save: {}".format(file_path))
        self.statusBar.showMessage(f"save png: {file_path}")

    def save_segmentation(self, mask=None, image_path=None):
        file_path = self.lineEdit_TargetFiles.text()
        file_path = os.path.join(file_path, "segmentation")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # png or folder save
        if self.file_type ==3:#png
            _, image_name = os.path.split(image_path)
            file_name = image_name.split(".")[0] + self.Rename_2.text() + ".png"
            file_name = os.path.join(file_path, file_name)
            image = Image.open(image_path)
            image = np.array(image)
            mask = mask.astype(np.uint8)
            mask = mask.squeeze(0)
            mask_3channel = np.stack([mask] * 4, axis=-1)
            mask_image = mask_3channel * image
            result_image = Image.fromarray(mask_image.astype(np.uint8))
            result_image.save(file_name)
            self.speaker("segmentation save: {}".format(file_path))
        elif self.file_type==2:# img folder
            file_name = str(self.current_slice_idx) + ".png"
        elif self.file_type==1:
            file_path,file_name = os.path.split(self.nii_path)
            file_name = file_name.split(".")[0] + self.Rename_2.text() + ".nii.gz"
            print(file_name)
            masked_array =None
            if self.video_segments is None:
                return
            for i in range(self.axis_nii_len):
                slice,_ = self.nii_img.get_slice_array(self.axis, i)
                if i not in self.video_segments:
                    mask_2d = np.zeros_like(slice.shape,dtype=np.uint8)
                else:
                    mask_2d = np.zeros_like(slice.shape, dtype=np.uint8)
                    for out_obj_id, out_mask in self.video_segments[i].items():
                        if out_mask.shape[0] == 1: out_mask = out_mask.squeeze(0)
                        # out_mask |= out_mask
                    mask_2d = (out_mask != 0).astype(np.uint8)
                # if mask_2d is None:
                #     mask_2d = np.zeros_like(slice,dtype=np.uint8)

                seg_slice = slice*mask_2d
                masked_array=self.nii_img.align_to_me(self.axis, seg_slice, i, mask_array=masked_array)
            self.nii_img.save_seg(masked_array, os.path.join(file_path, file_name))
            self.speaker("segmentation save: {}".format(file_path),False,False,True)










    def _MG_prompt(self, check: bool):
        self.Model(1 if self.buttongroup.checkedId() == -1 else self.buttongroup.checkedId())
        if self.buttongroup.checkedId() == 1 or self.buttongroup.checkedId() == -1:
            self.speaker("dot model, you can change the model in the Prompt page.")
            # self.label_model.setText("Prompt:<b>Dot</b>")
        elif self.buttongroup.checkedId() == 2:
            self.speaker("box model, you can change the model in the Prompt page.")
            # self.label_model.setText("Prompt:<b>Box</b>")
        # if not check:
        #     self.radioButton_None.setChecked(True)
        #     self.Model(3)
        # self.label_model.setText("Prompt:<b>Not allowed</b>")

    def wheelEvent(self, event):
        if self.graphicsView.underMouse():
            try:
                if self.file_type == 2:
                    if event.angleDelta().y() > 0:
                        self.clear_treeView_pt()
                        self.current_index = max(self.current_index - 1, 0)

                    else:
                        self.clear_treeView_pt()
                        self.current_index = min(self.current_index + 1, len(self.image_files) - 1)

                    # Update the slider and show the new image
                    self.Slider_SelectFiles.setValue(self.current_index)
                    self.show_image(self.current_index)
                elif self.file_type == 1:
                    if event.angleDelta().y() > 0:  # Scroll up
                        self.current_slice_idx = max(self.current_slice_idx - 1,
                                                     0)  # Increment, but do not exceed the max index
                        self.clear_treeView_pt()

                    else:  # Scroll down
                        self.current_slice_idx = min(self.current_slice_idx + 1,
                                                     self.axis_nii_len - 1)  # Decrement, but do not go below 0
                        self.clear_treeView_pt()

                    self.Slider_SelectFiles.setValue(self.current_slice_idx)
                    # self.show_slice(self.current_slice_idx)
                    event.accept()
                else:
                    pass
            except Exception as e:
                self.speaker(f"WheelEvent Error:{e}", gui=False)

    def keyPressEvent(self, a0):
        if a0.key() == Qt.Key_V:
            self.CheckBox_mask_opacity.setChecked(not self.CheckBox_mask_opacity.isChecked())
        if a0.key() == Qt.Key_A:
            self.del_item_and_coord()

    def Model(self, id):
        if id == 1:
            self.modelSelect = 2
            # self.fine_View.setDragMode(QGraphicsView.NoDrag)
            self.label_model.setText("Prompt:<b>Dot</b>")
            self.speaker(
                "click model:Click with the mouse to select foreground (green dots) and background (red dots).")
        elif id == 2:
            self.modelSelect = 1
            # self.fine_View.setDragMode(QGraphicsView.NoDrag)
            self.label_model.setText("Prompt:<b>Box</b>")
            self.speaker("boxing model:Choose foreground  with box.")
        else:
            self.modelSelect = 3
            self.label_model.setText("Prompt:<b>Not allowed</b>")
            pass


    def propagateMask(self):
        try:
            self.Button_MG_prompt.setChecked(False)
            # self._MG_prompt()
            self.Button_Save_generation.setText("save")
            self.is_force = False
            self.speaker("Initialzing...")
            if self.start_page == None:
                self.set_start_page()
                self.CLB_start_page.setText(str(self.start_page))
            if self.end_page == None:
                self.end_page = self.image_length
                self.CLB_end_page.setText(str(self.end_page))
            if self.waiting_for_confirmation:
                self.speaker("Press again.")
                self.waiting_for_confirmation = False
                return
            self.waiting_for_confirmation = True
            if self.axis_click == [] and self.type_click == [] and self.boxList == []:
                self.speaker("the prompt is empty.Input something,then try again.")
                return
            file_path = self.source_path if self.file_type == 1 else self.img_folder_path
            if not self.inference_state:
                self.bar_embedding_calculate.show()
                self.inference_state,_ = CI.create_inference_state(self.v_sam2_model,
                                                                 file_path,
                                                                 prog_signal=self.bar_embedding_calculate.setValue,
                                                                 axis=self.axis,
                                                                 )
                self.bar_embedding_calculate.hide()
            else:
                if self.re_inintial_embedding:
                    self.inference_state,_ = CI.create_inference_state(self.v_sam2_model, file_path,
                                                                     axis=self.axis)
                else:
                    self.speaker("Attention: skip the process of embedding caculation.")
            if self.file_type == 1:
                nii_idx = self.current_slice_idx
            else:
                nii_idx = self.current_index
            print("type nii_idx",type(nii_idx))
            if self.CheckBox_to_two_sides.isChecked():
                # 正向迭代器
                gen_forward = CI._Continuous_inference(
                    self.v_sam2_model,
                    inference_state=self.inference_state,
                    ptsList=self.axis_click,
                    ptsTypeList=self.type_click,
                    box=self.boxList,
                    nii_idx=nii_idx + 1,
                    start_frame=nii_idx + 1,  # 从当前页开始
                    max_frame_num_to_track=self.end_page - nii_idx,  # 正向最大帧数
                    reverse=False,  # 正向
                    # prog_signal=self.bar_embedding_calculate.setValue
                )
                # 反向迭代器
                gen_reverse = CI._Continuous_inference(
                    self.v_sam2_model,
                    inference_state=deepcopy(self.inference_state),
                    ptsList=self.axis_click,
                    ptsTypeList=self.type_click,
                    box=self.boxList,
                    nii_idx=nii_idx - 1,
                    start_frame=self.current_slice_idx - 1,  # 同样从当前页
                    max_frame_num_to_track=nii_idx - self.start_page,  # 反向最大帧数
                    reverse=True,  # 反向
                    prog_signal=self.bar_embedding_calculate.setValue
                )
                masks = next(gen_forward)
                masks = next(gen_reverse)
                self.thread1 = MaskGeneratorThread(gen_forward)
                self.thread1.mask_generator_signal.connect(self.update_mask)
                self.thread1.finished.connect(self.on_finished)
                self.thread1.start()

                self.thread2 = MaskGeneratorThread(gen_reverse)
                self.thread2.mask_generator_signal.connect(self.update_mask)
                self.thread2.finished.connect(self.on_finished)
                self.thread2.start()

            else:
                start_page = self.start_page
                if self.start_page > self.end_page:
                    self.reverse = True
                    self.speaker(f"start:{self.start_page}")
                    self.speaker(f"end:{self.end_page}")
                    self.speaker(
                        "Attention:The start_page more than the end_page,the progress of the propagation will be reversing. "
                        f"Ready to track {self.start_page-self.end_page} frames. ")
                    max_frame_num_to_track = self.start_page - self.end_page
                    # start_page = self.end_page
                else:
                    self.reverse = False
                    self.speaker(f"start:{self.start_page}")
                    self.speaker(f"end:{self.end_page}")
                    self.speaker(
                        f"Ready to track {self.end_page-self.start_page} frames. ")
                    max_frame_num_to_track = self.end_page - self.start_page
                # 单方向
                gen = CI._Continuous_inference(
                    self.v_sam2_model,
                    inference_state=self.inference_state,
                    ptsList=self.axis_click,
                    ptsTypeList=self.type_click,
                    box=self.boxList,
                    nii_idx=nii_idx,
                    start_frame=start_page,
                    max_frame_num_to_track=max_frame_num_to_track,
                    reverse=self.reverse,
                    prog_signal=self.bar_embedding_calculate.setValue
                )

                self.re_inintial_embedding = False
                masks = next(gen)
            if masks.shape[0] != 1:
                for i, mask in enumerate(masks, start=0):
                    self.show_mask(mask.squeeze(0), i)
            else:
                self.show_mask(masks)
            self.label_embedding.setText('Status: <span style="color:#178a2d; font-weight:600;">Ready</span>')
            self.label_embedding.setTextFormat(Qt.RichText)
            self.speaker("Ready!Press 'propagate' to  start seg.")

            while not self.continue_inference and self.waiting_for_confirmation:
                QApplication.processEvents()
            if not self.waiting_for_confirmation:
                self.speaker("等待用户确认的逻辑被终止")
                return
            # self.re_inintial = True
            self.waiting_for_confirmation = False
            self.Slider_SelectFiles.setValue(self.start_page)

            self.thread = MaskGeneratorThread(gen)
            self.thread.mask_generator_signal.connect(self.update_mask)
            self.thread.finished.connect(self.on_finished)
            self.thread.start()

        except Exception as e:
            self.speaker(f"Error in propagate mask: {e}")

    def _thread_stop(self):
        self.thread.start()

    def _thread_continue(self):
        if self.thread.isRunning():
            self.continue_inference = True
            self.speaker("Continue inference.")
        else:
            self.speaker("Thread is not running.")

    def update_mask(self, out_frame_idx, out_obj_ids, mask):
        try:
            self.boxList = []
            self.axis_click = []
            self.type_click = []
            masks = deepcopy(mask)
            # masks = mask
            if not self.CheckBox_to_two_sides.isChecked():
                self.show_image(out_frame_idx) #here mixed use a function of show_image and show_slice
                self.Slider_SelectFiles.setStyleSheet(open('ui/QScrollBar_styles.qss', encoding='utf-8').read())

                self.Slider_SelectFiles.setValue(out_frame_idx)
            for i, mask in list(masks.items()):
                if not self.CheckBox_to_two_sides.isChecked():
                    self.show_mask(mask, i - 1)
                mask = (mask > 0) * i
            self.video_segments[out_frame_idx] = {
                out_obj_ids[j]: ((masks[out_obj_id] > 0) * out_obj_ids[j]).astype(np.uint8) for j, out_obj_id in
                enumerate(out_obj_ids, start=0)
            }
            if self.CheckBox_to_two_sides.isChecked():
                self.bar_embedding_calculate.setFormat("Propagate...%p%")
                self.bar_embedding_calculate.show()
                QApplication.processEvents()
            if self.CheckBox_SavePNG.isChecked():
                self.save_png(mask, self.slice_img)
        except Exception as e:
            self.speaker(f"update mask error: {e}")

    def on_finished(self):
        self.continue_inference = False
        self.speaker("Complate.")
        self.Slider_SelectFiles.setStyleSheet("")
        self.bar_embedding_calculate.hide()

    def progress_bar_cal(self, signal):
        if signal == 0:
            self.bar_embedding_calculate.setFormat("Complete...%p%")
            self.label_embedding.setText('Status: <span style="color:#178a2d; font-weight:600;">Ready</span>')
            self.bar_embedding_calculate.hide()
        if signal == 1:
            self.label_embedding.setText(
                'Status: <span style="color:#1976D2; font-weight:700;">Embedding calculating</span>')
            self.bar_embedding_calculate.setFormat("Computing Embedding...%p%")

            self.bar_embedding_calculate.show()

    def recieve_infrence_state(self, inference_state):
        self.inference_state = inference_state
    def speaker(self, text, console = True, gui = True,statusbar = False):
        if console:
            print(f"{text}")
        if gui:
            self.textBrowser.append(f"{text}")
        if statusbar:
            self.statusBar.showMessage(f"{text}")
class MaskGeneratorThread(QThread):
    mask_generator_signal = pyqtSignal(int, list, dict)
    finished = pyqtSignal()

    def __init__(self, gen):
        super().__init__()
        self.gen = gen
        self.is_running = False
        self.stop_flag = False
        print("创建了一个Mask生成器。")

    def run(self):
        try:
            self.is_running = True
            mask ={}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.gen:
                if self.stop_flag:
                    break
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask[out_obj_id]=(out_mask_logits[i] > 0.0).cpu().numpy()
                 # mask= (out_mask_logits[i] > 0.0).cpu().numpy()
                self.mask_generator_signal.emit(out_frame_idx, out_obj_ids, copy.deepcopy(mask))
            self.is_running = False
            self.finished.emit()
        except Exception as e:
            print(f"thread error: {e}")

    def stop(self):
        self.stop_flag = False
        self.wait()

class calcute_embedding_thread(QThread):
    prog_signal = pyqtSignal(int)
    finished = pyqtSignal(object)
    prog_show_and_hide=pyqtSignal(int)
    def __init__(self, predictor,video_dir,axis):
        super().__init__()
        self.predictor = predictor
        self.video_dir = video_dir
        self.axis = axis
    @pyqtSlot()
    def run(self):
        self.prog_show_and_hide.emit(1)
        inference_state = CI.create_inference_state(self.predictor,
                                                    self.video_dir,
                                                    prog_signal=self.prog_signal.emit,
                                                    axis=self.axis,
                                                    )
        self.finished.emit(inference_state)
        self.prog_show_and_hide.emit(0)

class Multi_gen_mask(QThread):
    class Signals(QObject):
        result = pyqtSignal(object)
        mask_generator_signal = pyqtSignal(int, int, np.ndarray)
        seg_dict = pyqtSignal(str,dict)
        progress = pyqtSignal(int)
    def __init__(self, predictor, video_dir_list, axis,prog_signal=None,pt=None,ptTp=None,box=None):
        super().__init__()
        self.predictor = predictor
        self.video_dir_list = video_dir_list
        self.axis = axis
        self.prog_signal = prog_signal
        self.signals = Multi_gen_mask.Signals()
        self.pt=pt
        self.ptTp=ptTp
        self.box=box
    def run(self):
        for idx, video_dir in enumerate(self.video_dir_list):
            self.signals.progress.emit(int(f"{idx}0"))
            video_segments = {"name":video_dir}
            inference_state = CI.create_inference_state(self.predictor,
                                                        video_dir,
                                                        axis=self.axis,
                                                        )
            gen = CI.new_inference(
                self.predictor,
                inference_state=inference_state,
                ptsList=self.pt,
                ptsTypeList=self.ptTp,
                box=[],
            )#
            self.signals.progress.emit(int(f"{idx}2"))
            for out_frame_idx, out_obj_ids, out_mask_logits in gen:
                video_segments[out_frame_idx] = {
                    out_obj_ids[i]: (out_mask_logits[i] > 0.0).cpu().numpy() for i,out_obj_id in enumerate(out_obj_ids,start=0)

                }
            self.signals.seg_dict.emit(video_dir,video_segments)
            self.signals.progress.emit(int(f"{idx}3"))
            # self.signals.result.emit(res)
        self.signals.progress.emit(int(f"{idx}4"))
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

