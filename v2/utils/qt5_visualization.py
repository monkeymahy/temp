import json  # 标准库：JSON读写
import pickle  # 标准库：Pickle读写
import os  # 标准库：环境变量与路径
import sys  # 标准库：模块搜索路径
import time  # 标准库：计时
from pathlib import Path  # 标准库：路径对象
from typing import Dict, List, Optional, Tuple  # 类型标注

project_root = Path(__file__).resolve().parents[2]  # 项目根目录
if str(project_root) not in sys.path:  # 避免重复追加
    sys.path.append(str(project_root))  # 追加根路径

from OCC.Display.backend import load_backend  # OCC显示后端加载
from OCC.Display.OCCViewer import rgb_color  # OCC颜色
from OCC.Core.AIS import AIS_ColoredShape  # OCC着色形体
from OCC.Core.Bnd import Bnd_Box  # 包围盒
from OCC.Core.BRepBndLib import brepbndlib_Add  # 包围盒计算
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # 曲面适配器
from OCC.Core.BRepTools import breptools_UVBounds  # 曲面UV范围
from OCC.Core.GeomLProp import GeomLProp_SLProps  # 曲面微分属性
from OCC.Extend.TopologyUtils import TopologyExplorer  # OCC拓扑遍历
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Vec  # 几何对象
from PyQt5.QtCore import Qt, QEvent  # Qt常量
from PyQt5.QtGui import QKeySequence  # 快捷键序列
from PyQt5.QtWidgets import (  # Qt控件
    QApplication,  # 应用入口
    QWidget,  # 基础控件
    QPushButton,  # 按钮
    QHBoxLayout,  # 水平布局
    QGridLayout,  # 网格布局
    QVBoxLayout,  # 垂直布局
    QDialog,  # 对话框基类
    QFileDialog,  # 文件对话框
    QMessageBox,  # 消息提示框
    QListWidget,  # 列表控件
    QLabel,  # 文本标签
    QInputDialog,  # 输入对话框
    QMenu,  # 菜单
    QAction,  # 菜单项
    QToolButton,  # 工具按钮
    QShortcut,  # 快捷键
    QComboBox,  # 下拉选择框
)
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED  # 拓扑类型

load_backend("qt-pyqt5")  # 指定Qt后端
import OCC.Display.qtDisplay as qtDisplay  # OCC的Qt显示封装

import numpy as np  # 数值计算
import torch  # PyTorch推理
import yaml  # YAML解析

from dataset.AAGExtractor import AAGExtractor, TopologyChecker  # AAG提取与拓扑检查
from utils.data_utils import load_body_from_step  # STEP读取
from v2.models.segmentors import AAGNetSegmentor  # Lightning模型
from v2.utils.data_utils import (  # v2数据工具
    load_one_graph,  # 构图
    load_json_or_pkl,  # 读取配置
    load_statistics,  # 读取统计量
    standardization,  # 特征标准化
    center_and_scale,  # 居中缩放
    extract_labels_from_payload,  # 标签解析
    normalize_label_payload,  # 标签结构标准化
    append_label_version,  # 版本追加
    rollback_payload,  # 版本回滚
)


class FeatureClass:  # 类别聚合信息
    def __init__(self, name: str, faces: List[int]):  # 初始化
        self.name = name  # 类别名称
        self.faces = faces  # 面索引列表


class App(QDialog):  # 主界面
    def __init__(
        self,
        config_path: Optional[Path] = None,
        ckpt_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        super().__init__()  # 初始化父类

        # UI设置
        self.title = "AAGNet可视化工具（v2,Lightning）"  # 标题
        self.left = 300  # 初始X
        self.top = 300  # 初始Y
        self.width = 1366  # 初始宽度
        self.height = 900  # 初始高度
        self.canvas_width = 1000  # 画布宽度
        self.height_width = 700  # 画布高度

        # 成员变量
        self.gt_ais_shape = None  # GT视口OCC显示对象
        self.pred_ais_shape = None  # Prediction视口OCC显示对象
        self.file_name = None  # STEP路径
        self.faces_list = []  # 面列表
        self.pred_features_list: List[FeatureClass] = []  # 预测类别聚合
        self.pred_class_index_by_row: List[int] = []  # 预测列表行到类别索引
        self.gt_features_list: List[FeatureClass] = []  # GT类别聚合
        self.gt_class_index_by_row: List[int] = []  # GT列表行到类别索引
        self.step_dir: Optional[Path] = None  # STEP文件夹
        self.label_dir: Optional[Path] = None  # 标签文件夹
        self.gt_enabled = False  # GT显示开关
        self.step_files: List[Path] = []  # STEP文件列表
        self.last_step_file: Optional[Path] = None  # 上次选择的STEP文件
        self.face_index_by_hash: Dict[int, int] = {}  # 面哈希到索引
        self.face_select_enabled = False  # 面选择开关
        self.selected_face_indices_gt: List[int] = []  # GT当前选中面索引列表
        self.selected_face_indices_pred: List[int] = []  # Prediction当前选中面索引列表
        self.current_gt_labels: Optional[List[int]] = None  # 当前GT标签
        self.current_pred_labels: Optional[List[int]] = None  # 当前预测标签
        self.gt_label_path: Optional[Path] = None  # 当前GT文件路径
        self.gt_label_format: Optional[str] = None  # 当前GT格式
        self.gt_label_data = None  # 当前GT原始数据
        self.pred_label_path: Optional[Path] = None  # 当前预测保存路径
        self.pred_label_format: Optional[str] = None  # 当前预测保存格式
        self.pred_label_data = None  # 当前预测模板数据
        self.view_sync_lock = False  # 视口同步重入锁
        self._viewport_press_pos = {"gt": None, "pred": None}  # 视口左键按下位置
        self._viewport_dragging = {"gt": False, "pred": False}  # 视口是否处于拖拽旋转
        self._suppress_next_select = False  # 拖拽释放后抑制一次选择回调
        self.undo_stack: List[Dict] = []  # 撤销栈
        self.redo_stack: List[Dict] = []  # 重做栈
        self.dialog_dirs: Dict[str, Optional[Path]] = {  # 各类对话框最近目录
            "config": None,
            "ckpt": None,
            "step_file": None,
            "step_folder": None,
            "label_folder": None,
            "save_pred": None,
        }
        self.step_avg_confidence_by_path: Dict[str, float] = {}  # STEP平均置信度缓存
        self.step_pred_label_cache: Dict[str, List[int]] = {}  # STEP预测标签缓存
        self.displayed_step_files: List[Path] = []  # 当前列表显示顺序
        self.step_sort_mode = "name"  # STEP排序模式（name/confidence_desc/confidence_asc）
        self.label_author = os.getenv("AAGNET_USER") or os.getenv("USERNAME") or "unknown"
        self.gt_version_items: List[Dict[str, Optional[int]]] = []  # GT版本列表条目
        self.gt_session_base_labels: Optional[List[int]] = None  # 本次窗口基准标签
        self.gt_session_dirty = False  # 本次窗口是否有修改
        self.gt_version_preview_active = False  # 是否处于版本预览模式

        self.gt_display = None  # GT显示对象
        self.pred_display = None  # Prediction显示对象
        self.gt_canvas = None  # GT画布
        self.pred_canvas = None  # Prediction画布

        # 运行时设置
        self.project_root = Path(__file__).resolve().parents[1]  # v2根目录
        self.config_path = config_path or self.project_root / "configs" / "sf_csy.yaml"  # 默认配置
        self.ckpt_path = ckpt_path  # 权重路径
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # 推理设备

        # 数据/模型配置（由_load_config填充）
        self.model_cfg: Dict = {}  # 模型配置
        self.data_cfg: Dict = {}  # 数据配置
        self.label_names: List[str] = []  # 类别名称
        self.normalize = False  # 标准化开关
        self.do_center_and_scale = False  # 居中缩放开关
        self.attribute_schema = None  # 特征schema
        self.stat = None  # 标准化统计量
        self.recognizer = None  # 推理模型实例
        self.model_dirty = True  # 模型是否需要重新加载
        self.model_load_error: Optional[str] = None  # 最近一次模型加载错误

        self.topo_checker = TopologyChecker()  # 拓扑检查器

        self._load_state()  # 恢复上次状态

        self._load_config()  # 读取配置
        self._load_feature_schema()  # 读取特征schema
        self._load_statistics_if_needed()  # 读取统计量
        self.model_dirty = True  # 启动时延迟加载模型
        self.initUI()  # 初始化界面

    def initUI(self):
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )  # 强制使用可最大化的标准窗口标志
        self.setWindowTitle(self.title)  # 设置窗口标题
        self.setGeometry(self.left, self.top, self.width, self.height)  # 设置位置与大小
        self.setMinimumSize(900, 600)  # 设置最小尺寸
        self.setMaximumSize(16777215, 16777215)  # 允许系统最大化
        self.createHorizontalLayout()  # 创建布局
        self._update_window_title()  # 更新标题与状态
        self.msgBox = QMessageBox()  # 消息框
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)  # 撤销快捷键
        self.undo_shortcut.activated.connect(self.undoLabelEdit)  # 绑定撤销
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)  # 重做快捷键
        self.redo_shortcut.activated.connect(self.redoLabelEdit)  # 绑定重做
        self.face_select_enabled = True  # 默认开启面选择
        self._restore_step_list()  # 恢复STEP文件列表与选择

        windowLayout = QHBoxLayout()  # 主布局
        windowLayout.addWidget(self.button_panel)  # 左侧面板
        windowLayout.addWidget(self.horizontalGroupBox)  # 右侧画布
        self.setLayout(windowLayout)  # 设置布局
        self.show()  # 显示窗口

    def _load_config(self):
        if not self.config_path or not Path(self.config_path).exists():  # 配置不存在
            return  # 直接返回
        with open(self.config_path, "r", encoding="utf-8") as f:  # 打开配置
            config = yaml.safe_load(f)  # 解析YAML

        self.model_cfg = config.get("model", {}) or {}  # 读取模型配置
        self.data_cfg = config.get("data", {}) or {}  # 读取数据配置

        self.normalize = bool(self.data_cfg.get("normalize", False))  # 标准化开关
        self.do_center_and_scale = bool(self.data_cfg.get("center_and_scale", False))  # 居中缩放开关
        self.label_names = self._resolve_label_names(  # 解析类别名称
            config_path=Path(self.config_path),  # 配置路径
            num_classes=self.model_cfg.get("num_classes"),  # 类别数量
        )

    def _resolve_label_names(self, config_path: Path, num_classes: Optional[int]) -> List[str]:
        """从配置路径解析类别名称：优先读取label_names，否则生成默认名"""
        if not config_path or not config_path.exists():  # 配置不存在
            return []  # 返回空列表
        with open(config_path, "r", encoding="utf-8") as f:  # 打开配置
            config = yaml.safe_load(f)  # 解析YAML
        data_cfg = config.get("data", {}) or {}  # 读取数据配置
        label_names = data_cfg.get("label_names")  # 读取类别名称
        if label_names:  # 若存在
            return list(label_names)  # 返回列表
        if num_classes is None or num_classes <= 0:  # 类别数量无效
            return []  # 返回空列表
        return [f"class_{i}" for i in range(num_classes)]  # 生成默认名

    def _init_model(self):
        """初始化分割模型：根据配置和权重路径创建AAGNetSegmentor推理器"""
        self.model_load_error = None  # 清空历史错误
        if not self.model_cfg:  # 模型配置为空
            self.recognizer = None  # 清空模型实例
            self.model_load_error = "模型配置为空，请先加载有效配置文件"
            return  # 跳过初始化
        if not self.ckpt_path:  # 未指定权重路径
            self.recognizer = None  # 清空模型实例
            self.model_load_error = "未选择权重文件，请先加载权重"
            return  # 等待用户选择权重后再加载
        ckpt_str = str(self.ckpt_path)  # 权重路径字符串
        try:
            self.recognizer = AAGNetSegmentor.load_from_checkpoint(  # 加载Lightning模型
                checkpoint_path=ckpt_str,  # 权重文件路径
                **self.model_cfg,  # 模型超参数配置
                map_location=self.device,  # 加载到指定设备
            )
            self.recognizer.to(self.device)  # 移动模型到推理设备
            self.recognizer.eval()  # 切换为评估模式（关闭Dropout等）
        except Exception as e:
            self.recognizer = None  # 失败时保持为空
            self.model_load_error = f"模型加载失败（配置与权重可能不匹配）: {e}"

    def _ensure_model_ready(self) -> bool:
        """仅在需要推理时加载/校验模型，失败时返回False并保留错误信息"""
        if self.model_dirty or self.recognizer is None:
            self._init_model()  # 延迟加载
            self.model_dirty = False  # 本轮已尝试加载
        return self.recognizer is not None

    def _load_feature_schema(self):
        """加载特征schema：从feature_lists目录读取AAG特征定义"""
        feature_schema_path = self.project_root.parent / "feature_lists" / "all.json"  # schema路径
        cfg_schema = self.data_cfg.get("feature_schema") if isinstance(self.data_cfg, dict) else None  # 配置中的schema路径
        if cfg_schema:  # 优先使用配置指定schema
            cfg_schema_path = Path(cfg_schema)
            if not cfg_schema_path.is_absolute():
                cfg_schema_path = (self.project_root.parent / cfg_schema_path).resolve()
            if cfg_schema_path.exists():
                feature_schema_path = cfg_schema_path
        if feature_schema_path.exists():  # 若存在
            self.attribute_schema = load_json_or_pkl(feature_schema_path)  # 读取schema

    def _resolve_data_root(self) -> Optional[Path]:
        root_dir = self.data_cfg.get("root_dir") if isinstance(self.data_cfg, dict) else None  # 数据根目录
        if not root_dir:
            return None
        root_path = Path(root_dir)
        if not root_path.is_absolute():
            root_path = (self.project_root.parent / root_path).resolve()
        if root_path.exists():
            return root_path
        return None

    def _load_precomputed_aag_data(self) -> Optional[Dict]:
        if not self.file_name:
            return None
        step_path = Path(self.file_name)
        candidates: List[Path] = []

        data_root = self._resolve_data_root()
        if data_root is not None:
            candidates.append(data_root / "aag" / f"{step_path.stem}.json")

        if step_path.parent.name.lower() == "steps":
            candidates.append(step_path.parent.parent / "aag" / f"{step_path.stem}.json")

        for aag_path in candidates:
            if not aag_path.exists():
                continue
            try:
                payload = load_json_or_pkl(aag_path)
            except Exception:
                continue
            if isinstance(payload, list) and len(payload) == 2 and isinstance(payload[1], dict):
                return payload[1]
            if isinstance(payload, dict):
                return payload
        return None

    def _validate_graph_feature_dims(self, graph) -> Optional[str]:
        node_attr_expected = int(self.model_cfg.get("node_attr_dim", 0) or 0)
        edge_attr_expected = int(self.model_cfg.get("edge_attr_dim", 0) or 0)
        node_grid_expected = int(self.model_cfg.get("node_grid_dim", 0) or 0)

        node_attr_actual = int(graph.ndata["x"].shape[1]) if "x" in graph.ndata else 0
        edge_attr_actual = int(graph.edata["x"].shape[1]) if "x" in graph.edata else 0
        node_grid_actual = int(graph.ndata["grid"].shape[1]) if "grid" in graph.ndata else 0

        if node_attr_expected and node_attr_actual != node_attr_expected:
            return f"节点属性维度不匹配：期望 {node_attr_expected}，实际 {node_attr_actual}"
        if edge_attr_expected and edge_attr_actual != edge_attr_expected:
            return f"边属性维度不匹配：期望 {edge_attr_expected}，实际 {edge_attr_actual}"
        if node_grid_expected and node_grid_actual != node_grid_expected:
            return f"面网格通道数不匹配：期望 {node_grid_expected}，实际 {node_grid_actual}"
        return None

    def _load_statistics_if_needed(self):
        """延迟加载统计量：仅当标准化开关打开时才读取统计文件，节省内存"""
        if not self.normalize:  # 标准化未启用
            return  # 跳过加载
        stat_path = self.project_root.parent / "training_data" / "stat.json"  # 统计量路径
        if stat_path.exists():  # 文件存在
            self.stat = load_statistics(stat_path)  # 读取统计量

    def createHorizontalLayout(self):
        """创建主界面布局：左侧控制面板+右侧3D画布"""
        self.horizontalGroupBox = QWidget()  # 右侧容器
        canvas_layout = QHBoxLayout()  # 右侧布局
        self.button_panel = QWidget()  # 左侧面板
        self.button_panel.setFixedWidth(220)  # 固定宽度
        panel_layout = QGridLayout()  # 左侧网格布局

        # 创建窗口大小调整菜单
        window_menu = QMenu("调整窗口大小", self)  # 窗口菜单
        action_small = QAction("小", self)  # 小尺寸
        action_small.triggered.connect(lambda: self._resize_window(1100, 700))  # 绑定事件
        action_medium = QAction("中", self)  # 中尺寸
        action_medium.triggered.connect(lambda: self._resize_window(1366, 900))  # 绑定事件
        action_large = QAction("大", self)  # 大尺寸
        action_large.triggered.connect(lambda: self._resize_window(1600, 1000))  # 绑定事件
        window_menu.addAction(action_small)  # 添加菜单项
        window_menu.addAction(action_medium)  # 添加菜单项
        window_menu.addAction(action_large)  # 添加菜单项

        # 创建窗口大小调整按钮
        self.window_button = QToolButton(self)  # 工具按钮
        self.window_button.setText("调整窗口大小")  # 按钮文字
        self.window_button.setMenu(window_menu)  # 绑定菜单
        self.window_button.setPopupMode(QToolButton.InstantPopup)  # 点击立即弹出
        self.window_button.setToolButtonStyle(Qt.ToolButtonTextOnly)  # 仅显示文字
        self.window_button.setStyleSheet(  # 设置按钮样式
            "QToolButton {padding: 6px 12px; border: 1px solid #888; "
            "border-radius: 4px; background: #f2f2f2;}"
            "QToolButton:hover {background: #e6e6e6;}"
            "QToolButton::menu-indicator {image: none;}"
        )
        panel_layout.addWidget(self.window_button, 0, 0, 1, 1, alignment=Qt.AlignLeft)  # 放置按钮

        # 创建状态标签
        self.status_label = QLabel("", self)  # 状态标签
        self.status_label.setWordWrap(True)  # 自动换行
        panel_layout.addWidget(self.status_label, 1, 0, 1, 1)  # 放置标签

        # 创建配置加载按钮
        btn_config = QPushButton("加载配置", self)  # 配置按钮
        btn_config.clicked.connect(self.openConfig)  # 绑定点击事件
        panel_layout.addWidget(btn_config, 2, 0, 1, 1)  # 放置按钮

        # 创建权重加载按钮
        btn_ckpt = QPushButton("加载权重", self)  # 权重按钮
        btn_ckpt.clicked.connect(self.openCheckpoint)  # 绑定点击事件
        panel_layout.addWidget(btn_ckpt, 3, 0, 1, 1)  # 放置按钮

        # 创建STEP加载按钮
        btn_load_step = QPushButton("加载 STEP", self)  # STEP按钮
        btn_load_step.clicked.connect(self.openShape)  # 绑定点击事件
        panel_layout.addWidget(btn_load_step, 4, 0, 1, 1)  # 放置按钮

        # 创建STEP文件夹加载按钮
        btn_load_step_dir = QPushButton("加载STEP文件夹", self)  # STEP文件夹按钮
        btn_load_step_dir.clicked.connect(self.openStepFolder)  # 绑定点击事件
        panel_layout.addWidget(btn_load_step_dir, 5, 0, 1, 1)  # 放置按钮

        # 创建标签文件夹加载按钮
        btn_load_label_dir = QPushButton("加载标签文件夹", self)  # 标签文件夹按钮
        btn_load_label_dir.clicked.connect(self.openLabelFolder)  # 绑定点击事件
        panel_layout.addWidget(btn_load_label_dir, 6, 0, 1, 1)  # 放置按钮

        # 创建STEP关闭按钮
        btn_close_step = QPushButton("关闭 STEP", self)  # 关闭按钮
        btn_close_step.clicked.connect(self.eraseShape)  # 绑定点击事件
        panel_layout.addWidget(btn_close_step, 7, 0, 1, 1)  # 放置按钮

        # 创建分割执行按钮
        btn_segmentation = QPushButton("运行分割", self)  # 分割按钮
        btn_segmentation.clicked.connect(self.featureRecog)  # 绑定点击事件
        panel_layout.addWidget(btn_segmentation, 8, 0, 1, 1)  # 放置按钮

        # 创建批量分割按钮
        btn_batch_segmentation = QPushButton("批量预测STEP列表", self)  # 批量分割按钮
        btn_batch_segmentation.clicked.connect(self.batchFeatureRecog)  # 绑定点击事件
        panel_layout.addWidget(btn_batch_segmentation, 9, 0, 1, 1)  # 放置按钮

        # 创建GT显示按钮
        self.btn_toggle_gt = QPushButton("显示GT标签", self)  # GT显示按钮
        self.btn_toggle_gt.clicked.connect(self.toggleGTVisualization)  # 绑定点击事件
        panel_layout.addWidget(self.btn_toggle_gt, 10, 0, 1, 1)  # 放置按钮

        # 创建预测保存按钮
        btn_save_pred = QPushButton("保存预测标签", self)  # 保存预测按钮
        btn_save_pred.clicked.connect(self.savePredictionLabels)  # 绑定点击事件
        panel_layout.addWidget(btn_save_pred, 11, 0, 1, 1)  # 放置按钮

        # 创建STEP文件列表头（左标题，右排序）
        self.step_header_widget = QWidget(self)  # STEP列表头容器
        step_header_layout = QHBoxLayout()  # STEP列表头布局
        step_header_layout.setContentsMargins(0, 0, 0, 0)  # 去边距
        self.step_list_label = QLabel("STEP文件列表", self)  # STEP列表标题
        self.stepSortCombo = QComboBox(self)  # 排序选择框
        self.stepSortCombo.addItems(["按名称排序", "按平均置信度排序", "按最低置信度排序"])  # 排序选项
        self.stepSortCombo.currentIndexChanged.connect(self._on_step_sort_changed)  # 绑定切换事件
        step_header_layout.addWidget(self.step_list_label)  # 左侧标题
        step_header_layout.addStretch(1)  # 中间弹性
        step_header_layout.addWidget(self.stepSortCombo)  # 右侧排序
        self.step_header_widget.setLayout(step_header_layout)  # 设置头布局
        panel_layout.addWidget(self.step_header_widget, 12, 0, 1, 1)  # 放置标题栏

        self.stepListWidget = QListWidget()  # STEP列表
        self.stepListWidget.itemClicked.connect(self.stepListItemClicked)  # 绑定点击事件
        panel_layout.addWidget(self.stepListWidget, 13, 0, 1, 1)  # 放置列表

        # 创建双视口容器（左GT，右Prediction）
        viewports_container = QWidget()  # 双视口容器
        viewports_layout = QHBoxLayout()  # 双视口布局
        viewports_layout.setContentsMargins(0, 0, 0, 0)  # 去边距

        gt_view_widget = QWidget()  # GT视口容器
        gt_view_layout = QVBoxLayout()  # GT视口布局
        gt_view_layout.setContentsMargins(0, 0, 0, 0)  # 去边距
        gt_label = QLabel("Ground Truth", self)  # GT标题
        gt_view_layout.addWidget(gt_label)  # 添加标题
        self.gt_canvas = qtDisplay.qtViewer3d(self)  # GT画布
        gt_view_layout.addWidget(self.gt_canvas, stretch=1)  # 添加GT画布
        gt_view_widget.setLayout(gt_view_layout)  # 设置布局

        pred_view_widget = QWidget()  # Prediction视口容器
        pred_view_layout = QVBoxLayout()  # Prediction视口布局
        pred_view_layout.setContentsMargins(0, 0, 0, 0)  # 去边距
        pred_label = QLabel("Prediction", self)  # Prediction标题
        pred_view_layout.addWidget(pred_label)  # 添加标题
        self.pred_canvas = qtDisplay.qtViewer3d(self)  # Prediction画布
        pred_view_layout.addWidget(self.pred_canvas, stretch=1)  # 添加Prediction画布
        pred_view_widget.setLayout(pred_view_layout)  # 设置布局

        viewports_layout.addWidget(gt_view_widget, stretch=1)  # 添加GT视口
        viewports_layout.addWidget(pred_view_widget, stretch=1)  # 添加Prediction视口
        viewports_container.setLayout(viewports_layout)  # 设置双视口布局
        canvas_layout.addWidget(viewports_container, stretch=1)  # 添加容器

        self.gt_canvas.resize(self.canvas_width // 2, self.height_width)  # 设置GT画布尺寸
        self.pred_canvas.resize(self.canvas_width // 2, self.height_width)  # 设置Prediction画布尺寸
        self.gt_canvas.InitDriver()  # 初始化GT渲染驱动
        self.pred_canvas.InitDriver()  # 初始化Prediction渲染驱动
        self.gt_display = self.gt_canvas._display  # GT显示对象
        self.pred_display = self.pred_canvas._display  # Prediction显示对象

        # 创建右侧面列表面板（GT/Prediction分离）
        self.face_panel = QWidget()  # 面列表容器
        self.face_panel.setFixedWidth(200)  # 固定宽度
        face_layout = QVBoxLayout()  # 面列表布局

        self.gt_face_list_label = QLabel("GT面列表", self)  # GT面列表标题
        face_layout.addWidget(self.gt_face_list_label)  # 放置标题

        self.gtFaceListWidget = QListWidget()  # GT面列表
        self.gtFaceListWidget.setSelectionMode(QListWidget.ExtendedSelection)  # 允许Ctrl多选
        self.gtFaceListWidget.itemClicked.connect(self.faceListItemClickedGT)  # 绑定点击事件
        self.gtFaceListWidget.setContextMenuPolicy(Qt.CustomContextMenu)  # 启用自定义右键
        self.gtFaceListWidget.customContextMenuRequested.connect(self.faceListContextMenuGT)  # 绑定右键菜单
        face_layout.addWidget(self.gtFaceListWidget)  # 放置列表

        self.pred_face_list_label = QLabel("Prediction面列表", self)  # Prediction面列表标题
        face_layout.addWidget(self.pred_face_list_label)  # 放置标题

        self.predFaceListWidget = QListWidget()  # Prediction面列表
        self.predFaceListWidget.setSelectionMode(QListWidget.ExtendedSelection)  # 允许Ctrl多选
        self.predFaceListWidget.itemClicked.connect(self.faceListItemClickedPred)  # 绑定点击事件
        self.predFaceListWidget.setContextMenuPolicy(Qt.CustomContextMenu)  # 启用自定义右键
        self.predFaceListWidget.customContextMenuRequested.connect(self.faceListContextMenuPred)  # 绑定右键菜单
        face_layout.addWidget(self.predFaceListWidget)  # 放置列表

        self.face_panel.setLayout(face_layout)  # 设置布局
        canvas_layout.addWidget(self.face_panel)  # 添加右侧面板

        # 创建GT与Prediction特征列表（上下布局）
        self.gtFeatureListLabel = QLabel("GT标注列表", self)  # GT标注列表标题
        panel_layout.addWidget(self.gtFeatureListLabel, 14, 0, 1, 1)  # 放置标题
        self.gtFeatureListWidget = QListWidget()  # GT特征列表
        self.gtFeatureListWidget.setSelectionMode(QListWidget.ExtendedSelection)  # 支持多类联动选中
        self.gtFeatureListWidget.itemClicked.connect(self.gtFeatureListDoubleClicked)  # 绑定点击事件
        panel_layout.addWidget(self.gtFeatureListWidget, 15, 0, 1, 1)  # 放置列表

        self.gtVersionListLabel = QLabel("GT版本列表", self)  # GT版本列表标题
        panel_layout.addWidget(self.gtVersionListLabel, 16, 0, 1, 1)
        self.gtVersionListWidget = QListWidget()  # GT版本列表
        self.gtVersionListWidget.itemClicked.connect(self._on_gt_version_selected)
        panel_layout.addWidget(self.gtVersionListWidget, 17, 0, 1, 1)

        gt_version_btns = QWidget(self)
        gt_version_btns_layout = QHBoxLayout()
        gt_version_btns_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_load_gt_version = QPushButton("加载版本", self)
        self.btn_load_gt_version.clicked.connect(lambda: self._load_selected_gt_version(rollback=False))
        self.btn_rollback_gt_version = QPushButton("回滚保存", self)
        self.btn_rollback_gt_version.clicked.connect(lambda: self._load_selected_gt_version(rollback=True))
        gt_version_btns_layout.addWidget(self.btn_load_gt_version)
        gt_version_btns_layout.addWidget(self.btn_rollback_gt_version)
        gt_version_btns.setLayout(gt_version_btns_layout)
        panel_layout.addWidget(gt_version_btns, 18, 0, 1, 1)

        self.predFeatureListLabel = QLabel("Prediction标注列表", self)  # Prediction标注列表标题
        panel_layout.addWidget(self.predFeatureListLabel, 19, 0, 1, 1)  # 放置标题
        self.predFeatureListWidget = QListWidget()  # Prediction特征列表
        self.predFeatureListWidget.setSelectionMode(QListWidget.ExtendedSelection)  # 支持多类联动选中
        self.predFeatureListWidget.itemClicked.connect(self.predFeatureListDoubleClicked)  # 绑定点击事件
        panel_layout.addWidget(self.predFeatureListWidget, 20, 0, 1, 1)  # 放置列表

        # 应用布局
        self.horizontalGroupBox.setLayout(canvas_layout)  # 设置右侧布局
        self.button_panel.setLayout(panel_layout)  # 设置左侧布局

        self.gt_canvas.installEventFilter(self)  # 监听GT画布事件
        self.pred_canvas.installEventFilter(self)  # 监听Prediction画布事件
        if hasattr(self.gt_display, "register_select_callback"):
            self.gt_display.register_select_callback(lambda selected_shapes, *a, **k: self._on_select("gt", selected_shapes, *a, **k))  # 注册GT选择回调
        if hasattr(self.pred_display, "register_select_callback"):
            self.pred_display.register_select_callback(lambda selected_shapes, *a, **k: self._on_select("pred", selected_shapes, *a, **k))  # 注册Prediction选择回调

    def openConfig(self):
        default_dir = self._dialog_dir("config", self.project_root / "configs")  # 默认目录
        config_path = QFileDialog.getOpenFileName(  # 打开配置文件对话框
            self,  # 父窗口
            "Open Config YAML",  # 标题
            default_dir,  # 默认目录
            "(*.yaml *.yml)",  # 文件过滤
        )[
            0
        ]  # 取第一个结果
        if not config_path:  # 未选择
            return  # 直接返回
        self.config_path = Path(config_path)  # 更新配置路径
        self._set_dialog_dir("config", self.config_path.parent)  # 记录目录
        self._load_config()  # 重新读取配置
        self._load_statistics_if_needed()  # 重新读取统计量
        self.recognizer = None  # 延迟重建模型
        self.model_dirty = True  # 标记待加载
        self.model_load_error = None  # 清空错误
        self._save_state()  # 保存状态
        self._update_window_title()  # 更新标题

    def openCheckpoint(self):
        default_dir = self._dialog_dir("ckpt", self.project_root.parent / "output" / "checkpoints")  # 默认目录
        ckpt_path = QFileDialog.getOpenFileName(  # 打开权重文件对话框
            self,  # 父窗口
            "Open Checkpoint",  # 标题
            default_dir,  # 默认目录
            "(*.ckpt)",  # 文件过滤
        )[
            0
        ]  # 取第一个结果
        if not ckpt_path:  # 未选择
            return  # 直接返回
        self.ckpt_path = Path(ckpt_path)  # 更新权重路径
        self._set_dialog_dir("ckpt", self.ckpt_path.parent)  # 记录目录
        self.recognizer = None  # 延迟重建模型
        self.model_dirty = True  # 标记待加载
        self.model_load_error = None  # 清空错误
        self._save_state()  # 保存状态
        self._update_window_title()  # 更新标题

    def _resize_window(self, width: int, height: int):
        self.resize(width, height)  # 调整窗口尺寸

    def _update_window_title(self):
        config_name = self._short_name(self.config_path)  # 配置文件名
        ckpt_name = self._short_name(self.ckpt_path)  # 权重文件名
        step_name = self._short_name(self.file_name)  # STEP文件名
        title = f"{self.title} | 配置: {config_name} | 权重: {ckpt_name} | STEP: {step_name}"  # 标题文本
        self.setWindowTitle(title)  # 设置标题
        if hasattr(self, "status_label"):  # 若存在状态标签
            self.status_label.setText(f"配置: {config_name}\n权重: {ckpt_name}\nSTEP: {step_name}")  # 更新状态

    def _short_name(self, path_value) -> str:
        if not path_value:  # 空值处理
            return "未选择"  # 统一占位
        try:  # 正常解析
            return Path(path_value).name  # 仅返回文件名
        except Exception:  # 异常兜底
            return "未选择"  # 统一占位

    def openShape(self):
        """加载STEP文件：文件选择→读取→拓扑检查→显示→面列表记录"""
        default_dir = self._dialog_dir("step_file", self.step_dir or Path("./"))  # 默认路径
        step_file_path = QFileDialog.getOpenFileName(self, "选择STEP文件", default_dir, "(*.st*p)")[0]  # 打开文件对话框
        if not step_file_path:  # 用户取消选择
            return  # 直接返回
        self._set_dialog_dir("step_file", Path(step_file_path).parent)  # 记录目录
        self._load_step_file(Path(step_file_path))  # 加载选择的STEP文件

    def _load_step_file(self, step_path: Path):
        self._commit_gt_session_changes()
        self.file_name = str(step_path)  # 保存文件路径
        solid_shape = load_body_from_step(self.file_name)  # 读取STEP文件为实体
        if not self.topo_checker(solid_shape):  # 拓扑检查失败（非流形等）
            self.msgBox.warning(self, "警告", "加载失败，STEP文件不支持或存在拓扑错误")  # 提示用户
            self.file_name = None  # 清空文件路径
            return  # 结束处理

        if self.gt_ais_shape:  # 若已存在旧的GT显示对象
            self.gt_display.Context.Erase(self.gt_ais_shape, True)  # 清除旧GT对象
        if self.pred_ais_shape:  # 若已存在旧的Prediction显示对象
            self.pred_display.Context.Erase(self.pred_ais_shape, True)  # 清除旧Prediction对象

        self.gtFeatureListWidget.clear()  # 清空GT特征列表控件
        self.predFeatureListWidget.clear()  # 清空Prediction特征列表控件
        self.gt_features_list.clear()  # 清空GT特征聚合数据
        self.pred_features_list.clear()  # 清空Prediction特征聚合数据
        self.gt_class_index_by_row.clear()  # 清空GT索引映射
        self.pred_class_index_by_row.clear()  # 清空Prediction索引映射
        self.current_pred_labels = None  # 清空预测标签
        self.undo_stack.clear()  # 清空撤销栈
        self.redo_stack.clear()  # 清空重做栈

        self.gt_ais_shape = AIS_ColoredShape(solid_shape)  # 创建GT可着色显示对象
        self.pred_ais_shape = AIS_ColoredShape(solid_shape)  # 创建Prediction可着色显示对象
        self.gt_display.Context.Display(self.gt_ais_shape, True)  # 显示GT对象
        self.pred_display.Context.Display(self.pred_ais_shape, True)  # 显示Prediction对象
        self.gt_display.FitAll()  # GT自适应相机
        self.pred_display.FitAll()  # Prediction自适应相机
        self._sync_view_from_to("gt", "pred")  # 同步相机

        topology_explorer = TopologyExplorer(solid_shape)  # 创建拓扑遍历器
        self.faces_list = list(topology_explorer.faces())  # 提取并记录所有面对象
        self.selected_face_indices_gt.clear()  # 新模型加载后清空GT选择
        self.selected_face_indices_pred.clear()  # 新模型加载后清空Prediction选择
        self._build_face_index()  # 构建面索引
        self._populate_face_list()  # 更新面列表
        self._set_shape_all_gray(self.gt_ais_shape)  # 默认全灰
        self._set_shape_all_gray(self.pred_ais_shape)  # 默认全灰
        self.gt_display.Context.Display(self.gt_ais_shape, True)
        self.pred_display.Context.Display(self.pred_ais_shape, True)
        self._update_window_title()  # 更新窗口标题显示当前文件

        if self.face_select_enabled:  # 若启用选择模式
            self.gt_display.Context.Activate(self.gt_ais_shape, TopAbs_FACE, True)  # 启用GT面选择
            self.pred_display.Context.Activate(self.pred_ais_shape, TopAbs_FACE, True)  # 启用Prediction面选择

        self.last_step_file = step_path  # 记录最后选择
        self._apply_cached_prediction_for_current_step()  # 若有批量预测缓存则直接显示
        self._save_state()  # 保存状态

        if self.gt_enabled:  # 若已开启GT显示
            self._apply_gt_labels()  # 自动刷新GT颜色

    def _apply_cached_prediction_for_current_step(self):
        if not (self.file_name and self.pred_ais_shape):
            return
        step_key = str(Path(self.file_name))
        cached_labels = self.step_pred_label_cache.get(step_key)
        if not cached_labels:
            return
        if len(cached_labels) != len(self.faces_list):
            return
        self.current_pred_labels = list(cached_labels)
        self._apply_pred_labels_from_labels(self.current_pred_labels)

    def eraseShape(self):
        if self.gt_ais_shape or self.pred_ais_shape:  # 若存在对象
            self._commit_gt_session_changes()
            if self.gt_ais_shape:
                self.gt_display.Context.Erase(self.gt_ais_shape, True)  # 清除GT显示
            if self.pred_ais_shape:
                self.pred_display.Context.Erase(self.pred_ais_shape, True)  # 清除Prediction显示
            self.gt_ais_shape = None  # 清空GT对象
            self.pred_ais_shape = None  # 清空Prediction对象
            self.file_name = None  # 清空路径
            self.gtFeatureListWidget.clear()  # 清空GT列表
            self.predFeatureListWidget.clear()  # 清空Prediction列表
            self.gt_features_list.clear()  # 清空GT结果
            self.pred_features_list.clear()  # 清空Prediction结果
            self.faces_list.clear()  # 清空面列表
            self.gt_class_index_by_row.clear()  # 清空GT行映射
            self.pred_class_index_by_row.clear()  # 清空Prediction行映射
            self.face_index_by_hash.clear()  # 清空面哈希
            self.selected_face_indices_gt.clear()  # 清空GT选中面
            self.selected_face_indices_pred.clear()  # 清空Prediction选中面
            if hasattr(self, "gtFaceListWidget"):
                self.gtFaceListWidget.clear()  # 清空GT面列表
            if hasattr(self, "predFaceListWidget"):
                self.predFaceListWidget.clear()  # 清空Prediction面列表
            if hasattr(self, "gtVersionListWidget"):
                self.gtVersionListWidget.clear()  # 清空GT版本列表
            self.current_gt_labels = None  # 清空GT标签
            self.current_pred_labels = None  # 清空预测标签
            self.gt_label_path = None  # 清空GT路径
            self.gt_label_format = None  # 清空GT格式
            self.gt_label_data = None  # 清空GT数据
            self.gt_session_base_labels = None  # 清空GT窗口基准
            self.gt_session_dirty = False  # 清空GT窗口状态
            self.gt_version_preview_active = False
            self.pred_label_path = None  # 清空预测保存路径
            self.pred_label_format = None  # 清空预测保存格式
            self.pred_label_data = None  # 清空预测模板数据
            self.undo_stack.clear()  # 清空撤销栈
            self.redo_stack.clear()  # 清空重做栈
            self._update_window_title()  # 更新标题

    def featureRecog(self):
        """执行特征分割：AAG提取→图构建→推理→结果聚合→列表显示"""
        if not (self.pred_ais_shape and self.file_name and self.attribute_schema):  # 前置检查
            return  # 直接返回

        start_time = time.time()  # 开始计时
        labels, avg_confidence, err = self._predict_step(Path(self.file_name))  # 执行单文件预测
        if err:
            self.msgBox.warning(self, "warning", err)  # 提示
            return
        if labels is None:
            self.msgBox.warning(self, "warning", "预测失败：未返回标签")  # 提示
            return
        if len(labels) != len(self.faces_list):  # 防御性检查
            self.msgBox.warning(self, "warning", "预测标签数量与当前STEP面数量不一致")
            return

        class_to_faces_map: Dict[int, List[int]] = {}  # 类别到面索引的映射字典
        for face_idx, class_idx in enumerate(labels):  # 遍历预测结果
            class_to_faces_map.setdefault(class_idx, []).append(face_idx)  # 聚合同类别的面

        self.current_pred_labels = labels  # 缓存预测标签
        self._populate_pred_feature_list(class_to_faces_map)  # 更新预测特征列表
        self._apply_pred_labels_from_labels(self.current_pred_labels)  # 应用预测颜色

        if avg_confidence is not None:
            step_key = str(Path(self.file_name))
            self.step_avg_confidence_by_path[step_key] = float(avg_confidence)  # 缓存平均置信度
            self.step_pred_label_cache[step_key] = list(labels)  # 缓存预测标签
            self._populate_step_list()  # 刷新列表显示置信度/排序

        postprocess_time = time.time()  # 后处理完成时间
        print(f"后处理耗时: {postprocess_time - start_time:.3f}s")  # 输出耗时
        print(f"总耗时: {time.time() - start_time:.3f}s")  # 输出总耗时

    def _load_precomputed_aag_data_for_step(self, step_path: Path) -> Optional[Dict]:
        candidates: List[Path] = []

        data_root = self._resolve_data_root()
        if data_root is not None:
            candidates.append(data_root / "aag" / f"{step_path.stem}.json")

        if step_path.parent.name.lower() == "steps":
            candidates.append(step_path.parent.parent / "aag" / f"{step_path.stem}.json")

        for aag_path in candidates:
            if not aag_path.exists():
                continue
            try:
                payload = load_json_or_pkl(aag_path)
            except Exception:
                continue
            if isinstance(payload, list) and len(payload) == 2 and isinstance(payload[1], dict):
                return payload[1]
            if isinstance(payload, dict):
                return payload
        return None

    def _predict_step(self, step_path: Path) -> Tuple[Optional[List[int]], Optional[float], Optional[str]]:
        if not self.attribute_schema:
            return None, None, "特征schema未加载，请先加载配置"
        if not self._ensure_model_ready():
            return None, None, self.model_load_error or "模型未初始化，请先加载配置和权重"

        step_path_str = str(step_path)
        aag_data = self._load_precomputed_aag_data_for_step(step_path)
        if aag_data is None:
            try:
                aag_extractor = AAGExtractor(step_path_str, self.attribute_schema)
                aag_data = aag_extractor.process()
            except Exception as e:
                return None, None, f"AAG提取失败 ({step_path.name}): {e}"

        try:
            sample_dict = load_one_graph(step_path_str, aag_data)
        except Exception as e:
            return None, None, f"图构建失败 ({step_path.name}): {e}"

        dim_err = self._validate_graph_feature_dims(sample_dict["graph"])
        if dim_err:
            return None, None, f"输入特征维度与模型配置不匹配 ({step_path.name})：{dim_err}"

        if self.normalize and self.stat is not None:
            sample_dict = standardization(sample_dict, self.stat)
        if self.do_center_and_scale:
            sample_dict = center_and_scale(sample_dict)

        input_graph = sample_dict["graph"].to(self.device)
        del sample_dict

        with torch.no_grad():
            try:
                segmentation_logits = self.recognizer(input_graph)
            except Exception as e:
                return None, None, f"推理失败 ({step_path.name}): {e}"

            probabilities = torch.softmax(segmentation_logits, dim=1)
            max_probabilities, predicted_classes = torch.max(probabilities, dim=1)
            avg_confidence = float(max_probabilities.mean().item()) if max_probabilities.numel() > 0 else 0.0
            labels = predicted_classes.cpu().tolist()

        del input_graph, segmentation_logits, probabilities, max_probabilities, predicted_classes
        return labels, avg_confidence, None

    def batchFeatureRecog(self):
        if not self.step_files:
            self.msgBox.warning(self, "warning", "请先加载STEP文件夹并确保列表非空")
            return

        start_time = time.time()
        success_count = 0
        fail_count = 0
        fail_examples: List[str] = []

        for step_path in self.step_files:
            labels, avg_confidence, err = self._predict_step(step_path)
            if err or labels is None or avg_confidence is None:
                fail_count += 1
                if err and len(fail_examples) < 3:
                    fail_examples.append(err)
                QApplication.processEvents()
                continue

            step_key = str(step_path)
            self.step_avg_confidence_by_path[step_key] = float(avg_confidence)
            self.step_pred_label_cache[step_key] = list(labels)
            success_count += 1
            QApplication.processEvents()

        self._populate_step_list()  # 刷新列表（含排序与置信度显示）

        elapsed = time.time() - start_time
        msg = f"批量预测完成\n成功: {success_count}\n失败: {fail_count}\n耗时: {elapsed:.2f}s"
        if fail_examples:
            msg += "\n\n失败示例:\n" + "\n".join(fail_examples)
        self.msgBox.information(self, "提示", msg)

    def _gray_color(self) -> Tuple[float, float, float]:
        return (0.65, 0.65, 0.65)

    def _selection_highlight_color(self) -> Tuple[float, float, float]:
        return (1.0, 0.85, 0.2)

    def _set_shape_all_gray(self, ais_shape):
        if not ais_shape:
            return
        ais_shape.ClearCustomAspects()
        gray = self._gray_color()
        for face_obj in self.faces_list:
            ais_shape.SetCustomColor(face_obj, rgb_color(*gray))
            ais_shape.SetCustomTransparency(face_obj, 0.0)

    def _apply_selection_coloring_for_view(self, target: str):
        if target == "gt":
            ais_shape = self.gt_ais_shape
            display = self.gt_display
            labels = self.current_gt_labels if self.gt_enabled else None
        else:
            ais_shape = self.pred_ais_shape
            display = self.pred_display
            labels = self.current_pred_labels

        if not ais_shape:
            return

        self._set_shape_all_gray(ais_shape)
        has_valid_labels = bool(labels) and len(labels) == len(self.faces_list)
        selected_face_indices = self._get_selection(target)
        for face_idx in selected_face_indices:
            if not (0 <= face_idx < len(self.faces_list)):
                continue
            face_obj = self.faces_list[face_idx]
            if has_valid_labels:
                class_idx = int(labels[face_idx])
                color = self._class_color(class_idx)
            else:
                color = self._selection_highlight_color()
            ais_shape.SetCustomColor(face_obj, rgb_color(*color))
            ais_shape.SetCustomTransparency(face_obj, 0.0)

        if display is not None:
            display.Context.Display(ais_shape, True)

    def _class_name(self, class_idx: int) -> str:
        default_label_names = ["other", "hole", "slot groove"]  # 默认类别名顺序
        label_names = self.label_names  # or default_label_names  # 优先使用配置中的类别名
        if 0 <= class_idx < len(label_names):  # 索引有效
            return f"class_{class_idx} {label_names[class_idx]}"  # class后追加类别名
        return f"class_{class_idx}"  # 返回默认名

    def _class_color(self, class_idx: int) -> Tuple[float, float, float]:
        if class_idx < 0:  # 非法索引
            return (1.0, 1.0, 1.0)  # 默认白色
        hue = (class_idx * 0.61803398875) % 1.0  # 伪随机色相
        return self._hsv_to_rgb(hue, 0.65, 0.95)  # 转RGB

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        i = int(h * 6.0)  # 色相分段
        f = h * 6.0 - i  # 小数部分
        p = v * (1.0 - s)  # 低亮度
        q = v * (1.0 - f * s)  # 过渡值
        t = v * (1.0 - (1.0 - f) * s)  # 过渡值
        i = i % 6  # 分段取模
        if i == 0:  # 区间0
            return (v, t, p)  # 返回RGB
        if i == 1:  # 区间1
            return (q, v, p)  # 返回RGB
        if i == 2:  # 区间2
            return (p, v, t)  # 返回RGB
        if i == 3:  # 区间3
            return (p, q, v)  # 返回RGB
        if i == 4:  # 区间4
            return (t, p, v)  # 返回RGB
        return (v, p, q)  # 区间5

    def predFeatureListDoubleClicked(self):
        """点击Prediction类别项：默认整类选中；按住Ctrl从当前选择中减去该类"""
        if not (self.pred_ais_shape and self.file_name and len(self.pred_features_list) != 0):  # 前置条件检查
            return  # 直接返回

        selected_row_idx = self.predFeatureListWidget.currentRow()  # 获取选中的行索引
        if selected_row_idx < 0 or selected_row_idx >= len(self.pred_features_list):  # 越界保护
            return  # 直接返回

        selected_feature = self.pred_features_list[selected_row_idx]  # 获取选中的特征对象
        class_faces = sorted({idx for idx in selected_feature.faces if 0 <= idx < len(self.faces_list)})
        ctrl_pressed = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl_pressed:
            selected_set = set(self.selected_face_indices_pred)
            for face_idx in class_faces:
                if face_idx in selected_set:
                    selected_set.remove(face_idx)
                else:
                    selected_set.add(face_idx)
            self._set_selection("pred", sorted(selected_set), record_history=True)
        else:
            self._set_selection("pred", class_faces, record_history=True)

    def gtFeatureListDoubleClicked(self):
        """点击GT类别项：默认整类选中；按住Ctrl从当前选择中减去该类"""
        if not (self.gt_ais_shape and self.file_name and len(self.gt_features_list) != 0):  # 前置条件检查
            return  # 直接返回

        selected_row_idx = self.gtFeatureListWidget.currentRow()  # 获取选中的行索引
        if selected_row_idx < 0 or selected_row_idx >= len(self.gt_features_list):  # 越界保护
            return  # 直接返回

        selected_feature = self.gt_features_list[selected_row_idx]  # 获取选中的特征对象
        class_faces = sorted({idx for idx in selected_feature.faces if 0 <= idx < len(self.faces_list)})
        ctrl_pressed = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl_pressed:
            selected_set = set(self.selected_face_indices_gt)
            for face_idx in class_faces:
                if face_idx in selected_set:
                    selected_set.remove(face_idx)
                else:
                    selected_set.add(face_idx)
            self._set_selection("gt", sorted(selected_set), record_history=True)
        else:
            self._set_selection("gt", class_faces, record_history=True)

    def openStepFolder(self):
        default_dir = self._dialog_dir("step_folder", self.step_dir or Path("./"))  # 默认路径
        folder_path = QFileDialog.getExistingDirectory(self, "选择STEP文件夹", default_dir)  # 打开文件夹对话框
        if not folder_path:  # 用户取消选择
            return  # 直接返回
        self.step_dir = Path(folder_path)  # 保存STEP文件夹
        self._set_dialog_dir("step_folder", self.step_dir)  # 记录目录
        self._set_dialog_dir("step_file", self.step_dir)  # 同步STEP文件选择目录
        self.step_files = self._scan_step_files(self.step_dir)  # 扫描STEP文件
        self._populate_step_list()  # 更新列表
        self._save_state()  # 保存状态
        self._update_window_title()  # 更新标题

    def openLabelFolder(self):
        default_dir = self._dialog_dir("label_folder", self.label_dir or Path("./"))  # 默认路径
        folder_path = QFileDialog.getExistingDirectory(self, "选择标签文件夹", default_dir)  # 打开文件夹对话框
        if not folder_path:  # 用户取消选择
            return  # 直接返回
        self.label_dir = Path(folder_path)  # 保存标签文件夹
        self._set_dialog_dir("label_folder", self.label_dir)  # 记录目录
        self._set_dialog_dir("save_pred", self.label_dir)  # 同步预测保存目录
        self._save_state()  # 保存状态
        self._update_window_title()  # 更新标题

    def toggleGTVisualization(self):
        if not (self.gt_ais_shape and self.file_name):  # 未加载STEP
            self.msgBox.warning(self, "警告", "请先加载STEP文件")  # 提示
            return  # 直接返回
        if not self.label_dir:  # 未指定标签目录
            self.msgBox.warning(self, "警告", "请先加载标签文件夹")  # 提示
            return  # 直接返回

        self.gt_enabled = not self.gt_enabled  # 切换开关
        self._update_gt_button_text()  # 更新按钮文字

        if self.gt_enabled:  # 开启GT显示
            self._apply_gt_labels()  # 应用GT颜色
        else:  # 关闭GT显示
            self._set_shape_all_gray(self.gt_ais_shape)  # 恢复全灰
            self.gt_display.Context.Display(self.gt_ais_shape, True)  # 刷新GT显示
            self.gtFeatureListWidget.clear()  # 清空GT列表
            self.gt_features_list.clear()  # 清空GT特征
            self.gt_class_index_by_row.clear()  # 清空GT映射

    def _update_gt_button_text(self):
        text = "关闭GT标签" if self.gt_enabled else "显示GT标签"  # 生成按钮文字
        if hasattr(self, "btn_toggle_gt"):  # 若按钮存在
            self.btn_toggle_gt.setText(text)  # 更新文字

    def _apply_gt_labels(self):
        label_path = self._resolve_label_path()  # 获取标签文件路径
        if not label_path:  # 未找到标签文件
            self.msgBox.warning(self, "警告", "未找到对应的标签文件")  # 提示
            return  # 直接返回

        labels = self._load_gt_labels(label_path)  # 读取GT标签
        if labels is None:  # 解析失败
            self.msgBox.warning(self, "警告", "GT标签格式无法解析")  # 提示
            return  # 直接返回
        if len(labels) != len(self.faces_list):  # 长度不一致
            self.msgBox.warning(self, "警告", "GT标签数量与面数量不一致")  # 提示
            return  # 直接返回
        self.current_gt_labels = labels  # 缓存GT标签
        if self.gt_session_base_labels is None:
            self.gt_session_base_labels = list(labels)
            self.gt_session_dirty = False
        self.gt_version_preview_active = False
        self._refresh_gt_version_list()
        self._apply_gt_labels_from_labels(labels)  # 应用颜色

    def _resolve_label_path(self) -> Optional[Path]:
        if not (self.label_dir and self.file_name):  # 条件不满足
            return None  # 直接返回
        step_stem = Path(self.file_name).stem  # STEP文件名
        json_path = self.label_dir / f"{step_stem}.json"  # 默认json
        if json_path.exists():  # 若存在
            return json_path  # 返回json
        pkl_path = self.label_dir / f"{step_stem}.pkl"  # 兜底pkl
        if pkl_path.exists():  # 若存在
            return pkl_path  # 返回pkl

        candidates = list(self.label_dir.glob(f"{step_stem}*.json"))  # 前缀匹配json
        candidates += list(self.label_dir.glob(f"{step_stem}*.pkl"))  # 前缀匹配pkl
        if not candidates:  # 无候选
            return None  # 未找到

        candidates = sorted(candidates, key=lambda p: (len(p.stem), p.name.lower()))  # 选最短且稳定
        return candidates[0]  # 返回最优匹配

    def _load_gt_labels(self, label_path: Path) -> Optional[List[int]]:
        try:  # 读取文件
            label_data = load_json_or_pkl(label_path)  # 读取json或pkl
        except Exception:  # 读取失败
            return None  # 返回空
        prev_label_path = self.gt_label_path
        self.gt_label_path = label_path  # 记录路径
        payload = normalize_label_payload(label_data)
        labels = extract_labels_from_payload(payload)
        if labels is None:
            return None
        self.gt_label_data = payload  # 记录标准化数据
        self.gt_label_format = "versioned"  # 统一写回版本化格式

        if self.gt_session_base_labels is None or prev_label_path != label_path:
            self.gt_session_base_labels = list(labels)
            self.gt_session_dirty = False
        self.gt_version_preview_active = False

        self._refresh_gt_version_list()

        return labels

    def _extract_seg_labels(self, label_data, num_faces: int) -> Optional[List[int]]:
        if isinstance(label_data, list):  # list情况
            if label_data and all(isinstance(x, int) for x in label_data):  # 直接标签列表
                return [int(x) for x in label_data]  # 转为int列表
            if len(label_data) == 2 and isinstance(label_data[1], dict):  # 简单二元结构
                return self._labels_from_label_dict(label_data[1], num_faces)  # 解析label dict
            if label_data and isinstance(label_data[0], list) and len(label_data[0]) == 2:  # MFInstSeg结构
                if isinstance(label_data[0][1], dict):  # label dict
                    return self._labels_from_label_dict(label_data[0][1], num_faces)  # 解析label dict
            return None  # 无法解析

        if isinstance(label_data, dict):  # dict情况
            return self._labels_from_label_dict(label_data, num_faces)  # 解析label dict

        return None  # 不支持格式

    def _labels_from_label_dict(self, label_dict: dict, num_faces: int) -> Optional[List[int]]:
        if "seg" in label_dict:  # 优先seg字段
            seg = label_dict["seg"]  # 取seg
            return self._labels_from_seg(seg, num_faces)  # 解析seg
        if "labels" in label_dict and isinstance(label_dict["labels"], list):  # 兜底labels
            return [int(x) for x in label_dict["labels"]]  # 转为int列表
        return None  # 无法解析

    def _labels_from_seg(self, seg, num_faces: int) -> Optional[List[int]]:
        if isinstance(seg, list):  # list标签
            return [int(x) for x in seg]  # 转为int列表
        if isinstance(seg, dict):  # dict标签
            labels = [-1] * num_faces  # 初始化
            for i in range(num_faces):  # 按面索引填充
                key = str(i)  # 字符串key
                if key in seg:  # 存在则赋值
                    labels[i] = int(seg[key])  # 设置标签
            return labels  # 返回
        return None  # 无法解析

    def _detect_label_format(self, label_data) -> Optional[str]:
        if isinstance(label_data, list):
            if label_data and all(isinstance(x, int) for x in label_data):
                return "list"
            if len(label_data) == 2 and isinstance(label_data[1], dict):
                return "pair"
            if label_data and isinstance(label_data[0], list) and len(label_data[0]) == 2:
                if isinstance(label_data[0][1], dict):
                    return "pair_list"
            return None
        if isinstance(label_data, dict):
            return "dict"
        return None

    def _refresh_gt_version_list(self):
        if not hasattr(self, "gtVersionListWidget"):
            return
        self.gtVersionListWidget.clear()
        self.gt_version_items = []
        if not isinstance(self.gt_label_data, dict):
            return
        current_version_id = int(self.gt_label_data.get("version_id", 1))
        self.gt_version_items.append({"kind": "current", "version_id": None})
        self.gtVersionListWidget.addItem(f"current v{current_version_id}")

        self.gt_version_items.append({"kind": "base", "version_id": 0})
        self.gtVersionListWidget.addItem("v0 | base")

        versions = self.gt_label_data.get("versions", [])
        if isinstance(versions, list):
            versions_sorted = sorted(versions, key=lambda v: int(v.get("version_id", 0)))
            for record in versions_sorted:
                version_id = int(record.get("version_id", 0))
                timestamp = str(record.get("timestamp", ""))
                author = str(record.get("author", ""))
                summary = str(record.get("summary", ""))
                label = f"v{version_id} | {timestamp} | {author}"
                if summary:
                    label = f"{label} | {summary}"
                self.gt_version_items.append({"kind": "version", "version_id": version_id})
                self.gtVersionListWidget.addItem(label)

    def _on_gt_version_selected(self):
        if self.current_gt_labels is None or not isinstance(self.gt_label_data, dict):
            return
        item = self._selected_gt_version_item()
        if not item:
            return
        labels = self._get_labels_for_version_item(item)
        if labels is None or len(labels) != len(self.faces_list):
            return
        self.current_gt_labels = list(labels)
        self.gt_version_preview_active = True
        if self.gt_enabled:
            self._apply_gt_labels_from_labels(self.current_gt_labels)

    def _selected_gt_version_item(self) -> Optional[Dict[str, Optional[int]]]:
        if not hasattr(self, "gtVersionListWidget"):
            return None
        row = self.gtVersionListWidget.currentRow()
        if row < 0 or row >= len(self.gt_version_items):
            return None
        return self.gt_version_items[row]

    def _get_labels_for_version_item(self, item: Dict[str, Optional[int]]) -> Optional[List[int]]:
        if not isinstance(self.gt_label_data, dict):
            return None
        if item.get("kind") == "current":
            labels = self.gt_label_data.get("labels")
            return list(labels) if isinstance(labels, list) else None
        if item.get("kind") == "base":
            labels = self.gt_label_data.get("labels_base")
            return list(labels) if isinstance(labels, list) else rollback_payload(self.gt_label_data, 0)
        version_id = item.get("version_id")
        if version_id is None:
            return None
        return rollback_payload(self.gt_label_data, int(version_id))

    def _load_selected_gt_version(self, rollback: bool):
        if self.current_gt_labels is None or not isinstance(self.gt_label_data, dict):
            self.msgBox.warning(self, "警告", "请先加载GT标签")
            return
        item = self._selected_gt_version_item()
        if not item:
            return
        labels = self._get_labels_for_version_item(item)
        if labels is None:
            self.msgBox.warning(self, "警告", "无法解析所选版本")
            return
        if len(labels) != len(self.faces_list):
            self.msgBox.warning(self, "警告", "版本标签数量与面数量不一致")
            return
        if rollback:
            ops = self._build_ops_from_label_arrays(
                old_labels=self.current_gt_labels,
                new_labels=labels,
            )
            if not ops:
                return
            self.current_gt_labels = list(labels)
            self._save_gt_labels(self.current_gt_labels, append_version=False)
            self._update_gt_session_dirty()
        else:
            self.current_gt_labels = list(labels)
            self.gt_version_preview_active = True
        if self.gt_enabled:
            self._apply_gt_labels_from_labels(self.current_gt_labels)

    def _populate_pred_feature_list(self, class_to_faces_map: Dict[int, List[int]]):
        self.predFeatureListWidget.clear()  # 清空Prediction列表控件
        self.pred_features_list.clear()  # 清空Prediction特征聚合列表
        self.pred_class_index_by_row.clear()  # 清空Prediction行索引映射

        for class_idx, face_indices in class_to_faces_map.items():  # 遍历类别映射
            class_name = self._class_name(class_idx)  # 获取类别名称
            self.pred_features_list.append(FeatureClass(name=class_name, faces=face_indices))  # 保存聚合结果
            self.pred_class_index_by_row.append(class_idx)  # 记录行到类别的映射
            self.predFeatureListWidget.addItem(f"{class_name} ({len(face_indices)})")  # 更新列表显示
        self._sync_feature_list_selection_from_state("pred")  # 根据当前面选择同步类别选择

    def _populate_gt_feature_list(self, class_to_faces_map: Dict[int, List[int]]):
        self.gtFeatureListWidget.clear()  # 清空GT列表控件
        self.gt_features_list.clear()  # 清空GT特征聚合列表
        self.gt_class_index_by_row.clear()  # 清空GT行索引映射

        for class_idx, face_indices in class_to_faces_map.items():  # 遍历类别映射
            class_name = self._class_name(class_idx)  # 获取类别名称
            self.gt_features_list.append(FeatureClass(name=class_name, faces=face_indices))  # 保存聚合结果
            self.gt_class_index_by_row.append(class_idx)  # 记录行到类别的映射
            self.gtFeatureListWidget.addItem(f"{class_name} ({len(face_indices)})")  # 更新列表显示
        self._sync_feature_list_selection_from_state("gt")  # 根据当前面选择同步类别选择

    def _build_face_index(self):
        self.face_index_by_hash.clear()  # 清空映射
        for idx, face in enumerate(self.faces_list):  # 遍历面
            self.face_index_by_hash[face.HashCode(2147483647)] = idx  # 哈希索引

    def _populate_face_list(self):
        if not hasattr(self, "gtFaceListWidget") or not hasattr(self, "predFaceListWidget"):
            return
        self.gtFaceListWidget.clear()
        self.predFaceListWidget.clear()
        for idx in range(len(self.faces_list)):
            text = f"Face {idx}"
            self.gtFaceListWidget.addItem(text)
            self.predFaceListWidget.addItem(text)

    def _faceListItemClicked(self, target: str, list_widget: QListWidget):
        selected_rows = sorted({list_widget.row(item) for item in list_widget.selectedItems()})
        ctrl_pressed = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        current_row = list_widget.currentRow()
        if ctrl_pressed and 0 <= current_row < len(self.faces_list):
            selected_set = set(self._get_selection(target))
            if current_row in selected_set:
                selected_set.remove(current_row)
            else:
                selected_set.add(current_row)
            self._set_selection(target, sorted(selected_set), record_history=True)
        else:
            self._set_selection(
                target,
                [idx for idx in selected_rows if 0 <= idx < len(self.faces_list)],
                record_history=True,
            )

    def faceListItemClickedGT(self):
        self._faceListItemClicked("gt", self.gtFaceListWidget)

    def faceListItemClickedPred(self):
        self._faceListItemClicked("pred", self.predFaceListWidget)

    def _faceListContextMenu(self, target: str, list_widget: QListWidget, pos):
        item = list_widget.itemAt(pos)  # 获取右键项
        ctrl_pressed = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if item is not None:
            row = list_widget.row(item)  # 获取行
            if 0 <= row < len(self.faces_list):
                if not ctrl_pressed:
                    self._set_selection(target, [row], record_history=True)
                else:
                    selected_set = set(self._get_selection(target))
                    if row in selected_set:
                        selected_set.remove(row)
                    else:
                        selected_set.add(row)
                    self._set_selection(target, sorted(selected_set), record_history=True)
        global_pos = list_widget.mapToGlobal(pos)  # 转全局坐标
        self._show_face_edit_menu(global_pos, target, show_locate=True)  # 右键菜单

    def faceListContextMenuGT(self, pos):
        self._faceListContextMenu("gt", self.gtFaceListWidget, pos)

    def faceListContextMenuPred(self, pos):
        self._faceListContextMenu("pred", self.predFaceListWidget, pos)

    def toggleFaceSelection(self):
        if not (self.gt_ais_shape and self.pred_ais_shape):  # 未加载STEP
            self.msgBox.warning(self, "警告", "请先加载STEP文件")  # 提示
            return  # 直接返回
        self.face_select_enabled = not self.face_select_enabled  # 切换选择模式
        if self.face_select_enabled:  # 开启
            self.gt_display.Context.Activate(self.gt_ais_shape, TopAbs_FACE, True)  # 启用GT面选择
            self.pred_display.Context.Activate(self.pred_ais_shape, TopAbs_FACE, True)  # 启用Prediction面选择
        else:  # 关闭
            self.gt_display.Context.Deactivate(self.gt_ais_shape)  # 关闭GT选择
            self.pred_display.Context.Deactivate(self.pred_ais_shape)  # 关闭Prediction选择
            self._set_selection("gt", [], record_history=True)  # 清空GT选中面
            self._set_selection("pred", [], record_history=True)  # 清空Prediction选中面

    def _on_select(self, source: str, selected_shapes, *args, **kwargs):
        if not self.face_select_enabled:  # 未开启选择
            return  # 直接返回
        if self._viewport_dragging.get(source, False):
            return  # 拖拽旋转过程中忽略选择
        if self._suppress_next_select:
            self._suppress_next_select = False
            return  # 拖拽释放后抑制一次误触发选择
        ctrl_pressed = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if not selected_shapes:  # 点击空白处取消选择
            if self._get_selection(source) and not ctrl_pressed:
                self._set_selection(source, [], record_history=True)  # 清空当前视口选中面
            return  # 直接返回
        selected_shape = selected_shapes[0]  # 取第一个
        face_idx = self.face_index_by_hash.get(selected_shape.HashCode(2147483647))  # 查找索引
        if face_idx is None:  # 未匹配
            return  # 直接返回
        if ctrl_pressed:
            selected_set = set(self._get_selection(source))
            if face_idx in selected_set:  # Ctrl+点击已选中面则取消
                selected_set.remove(face_idx)
            else:  # Ctrl+点击未选中面则追加
                selected_set.add(face_idx)
            self._set_selection(source, sorted(selected_set), record_history=True)
        else:  # 普通点击单选
            self._set_selection(source, [face_idx], record_history=True)

    def _get_selection(self, target: str) -> List[int]:
        if target == "gt":
            return self.selected_face_indices_gt
        if target == "pred":
            return self.selected_face_indices_pred
        return []

    def _set_selection(self, target: str, new_selection: List[int], record_history: bool = True):
        if target not in ("gt", "pred"):
            return
        normalized = sorted({idx for idx in new_selection if 0 <= idx < len(self.faces_list)})
        old_selection = list(self._get_selection(target))
        if normalized == old_selection:
            return
        if target == "gt":
            self.selected_face_indices_gt = normalized
        else:
            self.selected_face_indices_pred = normalized
        self._sync_face_list_selection_from_state(target)
        self._sync_feature_list_selection_from_state(target)
        self._apply_selection_coloring_for_view(target)
        if record_history:
            self.undo_stack.append(
                {
                    "op": "selection",
                    "selection_target": target,
                    "old_selection": old_selection,
                    "new_selection": list(normalized),
                }
            )
            self.redo_stack.clear()

    def _sync_face_list_selection_from_state(self, target: str):
        list_widget = self.gtFaceListWidget if target == "gt" else self.predFaceListWidget
        selected_face_indices = self._get_selection(target)
        if list_widget is None:
            return
        selected_set = set(selected_face_indices)
        list_widget.blockSignals(True)
        try:
            for idx in range(list_widget.count()):
                item = list_widget.item(idx)
                item.setSelected(idx in selected_set)
            if selected_face_indices:
                list_widget.setCurrentRow(selected_face_indices[-1])
            else:
                list_widget.clearSelection()
        finally:
            list_widget.blockSignals(False)

    def _sync_feature_list_selection_from_state(self, target: str):
        if target == "gt":
            list_widget = self.gtFeatureListWidget
            labels = self.current_gt_labels if self.gt_enabled else None
            class_index_by_row = self.gt_class_index_by_row
        else:
            list_widget = self.predFeatureListWidget
            labels = self.current_pred_labels
            class_index_by_row = self.pred_class_index_by_row

        if list_widget is None:
            return

        selected_face_indices = self._get_selection(target)
        selected_classes = set()
        if labels and len(labels) == len(self.faces_list):
            for face_idx in selected_face_indices:
                if 0 <= face_idx < len(labels):
                    selected_classes.add(int(labels[face_idx]))

        list_widget.blockSignals(True)
        try:
            if not selected_classes:
                list_widget.clearSelection()
                return
            for row in range(list_widget.count()):
                item = list_widget.item(row)
                class_idx = class_index_by_row[row] if row < len(class_index_by_row) else None
                item.setSelected(class_idx in selected_classes)
        finally:
            list_widget.blockSignals(False)

    def eventFilter(self, obj, event):
        if obj in (self.gt_canvas, self.pred_canvas):
            view_key = "gt" if obj is self.gt_canvas else "pred"
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._viewport_press_pos[view_key] = event.pos()
                self._viewport_dragging[view_key] = False

            if event.type() == QEvent.MouseMove and (event.buttons() & Qt.LeftButton):
                press_pos = self._viewport_press_pos.get(view_key)
                if press_pos is not None:
                    if (event.pos() - press_pos).manhattanLength() >= 6:
                        self._viewport_dragging[view_key] = True

            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self._viewport_dragging.get(view_key, False):
                    self._suppress_next_select = True
                self._viewport_press_pos[view_key] = None
                self._viewport_dragging[view_key] = False

            if self.face_select_enabled and event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                self._show_face_edit_menu(event.globalPos(), view_key)  # 右键菜单
                return True

            if event.type() in (QEvent.MouseMove, QEvent.MouseButtonRelease, QEvent.Wheel):
                if obj is self.gt_canvas:
                    self._sync_view_from_to("gt", "pred")  # 同步GT到Prediction
                else:
                    self._sync_view_from_to("pred", "gt")  # 同步Prediction到GT
        return super().eventFilter(obj, event)

    def _sync_view_from_to(self, source: str, target: str):
        if self.view_sync_lock:  # 防止重入
            return
        src_display = self.gt_display if source == "gt" else self.pred_display  # 源显示对象
        dst_display = self.gt_display if target == "gt" else self.pred_display  # 目标显示对象
        if src_display is None or dst_display is None:
            return
        try:
            self.view_sync_lock = True  # 加锁
            src_camera = src_display.View.Camera()  # 获取源相机
            dst_display.View.SetCamera(src_camera)  # 同步相机
            dst_display.View.Redraw()  # 刷新目标视图
        except Exception:
            pass
        finally:
            self.view_sync_lock = False  # 解锁

    def _show_face_edit_menu(self, global_pos, target: Optional[str] = None, show_locate: bool = False):
        menu = QMenu(self)  # 右键菜单
        action_edit_gt = QAction("修改GT标签", self)  # GT编辑项
        action_edit_pred = QAction("修改预测标签", self)  # 预测编辑项
        action_edit_gt.triggered.connect(self._edit_selected_face_gt_label)  # 绑定GT编辑
        action_edit_pred.triggered.connect(self._edit_selected_face_pred_label)  # 绑定预测编辑
        if show_locate and target in ("gt", "pred"):
            action_locate_face = QAction("定位面", self)  # 定位项
            action_locate_face.triggered.connect(lambda _checked=False, t=target: self._locate_selected_face(t))
            menu.addAction(action_locate_face)
            menu.addSeparator()
        if target == "gt":
            menu.addAction(action_edit_gt)  # GT视口仅显示GT编辑
        elif target == "pred":
            menu.addAction(action_edit_pred)  # Prediction视口仅显示预测编辑
        else:
            menu.addAction(action_edit_gt)  # 添加GT项
            menu.addAction(action_edit_pred)  # 添加预测项
        menu.exec_(global_pos)  # 弹出菜单

    def _locate_selected_face(self, target: str):
        selected_face_indices = self._get_selection(target)
        if not selected_face_indices:
            self.msgBox.warning(self, "警告", "请先选中一个或多个面")
            return
        face_idx = min(selected_face_indices)  # 多选时按最小编号定位
        self._center_face_in_view(target, face_idx)

    def _center_face_in_view(self, target: str, face_idx: int):
        if face_idx < 0 or face_idx >= len(self.faces_list):
            return
        display = self.gt_display if target == "gt" else self.pred_display
        if display is None:
            return

        face_obj = self.faces_list[face_idx]
        try:
            bbox = Bnd_Box()
            brepbndlib_Add(face_obj, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            center = gp_Pnt(
                0.5 * (xmin + xmax),
                0.5 * (ymin + ymax),
                0.5 * (zmin + zmax),
            )

            camera = display.View.Camera()
            if hasattr(camera, "SetCenter"):
                camera.SetCenter(center)
                normal = self._face_normal(face_obj)
                if normal is not None:
                    view_distance = camera.Eye().Distance(camera.Center())
                    if view_distance <= 1e-6:
                        view_distance = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)

                    eye = gp_Pnt(
                        center.X() + normal.X() * view_distance,
                        center.Y() + normal.Y() * view_distance,
                        center.Z() + normal.Z() * view_distance,
                    )
                    camera.SetEye(eye)

                    fallback_up = gp_Dir(0.0, 0.0, 1.0)
                    if abs(normal.Dot(fallback_up)) > 0.95:
                        fallback_up = gp_Dir(1.0, 0.0, 0.0)
                    up_vec = gp_Vec(fallback_up)
                    proj_len = up_vec.Dot(gp_Vec(normal))
                    up_vec = up_vec - gp_Vec(normal).Multiplied(proj_len)
                    if up_vec.Magnitude() > 1e-9:
                        camera.SetUp(gp_Dir(up_vec))
                display.View.SetCamera(camera)
                display.View.Redraw()
            else:
                display.FitAll()
        except Exception:
            try:
                display.FitAll()
            except Exception:
                pass

    def _face_normal(self, face_obj) -> Optional[gp_Dir]:
        try:
            u_min, u_max, v_min, v_max = breptools_UVBounds(face_obj)
            u_mid = 0.5 * (u_min + u_max)
            v_mid = 0.5 * (v_min + v_max)

            surface_adaptor = BRepAdaptor_Surface(face_obj, True)
            surface_handle = surface_adaptor.Surface().Surface()
            sl_props = GeomLProp_SLProps(surface_handle, u_mid, v_mid, 1, 1e-6)
            if not sl_props.IsNormalDefined():
                return None

            normal = sl_props.Normal()
            if face_obj.Orientation() == TopAbs_REVERSED:
                normal.Reverse()
            return normal
        except Exception:
            return None

    def _edit_selected_face_gt_label(self):
        if not self.selected_face_indices_gt:  # 未选中面
            self.msgBox.warning(self, "警告", "请先选中一个或多个面")  # 提示
            return  # 直接返回
        if not self.label_dir:  # 未选择标签目录
            self.msgBox.warning(self, "警告", "请先加载标签文件夹")  # 提示
            return  # 直接返回

        label_path = self._resolve_label_path()  # 获取标签路径
        if not label_path:  # 未找到标签文件
            self.msgBox.warning(self, "警告", "未找到对应的标签文件")  # 提示
            return  # 直接返回

        if self.current_gt_labels is None or self.gt_label_path != label_path:
            labels = self._load_gt_labels(label_path)  # 读取GT标签
            if labels is None:
                self.msgBox.warning(self, "警告", "GT标签格式无法解析")  # 提示
                return  # 直接返回
            self.current_gt_labels = labels  # 缓存

        options, option_map = self._label_options()  # 选项
        first_face_idx = self.selected_face_indices_gt[0]
        current_label = self.current_gt_labels[first_face_idx]  # 当前标签（以第一个选中面为准）
        current_text = option_map.get(current_label, options[0])  # 默认项
        current_index = options.index(current_text) if current_text in options else 0

        selected_text, ok = QInputDialog.getItem(
            self,
            "批量修改GT标签",
            "选择类别",
            options,
            current_index,
            False,
        )
        if not ok:
            return  # 取消

        new_label = int(selected_text.split(":", 1)[0])  # 解析标签
        self._apply_multi_label_change("gt", self.selected_face_indices_gt, new_label, record_history=True)  # 批量更新并记录

    def _edit_selected_face_pred_label(self):
        if not self.selected_face_indices_pred:  # 未选中面
            self.msgBox.warning(self, "警告", "请先选中一个或多个面")  # 提示
            return  # 直接返回

        if not self.current_pred_labels:  # 无预测结果
            self.msgBox.warning(self, "警告", "请先运行分割，生成预测标签")  # 提示
            return  # 直接返回

        options, option_map = self._label_options_for_labels(self.current_pred_labels)  # 选项
        first_face_idx = self.selected_face_indices_pred[0]
        current_label = self.current_pred_labels[first_face_idx]  # 当前标签（以第一个选中面为准）
        current_text = option_map.get(current_label, options[0])  # 默认项
        current_index = options.index(current_text) if current_text in options else 0

        selected_text, ok = QInputDialog.getItem(
            self,
            "批量修改预测标签",
            "选择类别",
            options,
            current_index,
            False,
        )
        if not ok:
            return  # 取消

        new_label = int(selected_text.split(":", 1)[0])  # 解析标签
        self._apply_multi_label_change("pred", self.selected_face_indices_pred, new_label, record_history=True)  # 批量更新并记录

    def _apply_multi_label_change(self, target: str, face_indices: List[int], new_label: int, record_history: bool):
        unique_face_indices = sorted({idx for idx in face_indices if 0 <= idx < len(self.faces_list)})
        if not unique_face_indices:
            return

        old_labels_map: Dict[int, int] = {}
        for face_idx in unique_face_indices:
            if target == "gt":
                if self.current_gt_labels is None:
                    return
                old_label = int(self.current_gt_labels[face_idx])
            elif target == "pred":
                if self.current_pred_labels is None:
                    return
                old_label = int(self.current_pred_labels[face_idx])
            else:
                return
            if old_label != int(new_label):
                old_labels_map[int(face_idx)] = old_label

        if not old_labels_map:
            return

        if target == "gt":
            self._apply_gt_label_change_batch(
                indices=sorted(old_labels_map.keys()),
                new_labels=[int(new_label) for _ in sorted(old_labels_map.keys())],
            )
        else:
            for face_idx in old_labels_map.keys():
                self._apply_single_label_change(
                    target=target,
                    face_idx=face_idx,
                    new_label=int(new_label),
                    record_history=False,
                )

        if record_history:
            self.undo_stack.append(
                {
                    "target": target,
                    "face_indices": sorted(old_labels_map.keys()),
                    "old_labels": [old_labels_map[idx] for idx in sorted(old_labels_map.keys())],
                    "new_label": int(new_label),
                }
            )
            self.redo_stack.clear()

    def _apply_single_label_change(self, target: str, face_idx: int, new_label: int, record_history: bool):
        if face_idx < 0 or face_idx >= len(self.faces_list):
            return

        if target == "gt":
            if self.current_gt_labels is None:
                return
            old_label = int(self.current_gt_labels[face_idx])
            if old_label == int(new_label):
                return
            self._apply_gt_label_change_batch(
                indices=[int(face_idx)],
                new_labels=[int(new_label)],
            )
        elif target == "pred":
            if self.current_pred_labels is None:
                return
            old_label = int(self.current_pred_labels[face_idx])
            if old_label == int(new_label):
                return
            self.current_pred_labels[face_idx] = int(new_label)
            self._apply_pred_labels_from_labels(self.current_pred_labels)  # 刷新Prediction显示
        else:
            return

        if record_history:
            self.undo_stack.append(
                {
                    "target": target,
                    "face_idx": int(face_idx),
                    "old_label": int(old_label),
                    "new_label": int(new_label),
                }
            )
            self.redo_stack.clear()  # 新操作后清空重做栈

    def undoLabelEdit(self):
        if not self.undo_stack:
            return
        op = self.undo_stack.pop()
        if op.get("op") == "selection":
            target = op.get("selection_target", "pred")
            self._set_selection(target, op.get("old_selection", []), record_history=False)
            self.redo_stack.append(op)
            return
        if "face_indices" in op and "old_labels" in op:
            if op.get("target") == "gt":
                self._apply_gt_label_change_batch(
                    indices=op["face_indices"],
                    new_labels=op["old_labels"],
                )
            else:
                for face_idx, old_label in zip(op["face_indices"], op["old_labels"]):
                    self._apply_single_label_change(
                        target=op["target"],
                        face_idx=face_idx,
                        new_label=old_label,
                        record_history=False,
                    )
            self.redo_stack.append(op)
            return
        if op.get("target") == "gt":
            self._apply_gt_label_change_batch(
                indices=[op["face_idx"]],
                new_labels=[op["old_label"]],
            )
        else:
            self._apply_single_label_change(
                target=op["target"],
                face_idx=op["face_idx"],
                new_label=op["old_label"],
                record_history=False,
            )
        self.redo_stack.append(op)

    def redoLabelEdit(self):
        if not self.redo_stack:
            return
        op = self.redo_stack.pop()
        if op.get("op") == "selection":
            target = op.get("selection_target", "pred")
            self._set_selection(target, op.get("new_selection", []), record_history=False)
            self.undo_stack.append(op)
            return
        if "face_indices" in op and "new_label" in op:
            if op.get("target") == "gt":
                self._apply_gt_label_change_batch(
                    indices=op["face_indices"],
                    new_labels=[op["new_label"] for _ in op["face_indices"]],
                )
            else:
                self._apply_multi_label_change(
                    target=op["target"],
                    face_indices=op["face_indices"],
                    new_label=op["new_label"],
                    record_history=False,
                )
            self.undo_stack.append(op)
            return
        if op.get("target") == "gt":
            self._apply_gt_label_change_batch(
                indices=[op["face_idx"]],
                new_labels=[op["new_label"]],
            )
        else:
            self._apply_single_label_change(
                target=op["target"],
                face_idx=op["face_idx"],
                new_label=op["new_label"],
                record_history=False,
            )
        self.undo_stack.append(op)

    def _label_options(self) -> Tuple[List[str], Dict[int, str]]:
        return self._label_options_for_labels(self.current_gt_labels)  # 兼容旧逻辑

    def _label_options_for_labels(self, labels: Optional[List[int]]) -> Tuple[List[str], Dict[int, str]]:
        options: List[str] = []
        option_map: Dict[int, str] = {}
        if self.label_names:
            for idx, name in enumerate(self.label_names):
                text = f"{idx}: {name}"
                options.append(text)
                option_map[idx] = text
        else:
            max_label = 0
            if labels:
                max_label = max(labels)
            for idx in range(max_label + 1):
                text = f"{idx}: class_{idx}"
                options.append(text)
                option_map[idx] = text
        if not options:
            options = ["0: class_0"]
            option_map[0] = "0: class_0"
        return options, option_map

    def _build_label_data_with_format(self, labels: List[int], label_format: str, template_data):
        if label_format == "list":
            return [int(x) for x in labels]
        if label_format == "dict" and isinstance(template_data, dict):
            return self._update_label_dict(template_data, labels)
        if label_format == "pair" and isinstance(template_data, list) and len(template_data) == 2:
            return [template_data[0], self._update_label_dict(template_data[1], labels)]
        if label_format == "pair_list" and isinstance(template_data, list) and template_data:
            item = template_data[0]
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict):
                template_data[0] = [item[0], self._update_label_dict(item[1], labels)]
                return template_data
        return [int(x) for x in labels]

    def savePredictionLabels(self):
        if not self.file_name:  # 未加载STEP
            self.msgBox.warning(self, "警告", "请先加载STEP文件")
            return
        if not self.current_pred_labels:  # 无预测
            self.msgBox.warning(self, "警告", "请先运行分割，生成预测标签")
            return

        template_path = self._resolve_label_path()  # 尝试使用当前GT标签作为模板
        template_data = None
        template_format = None
        suffix = ".json"
        if template_path and template_path.exists():
            try:
                template_data = load_json_or_pkl(template_path)
                template_format = self._detect_label_format(template_data)
                suffix = template_path.suffix.lower()
            except Exception:
                template_data = None
                template_format = None

        if template_data is None or template_format is None:
            template_data = [0 for _ in self.current_pred_labels]
            template_format = "list"
            suffix = ".json"

        default_dir = Path(self._dialog_dir("save_pred", self.label_dir or Path(self.file_name).parent))
        default_name = f"{Path(self.file_name).stem}_pred{suffix}"
        filter_text = "PKL Files (*.pkl)" if suffix == ".pkl" else "JSON Files (*.json)"
        save_path_str = QFileDialog.getSaveFileName(
            self,
            "保存预测标签",
            str(default_dir / default_name),
            filter_text,
        )[0]
        if not save_path_str:
            return

        save_path = Path(save_path_str)
        self._set_dialog_dir("save_pred", save_path.parent)  # 记录目录
        data_to_save = self._build_label_data_with_format(
            labels=self.current_pred_labels,
            label_format=template_format,
            template_data=template_data,
        )

        try:
            if save_path.suffix.lower() == ".pkl":
                with open(save_path, "wb") as f:
                    pickle.dump(data_to_save, f)
            else:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, ensure_ascii=True, indent=2)
        except Exception as e:
            self.msgBox.warning(self, "警告", f"保存预测标签失败: {e}")
            return

        self.pred_label_path = save_path  # 记录保存路径
        self.pred_label_format = template_format  # 记录保存格式
        self.pred_label_data = data_to_save  # 记录保存数据
        self.msgBox.information(self, "提示", f"预测标签已保存: {save_path}")

    def _apply_gt_labels_from_labels(self, labels: List[int]):
        if not self.gt_ais_shape:
            return
        class_to_faces_map: Dict[int, List[int]] = {}  # 类别到面索引映射
        for face_idx, class_idx in enumerate(labels):  # 遍历标签
            class_to_faces_map.setdefault(class_idx, []).append(face_idx)  # 归类
        self._populate_gt_feature_list(class_to_faces_map)  # 更新GT特征列表
        self._apply_selection_coloring_for_view("gt")  # 仅给选中面按GT分类着色

    def _apply_pred_labels_from_labels(self, labels: List[int]):
        if not self.pred_ais_shape:
            return
        class_to_faces_map: Dict[int, List[int]] = {}  # 类别到面索引映射
        for face_idx, class_idx in enumerate(labels):  # 遍历标签
            class_to_faces_map.setdefault(class_idx, []).append(face_idx)  # 归类
        self._populate_pred_feature_list(class_to_faces_map)  # 更新预测特征列表
        self._apply_selection_coloring_for_view("pred")  # 仅给选中面按预测分类着色

    def _refresh_selection_colors(self):
        self._apply_selection_coloring_for_view("gt")
        self._apply_selection_coloring_for_view("pred")

    def _apply_selection_overlay_to_shape(self, ais_shape):
        return

    def _update_label_dict(self, label_dict: dict, labels: List[int]) -> dict:
        if "seg" in label_dict:
            if isinstance(label_dict["seg"], list):
                label_dict["seg"] = [int(x) for x in labels]
            elif isinstance(label_dict["seg"], dict):
                label_dict["seg"] = {str(i): int(labels[i]) for i in range(len(labels))}
            else:
                label_dict["seg"] = [int(x) for x in labels]
            return label_dict
        label_dict["labels"] = [int(x) for x in labels]
        return label_dict

    def _save_gt_labels(
        self,
        labels: List[int],
        ops: Optional[List[Dict[str, int]]] = None,
        append_version: bool = True,
    ):
        if not self.gt_label_path or not self.gt_label_format:
            return
        if self.gt_label_format == "versioned" and isinstance(self.gt_label_data, dict):
            payload = self.gt_label_data
            payload["labels"] = [int(x) for x in labels]
            domains = payload.get("domains")
            if isinstance(domains, dict):
                geometry = domains.get("geometry")
                if isinstance(geometry, dict):
                    geometry["face"] = [int(x) for x in labels]
            if ops and append_version:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                append_label_version(
                    payload,
                    author=self.label_author,
                    ops=ops,
                    timestamp=timestamp,
                )
            data = payload
        else:
            data = self._build_label_data_with_format(labels, self.gt_label_format, self.gt_label_data)

        try:
            if self.gt_label_path.suffix.lower() == ".pkl":
                with open(self.gt_label_path, "wb") as f:
                    pickle.dump(data, f)
            else:
                with open(self.gt_label_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=True, indent=2)
        except Exception:
            return

        self._refresh_gt_version_list()

    def _apply_gt_label_change_batch(self, indices: List[int], new_labels: List[int]):
        if self.current_gt_labels is None:
            return
        if len(indices) != len(new_labels):
            return
        if self.gt_version_preview_active:
            self.gt_session_base_labels = list(self.current_gt_labels)
            self.gt_version_preview_active = False
        updated_indices: List[int] = []
        old_labels: List[int] = []
        applied_new_labels: List[int] = []
        for face_idx, new_label in zip(indices, new_labels):
            if not (0 <= face_idx < len(self.faces_list)):
                continue
            old_label = int(self.current_gt_labels[face_idx])
            if old_label == int(new_label):
                continue
            self.current_gt_labels[face_idx] = int(new_label)
            updated_indices.append(int(face_idx))
            old_labels.append(int(old_label))
            applied_new_labels.append(int(new_label))
        if not updated_indices:
            return
        self._save_gt_labels(self.current_gt_labels, append_version=False)
        self._update_gt_session_dirty()
        if self.gt_enabled:
            self._apply_gt_labels_from_labels(self.current_gt_labels)

    def _build_ops_from_label_changes(self, old_labels_map: Dict[int, int], new_label: int) -> List[Dict[str, int]]:
        indices = sorted(old_labels_map.keys())
        old_labels = [int(old_labels_map[idx]) for idx in indices]
        new_labels = [int(new_label) for _ in indices]
        return self._build_ops_from_pairs(indices, old_labels, new_labels)

    def _build_ops_from_pairs(self, indices: List[int], old_labels: List[int], new_labels: List[int]) -> List[Dict[str, int]]:
        if not indices:
            return []
        return [
            {
                "domain": "geometry.face",
                "indices": indices,
                "old_labels": old_labels,
                "new_labels": new_labels,
            }
        ]

    def _build_ops_from_label_arrays(self, old_labels: List[int], new_labels: List[int]) -> List[Dict[str, int]]:
        if len(old_labels) != len(new_labels):
            return []
        indices: List[int] = []
        prev_labels: List[int] = []
        next_labels: List[int] = []
        for idx, (old_label, new_label) in enumerate(zip(old_labels, new_labels)):
            if int(old_label) == int(new_label):
                continue
            indices.append(int(idx))
            prev_labels.append(int(old_label))
            next_labels.append(int(new_label))
        return self._build_ops_from_pairs(indices, prev_labels, next_labels)

    def _update_gt_session_dirty(self):
        if self.current_gt_labels is None or self.gt_session_base_labels is None:
            self.gt_session_dirty = False
            return
        if len(self.current_gt_labels) != len(self.gt_session_base_labels):
            self.gt_session_dirty = True
            return
        self.gt_session_dirty = any(
            int(a) != int(b)
            for a, b in zip(self.current_gt_labels, self.gt_session_base_labels)
        )

    def _commit_gt_session_changes(self):
        if not self.gt_session_dirty:
            return
        if (
            self.current_gt_labels is None
            or self.gt_session_base_labels is None
            or not isinstance(self.gt_label_data, dict)
        ):
            self.gt_session_dirty = False
            return
        ops = self._build_ops_from_label_arrays(
            old_labels=self.gt_session_base_labels,
            new_labels=self.current_gt_labels,
        )
        if not ops:
            self.gt_session_dirty = False
            return
        self._save_gt_labels(self.current_gt_labels, ops=ops, append_version=True)
        self.gt_session_base_labels = list(self.current_gt_labels)
        self.gt_session_dirty = False

    def stepListItemClicked(self):
        row = self.stepListWidget.currentRow()  # 获取选中行
        if row < 0 or row >= len(self.displayed_step_files):  # 越界保护
            return  # 直接返回
        self._load_step_file(self.displayed_step_files[row])  # 加载选中的STEP

    def _scan_step_files(self, step_dir: Path) -> List[Path]:
        if not step_dir.exists():  # 目录不存在
            return []  # 返回空列表
        step_files = list(step_dir.glob("*.stp")) + list(step_dir.glob("*.step"))  # 收集STEP文件
        step_files = sorted(step_files, key=lambda p: p.name.lower())  # 按文件名排序
        return step_files  # 返回文件列表

    def _populate_step_list(self):
        if not hasattr(self, "stepListWidget"):  # 列表未创建
            return  # 直接返回

        if self.step_sort_mode == "confidence_desc":
            sorted_files = sorted(
                self.step_files,
                key=lambda p: (
                    -self.step_avg_confidence_by_path.get(str(p), -1.0),
                    p.name.lower(),
                ),
            )
        elif self.step_sort_mode == "confidence_asc":
            sorted_files = sorted(
                self.step_files,
                key=lambda p: (
                    self.step_avg_confidence_by_path.get(str(p), float("inf")),
                    p.name.lower(),
                ),
            )
        else:
            sorted_files = sorted(self.step_files, key=lambda p: p.name.lower())

        self.displayed_step_files = sorted_files
        self.stepListWidget.clear()  # 清空列表
        for step_path in self.displayed_step_files:  # 遍历文件
            confidence = self.step_avg_confidence_by_path.get(str(step_path))
            if confidence is None:
                self.stepListWidget.addItem(step_path.name)  # 无置信度时仅显示名称
            else:
                self.stepListWidget.addItem(f"{step_path.name} | conf={confidence:.4f}")  # 显示均值置信度

        if self.last_step_file:  # 若存在上次选择
            for idx, step_path in enumerate(self.displayed_step_files):  # 查找匹配项
                if step_path == self.last_step_file:  # 路径一致
                    self.stepListWidget.setCurrentRow(idx)  # 选中
                    break

    def _on_step_sort_changed(self):
        if not hasattr(self, "stepSortCombo"):
            return
        if self.stepSortCombo.currentIndex() == 1:
            self.step_sort_mode = "confidence_desc"
        elif self.stepSortCombo.currentIndex() == 2:
            self.step_sort_mode = "confidence_asc"
        else:
            self.step_sort_mode = "name"
        self._populate_step_list()

    def _restore_step_list(self):
        if self.step_dir and not self.step_files:  # 有目录但无列表
            self.step_files = self._scan_step_files(self.step_dir)  # 重新扫描
        self._populate_step_list()  # 填充列表

        if self.last_step_file and self.last_step_file.exists():  # 自动加载上次文件
            self._load_step_file(self.last_step_file)  # 加载文件

    def _state_path(self) -> Path:
        return self.project_root / "qt5_visualization_state.json"  # 状态文件路径

    def _dialog_dir(self, key: str, fallback: Path) -> str:
        saved = self.dialog_dirs.get(key)
        if isinstance(saved, Path) and saved.exists():
            return str(saved)
        if fallback and Path(fallback).exists():
            return str(fallback)
        return "./"

    def _set_dialog_dir(self, key: str, folder: Optional[Path]):
        if not folder:
            return
        folder_path = Path(folder)
        if folder_path.exists():
            self.dialog_dirs[key] = folder_path

    def _load_state(self):
        state_path = self._state_path()  # 状态文件路径
        if not state_path.exists():  # 状态文件不存在
            return  # 直接返回
        try:  # 读取状态
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:  # 读取失败
            return  # 直接返回

        step_dir = state.get("step_dir")  # STEP目录
        label_dir = state.get("label_dir")  # 标签目录
        last_step = state.get("last_step")  # 上次选择
        step_files = state.get("step_files")  # STEP文件列表
        config_path = state.get("config_path")  # 配置文件
        ckpt_path = state.get("ckpt_path")  # 权重文件
        dialog_dirs = state.get("dialog_dirs")  # 各对话框目录

        if step_dir:
            step_dir_path = Path(step_dir)
            if step_dir_path.exists():
                self.step_dir = step_dir_path

        if label_dir:
            label_dir_path = Path(label_dir)
            if label_dir_path.exists():
                self.label_dir = label_dir_path

        if step_files and isinstance(step_files, list):
            self.step_files = [Path(p) for p in step_files if Path(p).exists()]

        if last_step:
            last_step_path = Path(last_step)
            if last_step_path.exists():
                self.last_step_file = last_step_path

        if config_path:
            cfg_path = Path(config_path)
            if cfg_path.exists():
                self.config_path = cfg_path

        if ckpt_path:
            ckpt_path_obj = Path(ckpt_path)
            if ckpt_path_obj.exists():
                self.ckpt_path = ckpt_path_obj

        if isinstance(dialog_dirs, dict):
            for key in self.dialog_dirs.keys():
                saved_dir = dialog_dirs.get(key)
                if not saved_dir:
                    continue
                saved_path = Path(saved_dir)
                if saved_path.exists():
                    self.dialog_dirs[key] = saved_path

        if self.step_dir:
            self.dialog_dirs["step_file"] = self.dialog_dirs.get("step_file") or self.step_dir
            self.dialog_dirs["step_folder"] = self.dialog_dirs.get("step_folder") or self.step_dir
        if self.label_dir:
            self.dialog_dirs["label_folder"] = self.dialog_dirs.get("label_folder") or self.label_dir
            self.dialog_dirs["save_pred"] = self.dialog_dirs.get("save_pred") or self.label_dir
        if self.config_path:
            self.dialog_dirs["config"] = self.dialog_dirs.get("config") or self.config_path.parent
        if self.ckpt_path:
            self.dialog_dirs["ckpt"] = self.dialog_dirs.get("ckpt") or self.ckpt_path.parent

    def _save_state(self):
        state_path = self._state_path()  # 状态文件路径
        state = {
            "step_dir": str(self.step_dir) if self.step_dir else None,
            "label_dir": str(self.label_dir) if self.label_dir else None,
            "step_files": [str(p) for p in self.step_files],
            "last_step": str(self.last_step_file) if self.last_step_file else None,
            "config_path": str(self.config_path) if self.config_path else None,
            "ckpt_path": str(self.ckpt_path) if self.ckpt_path else None,
            "dialog_dirs": {
                k: (str(v) if isinstance(v, Path) else None) for k, v in self.dialog_dirs.items()
            },
        }
        try:  # 写入状态
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=True, indent=2)
        except Exception:
            return  # 静默失败


if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建应用
    ex = App()  # 创建窗口
    if os.getenv("APPVEYOR") is None:  # CI环境跳过
        sys.exit(app.exec_())  # 进入事件循环
