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
from OCC.Extend.TopologyUtils import TopologyExplorer  # OCC拓扑遍历
from PyQt5.QtCore import Qt, QEvent  # Qt常量
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
)
from OCC.Core.TopAbs import TopAbs_FACE  # 拓扑类型

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
        self.ais_shape = None  # OCC显示对象
        self.file_name = None  # STEP路径
        self.faces_list = []  # 面列表
        self.features_list: List[FeatureClass] = []  # 类别聚合
        self.class_index_by_row: List[int] = []  # 列表行到类别索引
        self.step_dir: Optional[Path] = None  # STEP文件夹
        self.label_dir: Optional[Path] = None  # 标签文件夹
        self.gt_enabled = False  # GT显示开关
        self.step_files: List[Path] = []  # STEP文件列表
        self.last_step_file: Optional[Path] = None  # 上次选择的STEP文件
        self.face_index_by_hash: Dict[int, int] = {}  # 面哈希到索引
        self.face_select_enabled = False  # 面选择开关
        self.selected_face_idx: Optional[int] = None  # 当前选中面索引
        self.current_gt_labels: Optional[List[int]] = None  # 当前GT标签
        self.gt_label_path: Optional[Path] = None  # 当前GT文件路径
        self.gt_label_format: Optional[str] = None  # 当前GT格式
        self.gt_label_data = None  # 当前GT原始数据

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

        self.topo_checker = TopologyChecker()  # 拓扑检查器

        self._load_state()  # 恢复上次状态

        self._load_config()  # 读取配置
        self._load_feature_schema()  # 读取特征schema
        self._load_statistics_if_needed()  # 读取统计量
        self._init_model()  # 初始化模型
        self.initUI()  # 初始化界面

    def initUI(self):
        self.setWindowTitle(self.title)  # 设置窗口标题
        self.setGeometry(self.left, self.top, self.width, self.height)  # 设置位置与大小
        self.setMinimumSize(900, 600)  # 设置最小尺寸
        self.setMaximumSize(1920, 1200)  # 设置最大尺寸
        self.createHorizontalLayout()  # 创建布局
        self._update_window_title()  # 更新标题与状态
        self.msgBox = QMessageBox()  # 消息框
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
        if not self.model_cfg:  # 模型配置为空
            return  # 跳过初始化
        if not self.ckpt_path:  # 未指定权重路径
            self.recognizer = None  # 清空模型实例
            return  # 等待用户选择权重后再加载
        ckpt_str = str(self.ckpt_path)  # 权重路径字符串
        self.recognizer = AAGNetSegmentor.load_from_checkpoint(  # 加载Lightning模型
            checkpoint_path=ckpt_str,  # 权重文件路径
            **self.model_cfg,  # 模型超参数配置
            map_location=self.device,  # 加载到指定设备
        )
        self.recognizer.to(self.device)  # 移动模型到推理设备
        self.recognizer.eval()  # 切换为评估模式（关闭Dropout等）

    def _load_feature_schema(self):
        """加载特征schema：从feature_lists目录读取AAG特征定义"""
        feature_schema_path = self.project_root.parent / "feature_lists" / "all.json"  # schema路径
        if feature_schema_path.exists():  # 若存在
            self.attribute_schema = load_json_or_pkl(feature_schema_path)  # 读取schema

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

        # 创建GT显示按钮
        self.btn_toggle_gt = QPushButton("显示GT标签", self)  # GT显示按钮
        self.btn_toggle_gt.clicked.connect(self.toggleGTVisualization)  # 绑定点击事件
        panel_layout.addWidget(self.btn_toggle_gt, 9, 0, 1, 1)  # 放置按钮

        # 创建STEP文件列表
        self.step_list_label = QLabel("STEP文件列表", self)  # STEP列表标题
        panel_layout.addWidget(self.step_list_label, 10, 0, 1, 1)  # 放置标题

        self.stepListWidget = QListWidget()  # STEP列表
        self.stepListWidget.itemClicked.connect(self.stepListItemClicked)  # 绑定点击事件
        panel_layout.addWidget(self.stepListWidget, 11, 0, 1, 1)  # 放置列表

        # 创建3D画布
        self.canvas = qtDisplay.qtViewer3d(self)  # OCC画布
        canvas_layout.addWidget(self.canvas, stretch=1)  # 添加画布到右侧布局
        self.canvas.resize(self.canvas_width, self.height_width)  # 设置画布尺寸
        self.canvas.InitDriver()  # 初始化渲染驱动
        self.display = self.canvas._display  # 获取显示对象

        # 创建右侧面列表面板
        self.face_panel = QWidget()  # 面列表容器
        self.face_panel.setFixedWidth(200)  # 固定宽度
        face_layout = QVBoxLayout()  # 面列表布局

        self.face_list_label = QLabel("面列表", self)  # 面列表标题
        face_layout.addWidget(self.face_list_label)  # 放置标题

        self.faceListWidget = QListWidget()  # 面列表
        self.faceListWidget.itemClicked.connect(self.faceListItemClicked)  # 绑定点击事件
        self.faceListWidget.setContextMenuPolicy(Qt.CustomContextMenu)  # 启用自定义右键
        self.faceListWidget.customContextMenuRequested.connect(self.faceListContextMenu)  # 绑定右键菜单
        face_layout.addWidget(self.faceListWidget)  # 放置列表

        self.face_panel.setLayout(face_layout)  # 设置布局
        canvas_layout.addWidget(self.face_panel)  # 添加右侧面板

        # 创建特征列表控件
        self.featureListWidget = QListWidget()  # 特征列表
        self.featureListWidget.itemDoubleClicked.connect(self.featureListDoubleClicked)  # 绑定双击事件
        panel_layout.addWidget(self.featureListWidget, 12, 0, 1, 1)  # 放置列表

        # 应用布局
        self.horizontalGroupBox.setLayout(canvas_layout)  # 设置右侧布局
        self.button_panel.setLayout(panel_layout)  # 设置左侧布局

        self.canvas.installEventFilter(self)  # 监听画布事件
        if hasattr(self.display, "register_select_callback"):
            self.display.register_select_callback(self._on_select)  # 注册选择回调

    def openConfig(self):
        config_path = QFileDialog.getOpenFileName(  # 打开配置文件对话框
            self,  # 父窗口
            "Open Config YAML",  # 标题
            str(self.project_root / "configs"),  # 默认目录
            "(*.yaml *.yml)",  # 文件过滤
        )[
            0
        ]  # 取第一个结果
        if not config_path:  # 未选择
            return  # 直接返回
        self.config_path = Path(config_path)  # 更新配置路径
        self._load_config()  # 重新读取配置
        self._load_statistics_if_needed()  # 重新读取统计量
        self._init_model()  # 重新初始化模型
        self._save_state()  # 保存状态
        self._update_window_title()  # 更新标题

    def openCheckpoint(self):
        ckpt_path = QFileDialog.getOpenFileName(  # 打开权重文件对话框
            self,  # 父窗口
            "Open Checkpoint",  # 标题
            str(self.project_root.parent / "output" / "checkpoints"),  # 默认目录
            "(*.ckpt)",  # 文件过滤
        )[
            0
        ]  # 取第一个结果
        if not ckpt_path:  # 未选择
            return  # 直接返回
        self.ckpt_path = Path(ckpt_path)  # 更新权重路径
        self._init_model()  # 重新初始化模型
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
        default_dir = str(self.step_dir) if self.step_dir else "./"  # 默认路径
        step_file_path = QFileDialog.getOpenFileName(self, "选择STEP文件", default_dir, "(*.st*p)")[0]  # 打开文件对话框
        if not step_file_path:  # 用户取消选择
            return  # 直接返回
        self._load_step_file(Path(step_file_path))  # 加载选择的STEP文件

    def _load_step_file(self, step_path: Path):
        self.file_name = str(step_path)  # 保存文件路径
        solid_shape = load_body_from_step(self.file_name)  # 读取STEP文件为实体
        if not self.topo_checker(solid_shape):  # 拓扑检查失败（非流形等）
            self.msgBox.warning(self, "警告", "加载失败，STEP文件不支持或存在拓扑错误")  # 提示用户
            self.file_name = None  # 清空文件路径
            return  # 结束处理

        if self.ais_shape:  # 若已存在旧的显示对象
            self.display.Context.Erase(self.ais_shape, True)  # 从上下文中清除旧对象

        self.featureListWidget.clear()  # 清空特征列表控件
        self.features_list.clear()  # 清空特征聚合数据

        self.ais_shape = AIS_ColoredShape(solid_shape)  # 创建可着色的显示对象
        self.display.Context.Display(self.ais_shape, True)  # 在上下文中显示对象
        self.display.FitAll()  # 自适应相机视角以显示全部内容

        topology_explorer = TopologyExplorer(solid_shape)  # 创建拓扑遍历器
        self.faces_list = list(topology_explorer.faces())  # 提取并记录所有面对象
        self._build_face_index()  # 构建面索引
        self._populate_face_list()  # 更新面列表
        self._update_window_title()  # 更新窗口标题显示当前文件

        if self.face_select_enabled:  # 若启用选择模式
            self.display.Context.Activate(self.ais_shape, TopAbs_FACE, True)  # 启用面选择

        self.last_step_file = step_path  # 记录最后选择
        self._save_state()  # 保存状态

        if self.gt_enabled:  # 若已开启GT显示
            self._apply_gt_labels()  # 自动刷新GT颜色

    def eraseShape(self):
        if self.ais_shape:  # 若存在对象
            self.display.Context.Erase(self.ais_shape, True)  # 清除显示
            self.ais_shape = None  # 清空对象
            self.file_name = None  # 清空路径
            self.featureListWidget.clear()  # 清空列表
            self.features_list.clear()  # 清空结果
            self.faces_list.clear()  # 清空面列表
            self.class_index_by_row.clear()  # 清空行映射
            self.face_index_by_hash.clear()  # 清空面哈希
            self.selected_face_idx = None  # 清空选中面
            if hasattr(self, "faceListWidget"):
                self.faceListWidget.clear()  # 清空面列表
            self.current_gt_labels = None  # 清空GT标签
            self.gt_label_path = None  # 清空GT路径
            self.gt_label_format = None  # 清空GT格式
            self.gt_label_data = None  # 清空GT数据
            self._update_window_title()  # 更新标题

    def featureRecog(self):
        """执行特征分割：AAG提取→图构建→推理→结果聚合→列表显示"""
        if not (self.ais_shape and self.file_name and self.attribute_schema):  # 前置检查
            return  # 直接返回

        if not hasattr(self, "recognizer"):  # 模型未初始化
            self.msgBox.warning(self, "warning", "模型未初始化，请先加载配置和权重")  # 提示
            return  # 直接返回

        start_time = time.time()  # 开始计时
        try:  # AAG提取
            aag_extractor = AAGExtractor(self.file_name, self.attribute_schema)  # 构建提取器
            aag_data = aag_extractor.process()  # 执行AAG提取
        except Exception as e:  # 异常处理
            self.msgBox.warning(self, "warning", f"AAG提取失败: {e}")  # 提示
            return  # 直接返回

        sample_dict = load_one_graph(self.file_name, aag_data)  # 构建DGL图
        if self.normalize and self.stat is not None:  # 标准化开关打开
            sample_dict = standardization(sample_dict, self.stat)  # 应用统计量标准化
        if self.do_center_and_scale:  # 居中缩放开关打开
            sample_dict = center_and_scale(sample_dict)  # 应用居中缩放变换

        input_graph = sample_dict["graph"].to(self.device)  # 移动图到推理设备
        del sample_dict  # 删除中间字典节省内存
        preprocess_time = time.time()  # 预处理完成时间
        print(f"预处理耗时: {preprocess_time - start_time:.3f}s")  # 输出耗时

        with torch.no_grad():  # 关闭梯度计算
            try:  # 前向推理
                segmentation_logits = self.recognizer(input_graph)  # 获取分割logits
            except Exception as e:  # 推理异常
                self.msgBox.warning(self, "warning", f"推理失败: {e}")  # 提示
                return  # 直接返回

            inference_time = time.time()  # 前向结束时间
            print(f"推理耗时: {inference_time - preprocess_time:.3f}s")  # 输出耗时

            face_logits_np = segmentation_logits.cpu().numpy()  # 转为numpy数组
            predicted_classes = np.argmax(face_logits_np, axis=1)  # 按行取最大值索引

        del input_graph, segmentation_logits, face_logits_np  # 删除推理中间变量节省内存

        class_to_faces_map: Dict[int, List[int]] = {}  # 类别到面索引的映射字典
        for face_idx, class_idx in enumerate(predicted_classes.tolist()):  # 遍历预测结果
            class_to_faces_map.setdefault(class_idx, []).append(face_idx)  # 聚合同类别的面

        self._populate_feature_list(class_to_faces_map)  # 更新特征列表

        postprocess_time = time.time()  # 后处理完成时间
        print(f"后处理耗时: {postprocess_time - inference_time:.3f}s")  # 输出耗时
        print(f"总耗时: {time.time() - start_time:.3f}s")  # 输出总耗时

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

    def featureListDoubleClicked(self):
        """双击列表项高亮选中类别：将选中类别的面着色不透明，其他面变白色半透明"""
        if not (self.ais_shape and self.file_name and len(self.features_list) != 0):  # 前置条件检查
            return  # 直接返回

        selected_row_idx = self.featureListWidget.currentRow()  # 获取选中的行索引
        if selected_row_idx < 0 or selected_row_idx >= len(self.features_list):  # 越界保护
            return  # 直接返回

        selected_feature = self.features_list[selected_row_idx]  # 获取选中的特征对象
        selected_class_idx = self.class_index_by_row[selected_row_idx]  # 获取对应的类别索引

        self.ais_shape.ClearCustomAspects()  # 清除所有自定义渲染属性
        for face_obj in self.faces_list:  # 遍历所有面对象
            self.ais_shape.SetCustomColor(face_obj, rgb_color(1, 1, 1))  # 设置为白色
            self.ais_shape.SetCustomTransparency(face_obj, 0.6)  # 设置60%透明度

        highlight_color = self._class_color(selected_class_idx)  # 计算高亮颜色
        for face_idx in selected_feature.faces:  # 遍历选中类别的面索引
            face_obj = self.faces_list[face_idx]  # 获取对应的面对象
            self.ais_shape.SetCustomColor(face_obj, rgb_color(*highlight_color))  # 设置类别颜色
            self.ais_shape.SetCustomTransparency(face_obj, 0.0)  # 设置不透明

        self.display.Context.Display(self.ais_shape, True)  # 刷新3D显示上下文

    def openStepFolder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择STEP文件夹", "./")  # 打开文件夹对话框
        if not folder_path:  # 用户取消选择
            return  # 直接返回
        self.step_dir = Path(folder_path)  # 保存STEP文件夹
        self.step_files = self._scan_step_files(self.step_dir)  # 扫描STEP文件
        self._populate_step_list()  # 更新列表
        self._save_state()  # 保存状态
        self._update_window_title()  # 更新标题

    def openLabelFolder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择标签文件夹", "./")  # 打开文件夹对话框
        if not folder_path:  # 用户取消选择
            return  # 直接返回
        self.label_dir = Path(folder_path)  # 保存标签文件夹
        self._save_state()  # 保存状态
        self._update_window_title()  # 更新标题

    def toggleGTVisualization(self):
        if not (self.ais_shape and self.file_name):  # 未加载STEP
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
            self.ais_shape.ClearCustomAspects()  # 清除自定义颜色
            self.display.Context.Display(self.ais_shape, True)  # 刷新显示

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
        self.gt_label_path = label_path  # 记录路径
        self.gt_label_data = label_data  # 记录原始数据
        self.gt_label_format = self._detect_label_format(label_data)  # 记录格式

        return self._extract_seg_labels(label_data, len(self.faces_list))  # 解析标签

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

    def _populate_feature_list(self, class_to_faces_map: Dict[int, List[int]]):
        self.featureListWidget.clear()  # 清空列表控件
        self.features_list.clear()  # 清空特征聚合列表
        self.class_index_by_row.clear()  # 清空行索引映射

        for class_idx, face_indices in class_to_faces_map.items():  # 遍历类别映射
            class_name = self._class_name(class_idx)  # 获取类别名称
            self.features_list.append(FeatureClass(name=class_name, faces=face_indices))  # 保存聚合结果
            self.class_index_by_row.append(class_idx)  # 记录行到类别的映射
            self.featureListWidget.addItem(f"{class_name} ({len(face_indices)})")  # 更新列表显示

    def _build_face_index(self):
        self.face_index_by_hash.clear()  # 清空映射
        for idx, face in enumerate(self.faces_list):  # 遍历面
            self.face_index_by_hash[face.HashCode(2147483647)] = idx  # 哈希索引

    def _populate_face_list(self):
        if not hasattr(self, "faceListWidget"):
            return
        self.faceListWidget.clear()
        for idx in range(len(self.faces_list)):
            self.faceListWidget.addItem(f"Face {idx}")

    def faceListItemClicked(self):
        row = self.faceListWidget.currentRow()  # 获取选中行
        if row < 0 or row >= len(self.faces_list):  # 越界保护
            return
        self.selected_face_idx = row  # 更新选中面
        self._refresh_selection_colors()  # 刷新颜色

    def faceListContextMenu(self, pos):
        item = self.faceListWidget.itemAt(pos)  # 获取右键项
        if item is None:
            return
        row = self.faceListWidget.row(item)  # 获取行
        if row < 0 or row >= len(self.faces_list):
            return
        self.selected_face_idx = row  # 更新选中面
        self._refresh_selection_colors()  # 刷新颜色
        self._edit_selected_face_label()  # 右键编辑

    def toggleFaceSelection(self):
        if not self.ais_shape:  # 未加载STEP
            self.msgBox.warning(self, "警告", "请先加载STEP文件")  # 提示
            return  # 直接返回
        self.face_select_enabled = not self.face_select_enabled  # 切换选择模式
        if self.face_select_enabled:  # 开启
            self.display.Context.Activate(self.ais_shape, TopAbs_FACE, True)  # 启用面选择
        else:  # 关闭
            self.display.Context.Deactivate(self.ais_shape)  # 关闭选择
            self.selected_face_idx = None  # 清空选中面
            self._refresh_selection_colors()  # 恢复颜色

    def _on_select(self, selected_shapes, *args, **kwargs):
        if not self.face_select_enabled:  # 未开启选择
            return  # 直接返回
        if not selected_shapes:  # 未选中
            return  # 直接返回
        selected_shape = selected_shapes[0]  # 取第一个
        face_idx = self.face_index_by_hash.get(selected_shape.HashCode(2147483647))  # 查找索引
        if face_idx is None:  # 未匹配
            return  # 直接返回
        if self.selected_face_idx == face_idx:  # 再次点击同一面则取消
            self.selected_face_idx = None  # 取消选中
        else:  # 选择新面
            self.selected_face_idx = face_idx  # 更新选中面

        self._refresh_selection_colors()  # 刷新颜色

    def eventFilter(self, obj, event):
        if obj is self.canvas and self.face_select_enabled:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                self._edit_selected_face_label()  # 右键编辑
                return True
        return super().eventFilter(obj, event)

    def _edit_selected_face_label(self):
        if self.selected_face_idx is None:  # 未选中面
            self.msgBox.warning(self, "警告", "请先选中一个面")  # 提示
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
        current_label = self.current_gt_labels[self.selected_face_idx]  # 当前标签
        current_text = option_map.get(current_label, options[0])  # 默认项
        current_index = options.index(current_text) if current_text in options else 0

        selected_text, ok = QInputDialog.getItem(
            self,
            "修改GT标签",
            "选择类别",
            options,
            current_index,
            False,
        )
        if not ok:
            return  # 取消

        new_label = int(selected_text.split(":", 1)[0])  # 解析标签
        self.current_gt_labels[self.selected_face_idx] = new_label  # 更新标签
        self._save_gt_labels(self.current_gt_labels)  # 写回文件

        if self.gt_enabled:  # 若开启GT显示
            self._apply_gt_labels_from_labels(self.current_gt_labels)  # 刷新显示

    def _label_options(self) -> Tuple[List[str], Dict[int, str]]:
        options: List[str] = []
        option_map: Dict[int, str] = {}
        if self.label_names:
            for idx, name in enumerate(self.label_names):
                text = f"{idx}: {name}"
                options.append(text)
                option_map[idx] = text
        else:
            max_label = 0
            if self.current_gt_labels:
                max_label = max(self.current_gt_labels)
            for idx in range(max_label + 1):
                text = f"{idx}: class_{idx}"
                options.append(text)
                option_map[idx] = text
        if not options:
            options = ["0: class_0"]
            option_map[0] = "0: class_0"
        return options, option_map

    def _apply_gt_labels_from_labels(self, labels: List[int]):
        self.ais_shape.ClearCustomAspects()  # 清除自定义颜色
        class_to_faces_map: Dict[int, List[int]] = {}  # 类别到面索引映射
        for face_idx, class_idx in enumerate(labels):  # 遍历标签
            class_to_faces_map.setdefault(class_idx, []).append(face_idx)  # 归类
            face_obj = self.faces_list[face_idx]  # 获取面
            color = self._class_color(class_idx)  # 颜色
            self.ais_shape.SetCustomColor(face_obj, rgb_color(*color))  # 设置颜色
            self.ais_shape.SetCustomTransparency(face_obj, 0.0)  # 设置不透明

        self._apply_selection_overlay()  # 叠加选中高亮

        self.display.Context.Display(self.ais_shape, True)  # 刷新显示
        self._populate_feature_list(class_to_faces_map)  # 更新特征列表

    def _refresh_selection_colors(self):
        if self.gt_enabled and self.current_gt_labels:  # 有GT颜色
            self._apply_gt_labels_from_labels(self.current_gt_labels)  # 刷新GT
            return

        self.ais_shape.ClearCustomAspects()  # 清除颜色
        self._apply_selection_overlay()  # 仅高亮选中
        self.display.Context.Display(self.ais_shape, True)  # 刷新显示

    def _apply_selection_overlay(self):
        if self.selected_face_idx is None:  # 无选中
            return  # 直接返回
        face_idx = self.selected_face_idx  # 当前选中面
        if 0 <= face_idx < len(self.faces_list):  # 越界保护
            face_obj = self.faces_list[face_idx]  # 获取面
            self.ais_shape.SetCustomColor(face_obj, rgb_color(1.0, 1.0, 0.0))  # 高亮颜色
            self.ais_shape.SetCustomTransparency(face_obj, 0.0)  # 不透明

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

    def _save_gt_labels(self, labels: List[int]):
        if not self.gt_label_path or not self.gt_label_format:
            return
        data = self.gt_label_data
        if self.gt_label_format == "list":
            data = [int(x) for x in labels]
        elif self.gt_label_format == "dict" and isinstance(data, dict):
            data = self._update_label_dict(data, labels)
        elif self.gt_label_format == "pair" and isinstance(data, list) and len(data) == 2:
            data = [data[0], self._update_label_dict(data[1], labels)]
        elif self.gt_label_format == "pair_list" and isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict):
                data[0] = [item[0], self._update_label_dict(item[1], labels)]
        else:
            return

        try:
            if self.gt_label_path.suffix.lower() == ".pkl":
                with open(self.gt_label_path, "wb") as f:
                    pickle.dump(data, f)
            else:
                with open(self.gt_label_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=True, indent=2)
        except Exception:
            return

    def stepListItemClicked(self):
        row = self.stepListWidget.currentRow()  # 获取选中行
        if row < 0 or row >= len(self.step_files):  # 越界保护
            return  # 直接返回
        self._load_step_file(self.step_files[row])  # 加载选中的STEP

    def _scan_step_files(self, step_dir: Path) -> List[Path]:
        if not step_dir.exists():  # 目录不存在
            return []  # 返回空列表
        step_files = list(step_dir.glob("*.stp")) + list(step_dir.glob("*.step"))  # 收集STEP文件
        step_files = sorted(step_files, key=lambda p: p.name.lower())  # 按文件名排序
        return step_files  # 返回文件列表

    def _populate_step_list(self):
        if not hasattr(self, "stepListWidget"):  # 列表未创建
            return  # 直接返回
        self.stepListWidget.clear()  # 清空列表
        for step_path in self.step_files:  # 遍历文件
            self.stepListWidget.addItem(step_path.name)  # 添加文件名

        if self.last_step_file:  # 若存在上次选择
            for idx, step_path in enumerate(self.step_files):  # 查找匹配项
                if step_path == self.last_step_file:  # 路径一致
                    self.stepListWidget.setCurrentRow(idx)  # 选中
                    break

    def _restore_step_list(self):
        if self.step_dir and not self.step_files:  # 有目录但无列表
            self.step_files = self._scan_step_files(self.step_dir)  # 重新扫描
        self._populate_step_list()  # 填充列表

        if self.last_step_file and self.last_step_file.exists():  # 自动加载上次文件
            self._load_step_file(self.last_step_file)  # 加载文件

    def _state_path(self) -> Path:
        return self.project_root / "qt5_visualization_state.json"  # 状态文件路径

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

    def _save_state(self):
        state_path = self._state_path()  # 状态文件路径
        state = {
            "step_dir": str(self.step_dir) if self.step_dir else None,
            "label_dir": str(self.label_dir) if self.label_dir else None,
            "step_files": [str(p) for p in self.step_files],
            "last_step": str(self.last_step_file) if self.last_step_file else None,
            "config_path": str(self.config_path) if self.config_path else None,
            "ckpt_path": str(self.ckpt_path) if self.ckpt_path else None,
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
