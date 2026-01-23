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
from PyQt5.QtCore import Qt  # Qt常量
from PyQt5.QtWidgets import (  # Qt控件
    QApplication,  # 应用入口
    QWidget,  # 基础控件
    QPushButton,  # 按钮
    QHBoxLayout,  # 水平布局
    QGridLayout,  # 网格布局
    QDialog,  # 对话框基类
    QFileDialog,  # 文件对话框
    QMessageBox,  # 消息提示框
    QListWidget,  # 列表控件
    QLabel,  # 文本标签
    QMenu,  # 菜单
    QAction,  # 菜单项
    QToolButton,  # 工具按钮
)

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

        # 创建STEP关闭按钮
        btn_close_step = QPushButton("关闭 STEP", self)  # 关闭按钮
        btn_close_step.clicked.connect(self.eraseShape)  # 绑定点击事件
        panel_layout.addWidget(btn_close_step, 5, 0, 1, 1)  # 放置按钮

        # 创建分割执行按钮
        btn_segmentation = QPushButton("运行分割", self)  # 分割按钮
        btn_segmentation.clicked.connect(self.featureRecog)  # 绑定点击事件
        panel_layout.addWidget(btn_segmentation, 6, 0, 1, 1)  # 放置按钮

        # 创建3D画布
        self.canvas = qtDisplay.qtViewer3d(self)  # OCC画布
        canvas_layout.addWidget(self.canvas)  # 添加画布到右侧布局
        self.canvas.resize(self.canvas_width, self.height_width)  # 设置画布尺寸
        self.canvas.InitDriver()  # 初始化渲染驱动
        self.display = self.canvas._display  # 获取显示对象

        # 创建特征列表控件
        self.featureListWidget = QListWidget()  # 特征列表
        self.featureListWidget.itemDoubleClicked.connect(self.featureListDoubleClicked)  # 绑定双击事件
        panel_layout.addWidget(self.featureListWidget, 7, 0, 1, 1)  # 放置列表

        # 应用布局
        self.horizontalGroupBox.setLayout(canvas_layout)  # 设置右侧布局
        self.button_panel.setLayout(panel_layout)  # 设置左侧布局

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
        step_file_path = QFileDialog.getOpenFileName(self, "选择STEP文件", "./", "(*.st*p)")[0]  # 打开文件对话框
        if not step_file_path:  # 用户取消选择
            return  # 直接返回
        self.file_name = step_file_path  # 保存文件路径
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
        self._update_window_title()  # 更新窗口标题显示当前文件

    def eraseShape(self):
        if self.ais_shape:  # 若存在对象
            self.display.Context.Erase(self.ais_shape, True)  # 清除显示
            self.ais_shape = None  # 清空对象
            self.file_name = None  # 清空路径
            self.featureListWidget.clear()  # 清空列表
            self.features_list.clear()  # 清空结果
            self.faces_list.clear()  # 清空面列表
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

        self.featureListWidget.clear()  # 清空列表控件
        self.features_list.clear()  # 清空特征聚合列表
        self.class_index_by_row.clear()  # 清空行索引映射

        for class_idx, face_indices in class_to_faces_map.items():  # 遍历类别映射
            class_name = self._class_name(class_idx)  # 获取类别名称
            self.features_list.append(FeatureClass(name=class_name, faces=face_indices))  # 保存聚合结果
            self.class_index_by_row.append(class_idx)  # 记录行到类别的映射
            self.featureListWidget.addItem(f"{class_name} ({len(face_indices)})")  # 更新列表显示

        postprocess_time = time.time()  # 后处理完成时间
        print(f"后处理耗时: {postprocess_time - inference_time:.3f}s")  # 输出耗时
        print(f"总耗时: {time.time() - start_time:.3f}s")  # 输出总耗时

    def _class_name(self, class_idx: int) -> str:
        default_label_names = ["other", "slot groove", "hole"]  # 默认类别名顺序
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


if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建应用
    ex = App()  # 创建窗口
    if os.getenv("APPVEYOR") is None:  # CI环境跳过
        sys.exit(app.exec_())  # 进入事件循环
