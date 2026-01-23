import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from OCC.Display.backend import load_backend
from OCC.Display.OCCViewer import rgb_color
from OCC.Core.AIS import AIS_ColoredShape
from OCC.Extend.TopologyUtils import TopologyExplorer
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QGridLayout,
    QDialog,
    QFileDialog,
    QMessageBox,
    QListWidget,
    QLabel,
    QMenu,
    QAction,
    QToolButton,
)

load_backend("qt-pyqt5")
import OCC.Display.qtDisplay as qtDisplay

import numpy as np
import torch
import yaml

from dataset.AAGExtractor import AAGExtractor, TopologyChecker
from utils.data_utils import load_body_from_step
from v2.models.segmentors import AAGNetSegmentor
from v2.utils.data_utils import (
    load_one_graph,
    load_json_or_pkl,
    load_statistics,
    standardization,
    center_and_scale,
)


class FeatureClass:
    def __init__(self, name: str, faces: List[int]):
        self.name = name
        self.faces = faces


class App(QDialog):
    def __init__(
        self,
        config_path: Optional[Path] = None,
        ckpt_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        super().__init__()

        # UI settings
        self.title = "AAGNet 可视化工具（v2, Lightning）"
        self.left = 300
        self.top = 300
        self.width = 1366
        self.height = 900
        self.canvas_width = 1000
        self.height_width = 700

        # members
        self.ais_shape = None
        self.file_name = None
        self.faces_list = []
        self.features_list: List[FeatureClass] = []
        self.class_index_by_row: List[int] = []

        # runtime settings
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = (
            config_path or self.project_root / "configs" / "sf_csy.yaml"
        )
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # data/model config (filled by _load_config)
        self.model_cfg: Dict = {}
        self.data_cfg: Dict = {}
        self.label_names: List[str] = []
        self.normalize = False
        self.do_center_and_scale = False
        self.attribute_schema = None
        self.stat = None

        self.topo_checker = TopologyChecker()

        self._load_config()
        self._load_feature_schema()
        self._load_statistics_if_needed()
        self._init_model()
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setMinimumSize(900, 600)
        self.setMaximumSize(1920, 1200)
        self.createHorizontalLayout()
        self._update_window_title()
        self.msgBox = QMessageBox()

        windowLayout = QHBoxLayout()
        windowLayout.addWidget(self.button_panel)
        windowLayout.addWidget(self.horizontalGroupBox)
        self.setLayout(windowLayout)
        self.show()

    def _load_config(self):
        if not self.config_path or not Path(self.config_path).exists():
            return
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.model_cfg = config.get("model", {}) or {}
        self.data_cfg = config.get("data", {}) or {}

        self.normalize = bool(self.data_cfg.get("normalize", False))
        self.do_center_and_scale = bool(
            self.data_cfg.get("center_and_scale", False)
        )
        self.label_names = self._resolve_label_names(
            config_path=Path(self.config_path),
            num_classes=self.model_cfg.get("num_classes"),
        )

    def _load_feature_schema(self):
        feature_schema_path = (
            self.project_root.parent / "feature_lists" / "all.json"
        )
        if feature_schema_path.exists():
            self.attribute_schema = load_json_or_pkl(feature_schema_path)

    def _load_statistics_if_needed(self):
        if not self.normalize:
            self.stat = None
            return

        root_dir = self.data_cfg.get("root_dir")
        if root_dir:
            stat_path = Path(root_dir) / "aag" / "attr_stat.json"
            if stat_path.exists():
                self.stat = load_statistics(stat_path)
                return

        fallback_stat = self.project_root.parent / "weights" / "attr_stat.json"
        if fallback_stat.exists():
            self.stat = load_statistics(fallback_stat)

    def _init_model(self):
        if not self.model_cfg:
            return

        torch.set_float32_matmul_precision("high")
        self.recognizer = AAGNetSegmentor(**self.model_cfg)

        if self.ckpt_path:
            ckpt_path = Path(self.ckpt_path)
            if ckpt_path.is_file():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt)
                self.recognizer.load_state_dict(state_dict, strict=False)

        self.recognizer = self.recognizer.to(self.device)
        self.recognizer.eval()

    def _resolve_label_names(
        self, config_path: Path, num_classes: Optional[int]
    ) -> List[str]:
        label_names = self.data_cfg.get("label_names") or self.model_cfg.get(
            "label_names"
        )
        if isinstance(label_names, list) and label_names:
            return label_names

        if not num_classes:
            return []

        name_lower = config_path.name.lower()
        if "sf" in name_lower:
            return ["other", "slot groove", "hole"]
        if "mfcad2" in name_lower:
            return ["other", "hole", "slot"]

        return [f"class_{i}" for i in range(num_classes)]

    def createHorizontalLayout(self):
        self.horizontalGroupBox = QWidget()
        layout = QHBoxLayout()
        self.button_panel = QWidget()
        self.button_panel.setFixedWidth(220)
        panel_layout = QGridLayout()

        window_menu = QMenu("调整窗口大小", self)
        action_small = QAction("小", self)
        action_small.triggered.connect(lambda: self._resize_window(1100, 700))
        action_medium = QAction("中", self)
        action_medium.triggered.connect(lambda: self._resize_window(1366, 900))
        action_large = QAction("大", self)
        action_large.triggered.connect(lambda: self._resize_window(1600, 1000))
        window_menu.addAction(action_small)
        window_menu.addAction(action_medium)
        window_menu.addAction(action_large)

        self.window_button = QToolButton(self)
        self.window_button.setText("调整窗口大小")
        self.window_button.setMenu(window_menu)
        self.window_button.setPopupMode(QToolButton.InstantPopup)
        self.window_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.window_button.setStyleSheet(
            "QToolButton {padding: 6px 12px; border: 1px solid #888; "
            "border-radius: 4px; background: #f2f2f2;}"
            "QToolButton:hover {background: #e6e6e6;}"
            "QToolButton::menu-indicator {image: none;}"
        )
        panel_layout.addWidget(
            self.window_button, 0, 0, 1, 1, alignment=Qt.AlignLeft
        )

        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        panel_layout.addWidget(self.status_label, 1, 0, 1, 1)

        btn_config = QPushButton("加载配置", self)
        btn_config.clicked.connect(self.openConfig)
        panel_layout.addWidget(btn_config, 2, 0, 1, 1)

        btn_ckpt = QPushButton("加载权重", self)
        btn_ckpt.clicked.connect(self.openCheckpoint)
        panel_layout.addWidget(btn_ckpt, 3, 0, 1, 1)

        disp = QPushButton("加载 STEP", self)
        disp.clicked.connect(self.openShape)
        panel_layout.addWidget(disp, 4, 0, 1, 1)

        eras = QPushButton("关闭 STEP", self)
        eras.clicked.connect(self.eraseShape)
        panel_layout.addWidget(eras, 5, 0, 1, 1)

        feature_rec = QPushButton("运行分割", self)
        feature_rec.clicked.connect(self.featureRecog)
        panel_layout.addWidget(feature_rec, 6, 0, 1, 1)

        self.canvas = qtDisplay.qtViewer3d(self)
        layout.addWidget(self.canvas)
        self.canvas.resize(self.canvas_width, self.height_width)
        self.canvas.InitDriver()
        self.display = self.canvas._display

        self.featureListWidget = QListWidget()
        self.featureListWidget.itemDoubleClicked.connect(
            self.featureListDoubleClicked
        )
        panel_layout.addWidget(self.featureListWidget, 7, 0, 1, 1)

        self.horizontalGroupBox.setLayout(layout)
        self.button_panel.setLayout(panel_layout)

    def openConfig(self):
        config_path = QFileDialog.getOpenFileName(
            self,
            "Open Config YAML",
            str(self.project_root / "configs"),
            "(*.yaml *.yml)",
        )[0]
        if not config_path:
            return
        self.config_path = Path(config_path)
        self._load_config()
        self._load_statistics_if_needed()
        self._init_model()
        self._update_window_title()

    def openCheckpoint(self):
        ckpt_path = QFileDialog.getOpenFileName(
            self,
            "Open Checkpoint",
            str(self.project_root.parent / "output" / "checkpoints"),
            "(*.ckpt)",
        )[0]
        if not ckpt_path:
            return
        self.ckpt_path = Path(ckpt_path)
        self._init_model()
        self._update_window_title()

    def _resize_window(self, width: int, height: int):
        self.resize(width, height)

    def _update_window_title(self):
        config_name = self._short_name(self.config_path)
        ckpt_name = self._short_name(self.ckpt_path)
        step_name = self._short_name(self.file_name)
        title = f"{self.title} | 配置: {config_name} | 权重: {ckpt_name} | STEP: {step_name}"
        self.setWindowTitle(title)
        if hasattr(self, "status_label"):
            self.status_label.setText(
                f"配置: {config_name}\n权重: {ckpt_name}\nSTEP: {step_name}"
            )

    def _short_name(self, path_value) -> str:
        if not path_value:
            return "未选择"
        try:
            return Path(path_value).name
        except Exception:
            return "未选择"

    def openShape(self):
        self.file_name = QFileDialog.getOpenFileName(
            self, "Open Step File", "./", "(*.st*p)"
        )[0]
        if not self.file_name:
            return
        solid = load_body_from_step(self.file_name)
        if not self.topo_checker(solid):
            self.msgBox.warning(
                self, "warning", "Fail to load, unsupported or wrong STEP."
            )
            self.file_name = None
            return

        if self.ais_shape:
            self.display.Context.Erase(self.ais_shape, True)

        self.featureListWidget.clear()
        self.features_list.clear()

        self.ais_shape = AIS_ColoredShape(solid)
        self.display.Context.Display(self.ais_shape, True)
        self.display.FitAll()

        topo = TopologyExplorer(solid)
        self.faces_list = list(topo.faces())
        self._update_window_title()

    def eraseShape(self):
        if self.ais_shape:
            self.display.Context.Erase(self.ais_shape, True)
            self.ais_shape = None
            self.file_name = None
            self.featureListWidget.clear()
            self.features_list.clear()
            self.faces_list.clear()
            self._update_window_title()

    def featureRecog(self):
        if not (self.ais_shape and self.file_name and self.attribute_schema):
            return

        if not hasattr(self, "recognizer"):
            self.msgBox.warning(
                self,
                "warning",
                "Model not initialized. Please load config/ckpt.",
            )
            return

        start_time = time.time()
        try:
            aag_ext = AAGExtractor(self.file_name, self.attribute_schema)
            aag = aag_ext.process()
        except Exception as e:
            self.msgBox.warning(self, "warning", f"AAG extraction failed: {e}")
            return

        sample = load_one_graph(self.file_name, aag)
        if self.normalize and self.stat is not None:
            sample = standardization(sample, self.stat)
        if self.do_center_and_scale:
            sample = center_and_scale(sample)

        graph = sample["graph"].to(self.device)
        pre_time = time.time()
        print(f"Pre-processing duration: {pre_time - start_time}")

        with torch.no_grad():
            try:
                seg_out = self.recognizer(graph)
            except Exception as e:
                self.msgBox.warning(self, "warning", f"Inference failed: {e}")
                return

            ff_time = time.time()
            print(f"Feed-forward duration: {ff_time - pre_time}")

            face_logits = seg_out.cpu().numpy()
            pred = np.argmax(face_logits, axis=1)

        class_to_faces: Dict[int, List[int]] = {}
        for face_idx, class_idx in enumerate(pred.tolist()):
            class_to_faces.setdefault(class_idx, []).append(face_idx)

        self.featureListWidget.clear()
        self.features_list.clear()
        self.class_index_by_row.clear()

        for class_idx, faces in class_to_faces.items():
            name = self._class_name(class_idx)
            self.features_list.append(FeatureClass(name=name, faces=faces))
            self.class_index_by_row.append(class_idx)
            self.featureListWidget.addItem(f"{name} ({len(faces)})")

        post_time = time.time()
        print(f"Post-processing duration: {post_time - ff_time}")
        print(f"Total duration: {time.time() - start_time}s")

    def _class_name(self, class_idx: int) -> str:
        if 0 <= class_idx < len(self.label_names):
            return self.label_names[class_idx]
        return f"class_{class_idx}"

    def _class_color(self, class_idx: int) -> Tuple[float, float, float]:
        if class_idx < 0:
            return (1.0, 1.0, 1.0)
        hue = (class_idx * 0.61803398875) % 1.0
        return self._hsv_to_rgb(hue, 0.65, 0.95)

    def _hsv_to_rgb(
        self, h: float, s: float, v: float
    ) -> Tuple[float, float, float]:
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        return (v, p, q)

    def featureListDoubleClicked(self):
        if not (
            self.ais_shape and self.file_name and len(self.features_list) != 0
        ):
            return

        selected_row = self.featureListWidget.currentRow()
        if selected_row < 0 or selected_row >= len(self.features_list):
            return

        selected_cls = self.features_list[selected_row]
        class_idx = self.class_index_by_row[selected_row]

        self.ais_shape.ClearCustomAspects()
        for face in self.faces_list:
            self.ais_shape.SetCustomColor(face, rgb_color(1, 1, 1))
            self.ais_shape.SetCustomTransparency(face, 0.6)

        color = self._class_color(class_idx)
        for face_idx in selected_cls.faces:
            face = self.faces_list[face_idx]
            self.ais_shape.SetCustomColor(face, rgb_color(*color))
            self.ais_shape.SetCustomTransparency(face, 0.0)

        self.display.Context.Display(self.ais_shape, True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    if os.getenv("APPVEYOR") is None:
        sys.exit(app.exec_())
