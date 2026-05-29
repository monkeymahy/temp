# V2 版本代码说明

## 1. 标签体系

v2 当前支持两类标签：

- `labels_full`：完整版本标签，用于可视化检查、人工修改、版本回放和训练导出。
- `labels_train/<export_id>/labels`：训练快照标签，只用于训练，不在训练过程中修改。

完整版本标签是 JSON dict，核心字段包括：

```json
{
  "sample_id": "graphs_00000001",
  "labels": [0, 1, 1, 0],
  "labels_base": [0, 1, 1, 0],
  "version_id": 1,
  "versions": [],
  "domains": {
    "geometry": {"face": [0, 1, 1, 0]},
    "instance": {
      "face_instance": [-1, 0, 0, -1],
      "instances": [
        {"instance_id": 0, "class_id": 1, "face_indices": [1, 2]}
      ]
    },
    "instance_base": {
      "face_instance": [-1, 0, 0, -1],
      "instances": [
        {"instance_id": 0, "class_id": 1, "face_indices": [1, 2]}
      ]
    }
  }
}
```

`seg_only` 训练标签仍保持旧格式，即纯 face label list：

```json
[0, 1, 1, 0]
```

`seg_inst` 训练标签按 MFInstSeg 风格导出，外层是单样本 list，内部是 `[sample_id, label_dict]`。本项目只保留 `seg` 和 `inst`，不导出 `bottom`：

```json
[
  [
    "graphs_00000001",
    {
      "seg": {"0": 0, "1": 1, "2": 1, "3": 0},
      "inst": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    }
  ]
]
```

## 2. 标签脚本

### 2.1 从 MFInstSeg/训练标签生成完整版本标签

用于把已有 MFInstSeg 风格标签或旧纯 list 标签转成 `labels_full`：

```bash
python v2\utils\create_full_labels_from_train_labels.py ^
  --labels-dir C:\Data\SF-JSON\mfinstseg_labels ^
  --output-dir C:\Data\SF-JSON\labels_full ^
  --author generated ^
  --background-labels 0 ^
  --overwrite
```

说明：

- 输入支持纯 list、`[sample_id, {"seg": ..., "inst": ...}]`、`[[sample_id, {"seg": ..., "inst": ...}]]`。
- 若输入带 `inst`，脚本会生成 `domains.instance` 和 `domains.instance_base`。
- 历史 MFInstSeg 标签里如果有 `bottom`，该字段会被忽略。
- 输出文件名优先使用 MFInstSeg 内层 `sample_id`。
- 单个文件读取、解析、转换或写入失败时会跳过并继续处理其他文件。
- 转换结束会在输出目录写入 `manifest.json` 和 `conversion_report.json`，其中 `skipped` 会记录失败文件、错误类型和原因。

### 2.2 从完整标签导出训练快照

导出面分割训练标签：

```bash
python v2\utils\export_train_labels.py ^
  --labels-full-dir C:\Data\SF-JSON\labels_full ^
  --export-id seg_only_v1 ^
  --task-mode seg_only ^
  --output-root C:\Data\SF-JSON\labels_train\seg_only_v1
```

导出面分割 + 实例训练标签：

```bash
python v2\utils\export_train_labels.py ^
  --labels-full-dir C:\Data\SF-JSON\labels_full ^
  --export-id seg_inst_v1 ^
  --task-mode seg_inst ^
  --output-root C:\Data\SF-JSON\labels_train\seg_inst_v1 ^
  --background-labels 0
```

可选参数：

- `--manifest path\to\manifest.json`：只导出 manifest 中列出的样本。
- `--use-manifest-versions`：按 manifest 的 `version_id` 导出历史版本。
- `--class-mapping path\to\mapping.json`：导出时映射类别 id。

导出的 `manifest.json` 会记录 `task_mode`、`label_format`、`labels_full_dir`、`class_mapping`、`background_labels` 和每个样本的完整标签路径。

## 3. 训练配置

### 3.1 seg_only

旧训练路径不需要实例标签：

```yaml
model:
  enable_inst_head: false

data:
  root_dir: C:\Data\SF-JSON
  label_dir: labels_train\seg_only_v1
  task_mode: seg_only
```

### 3.2 seg_inst

启用实例头时，训练标签目录必须指向 `seg_inst` 快照：

```yaml
model:
  enable_inst_head: true
  inst_loss_weight: 1.0

data:
  root_dir: C:\Data\SF-JSON
  label_dir: labels_train\seg_inst_v1
  task_mode: seg_inst
```

`SFDataset(task_mode=seg_inst)` 会读取 MFInstSeg 风格标签中的 `seg` 和 `inst`：

- `graph.ndata["y"]`：face 语义标签。
- `inst_y`：单样本 `N x N` 同实例邻接矩阵。
- batch 后额外返回 `inst_labels`、`inst_mask`、`num_faces`。

训练注意事项：

- `seg_inst` 标签必须有合法 `inst` 矩阵，尺寸等于 face 数且对称。
- padding 区域通过 `inst_mask` 排除，不参与 instance loss 和指标。
- `enable_inst_head=true` 时模型输出 `{"seg_logits": ..., "inst_logits": ...}`。
- 旧 checkpoint 继续使用时应保持 `enable_inst_head=false`，除非 checkpoint 本身包含实例头。

## 4. 常用命令

```bash
# 训练
python v2\main.py fit --data SFDataModule --config v2\configs\sf_csy.yaml

# 测试
python v2\main.py test --data SFDataModule --config v2\configs\sf.yaml --ckpt_path output\checkpoints\model.ckpt

# 启动可视化工具
python v2\utils\qt5_visualization.py

# 离线旋转增强
python v2\utils\offline_rotate_augment.py --input-root C:\Data\SF-JSON --output-root C:\Data\SF-JSON-Aug
```

## 5. 可视化与版本

可视化工具支持：

- 只加载完整版本标签 dict，不直接加载纯 list、MFInstSeg 原始标签或训练快照，避免原始标签被可视化保存逻辑污染。
- 加载 `labels_full` 中的 `domains.geometry.face` 和 `domains.instance`。
- 按实例 ID 或语义类别着色。
- GT / Prediction 实例列表展示和点击选面。
- 新建、加入、移出、合并、拆分实例，以及修改实例类别。
- GT 编辑先进入内存和撤销/重做栈，不实时落盘。
- 点击 `保存GT修改`、切换 STEP、关闭窗口、清空模型或切换 GT 显示状态时，将本次会话修改追加到 `versions`。
- 点击版本列表即可预览历史版本；`回滚为新版本` 会按约定生成新的版本记录。

推荐流程是：先用模型或 MFInstSeg 标签生成 `labels_full`，在可视化工具中检查和修订，再导出 `labels_train/<export_id>` 训练快照。若手上只有 MFInstSeg 原始标签，必须先运行 `create_full_labels_from_train_labels.py` 转换，不能直接把原始标签目录作为可视化 GT 标签目录。

## 6. 快速验证

语法检查：

```bash
python -m py_compile ^
  v2\utils\instance_label_utils.py ^
  v2\utils\create_full_labels_from_train_labels.py ^
  v2\utils\export_train_labels.py ^
  v2\dataset\SFdataset.py ^
  v2\dataset\SFdatamodule.py ^
  v2\models\segmentors.py ^
  v2\utils\qt5_visualization.py
```

建议在真实数据上额外做三步 smoke test：

1. 用少量 MFInstSeg 标签生成 `labels_full`，检查 `domains.instance` 是否正确。
2. 从 `labels_full` 导出 `seg_inst`，确认训练标签无 `bottom` 且 `seg/inst` 尺寸一致。
3. 用 `task_mode=seg_inst` 跑 1-2 个 batch，确认 loss 不是 NaN。
