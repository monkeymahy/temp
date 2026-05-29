# Instance Label 系统开发文档

## 1. 当前目标

当前实现把 instance label 分成两套标签体系：

- `labels_full`：完整版本标签，用于可视化检查、人工修订、版本管理、推理结果保存。
- `labels_train/<export_id>/labels`：训练快照标签，用于模型训练，导出后不可被可视化编辑覆盖。

推荐总流程：

```text
MFInstSeg / 旧训练标签
  -> create_full_labels_from_train_labels.py
  -> labels_full
  -> qt5_visualization.py 检查、修订、保存版本
  -> export_train_labels.py
  -> labels_train/<export_id>/labels
  -> SFDataset(task_mode=seg_inst)
  -> AAGNetSegmentor(enable_inst_head=true)
```

## 2. 标签体系

### 2.1 完整版本标签：labels_full

完整标签只使用 JSON dict。可视化工具只允许加载该格式，禁止直接加载纯 list、MFInstSeg 原始标签或训练快照，避免原始标签被保存逻辑污染。

核心结构：

```json
{
  "sample_id": "graphs_00000001",
  "labels": [0, 1, 1, 0],
  "labels_base": [0, 1, 1, 0],
  "version_id": 1,
  "versions": [],
  "domains": {
    "geometry": {
      "face": [0, 1, 1, 0]
    },
    "instance": {
      "face_instance": [-1, 0, 0, -1],
      "instances": [
        {
          "instance_id": 0,
          "class_id": 1,
          "face_indices": [1, 2]
        }
      ]
    },
    "instance_base": {
      "face_instance": [-1, 0, 0, -1],
      "instances": [
        {
          "instance_id": 0,
          "class_id": 1,
          "face_indices": [1, 2]
        }
      ]
    }
  }
}
```

字段说明：

- `labels`：当前最新 face 语义标签，兼容旧代码。
- `labels_base`：版本回放的 face 标签基线。
- `domains.geometry.face`：完整标签中的 face 语义标签域，和 `labels` 保持一致。
- `domains.instance.face_instance`：每个 face 的实例归属，背景或非实例 face 为 `-1`。
- `domains.instance.instances`：实例对象列表，供可视化展示和编辑。
- `domains.instance_base`：实例版本回放基线。
- `versions`：修改历史，记录 face label 和 instance label 的增量操作。

`domains.instance` 可为空或不存在，表示当前样本只有 face 语义标签。若存在，则 `face_instance` 与 `instances[*].face_indices` 必须一致。

### 2.2 训练快照标签：labels_train

`seg_only` 保持旧格式，单文件是纯 face label list：

```json
[0, 1, 1, 0]
```

`seg_inst` 使用 MFInstSeg 风格，但只保留 `seg` 和 `inst`，不导出 `bottom`：

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

训练标签规则：

- `seg[str(i)]` 是 face `i` 的语义类别。
- `inst[i][j] = 1` 表示 face `i` 与 face `j` 属于同一个实例。
- 背景 face 对应 `inst` 行列应为 0。
- 单面实例保留对角线 `1`。
- 历史 MFInstSeg 标签如果带 `bottom`，读取时会忽略；新导出的训练标签不生成 `bottom`。

## 3. 标签互相转换

### 3.1 MFInstSeg/旧训练标签 -> 完整版本标签

脚本：

- `v2/utils/create_full_labels_from_train_labels.py`

命令：

```bash
python v2\utils\create_full_labels_from_train_labels.py ^
  --labels-dir C:\Data\SF-JSON\mfinstseg_labels ^
  --output-dir C:\Data\SF-JSON\labels_full ^
  --author generated ^
  --background-labels 0 ^
  --overwrite
```

输入支持：

- 纯 list：`[0, 1, 1, 0]`
- MFInstSeg 单样本 pair：`[sample_id, {"seg": ..., "inst": ...}]`
- MFInstSeg 外层 list：`[[sample_id, {"seg": ..., "inst": ...}]]`

输出行为：

- 生成完整版本标签 dict。
- 若输入包含 `inst`，会从邻接矩阵还原 `domains.instance` 和 `domains.instance_base`。
- 输出文件名优先使用 MFInstSeg 内层 `sample_id`。
- 忽略输入中的 `bottom`。
- 单个文件失败时跳过并继续。

报告机制：

- 输出 `manifest.json`。
- 输出 `conversion_report.json`。
- `skipped` 记录失败文件、错误类型和原因。
- 控制台打印 `[skip] ...` 和最终成功/跳过数量。

### 3.2 完整版本标签 -> 训练快照

脚本：

- `v2/utils/export_train_labels.py`

导出 `seg_only`：

```bash
python v2\utils\export_train_labels.py ^
  --labels-full-dir C:\Data\SF-JSON\labels_full ^
  --export-id seg_only_v1 ^
  --task-mode seg_only ^
  --output-root C:\Data\SF-JSON\labels_train\seg_only_v1
```

导出 `seg_inst`：

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
- `--use-manifest-versions`：按 manifest 中的 `version_id` 导出历史版本。
- `--class-mapping path\to\mapping.json`：导出时映射类别 id。

导出 manifest 记录：

- `export_id`
- `task_mode`
- `label_format`
- `labels_full_dir`
- `class_mapping`
- `background_labels`
- `items[*].sample_id`
- `items[*].version_id`
- `items[*].label_path`
- `items[*].full_label_path`

## 4. 训练数据加载

涉及文件：

- `v2/dataset/SFdataset.py`
- `v2/dataset/SFdatamodule.py`
- `v2/utils/instance_label_utils.py`

### 4.1 SFDataset

`SFDataset` 新增 `task_mode`：

```python
SFDataset(..., task_mode="seg_only")
SFDataset(..., task_mode="seg_inst")
```

`seg_only` 行为：

- 读取纯 list 或完整标签中的 face labels。
- 写入 `graph.ndata["y"]`。
- 不要求 instance 字段。

`seg_inst` 行为：

- 读取 MFInstSeg 风格训练标签中的 `seg` 和 `inst`。
- `seg` 写入 `graph.ndata["y"]`。
- `inst` 写入样本字段 `inst_y`。
- 如果不是训练快照，而是完整标签，则可由 `face_instance` 动态生成 `inst_y`。
- 校验 `inst` 必须是 `N x N` 方阵，尺寸等于 face 数，且对称。

### 4.2 SFDataModule

`SFDataModule._collate` 在 `seg_inst` 下额外返回：

```python
{
  "graph": batched_graph,
  "filename": batched_filenames,
  "inst_labels": padded_adj,
  "inst_mask": padded_mask,
  "num_faces": num_faces_tensor
}
```

说明：

- `inst_labels` 形状为 `[B, max_faces, max_faces]`。
- `inst_mask` 同形状，padding 区域为 0。
- `num_faces` 保存每个样本真实 face 数。
- 模型 loss 和指标用 `inst_mask` 排除 padding。

## 5. 训练配置

### 5.1 seg_only

```yaml
model:
  enable_inst_head: false

data:
  root_dir: C:\Data\SF-JSON
  label_dir: labels_train\seg_only_v1
  task_mode: seg_only
```

### 5.2 seg_inst

```yaml
model:
  enable_inst_head: true
  inst_loss_weight: 1.0

data:
  root_dir: C:\Data\SF-JSON
  label_dir: labels_train\seg_inst_v1
  task_mode: seg_inst
```

训练注意事项：

- `enable_inst_head=false` 时模型保持原 face segmentation 行为。
- `enable_inst_head=true` 时 `forward` 返回 `{"seg_logits": ..., "inst_logits": ...}`。
- loss 为 `CrossEntropyLoss + inst_loss_weight * masked BCEWithLogitsLoss`。
- instance 指标包括 `inst_acc` 和 `inst_f1`。
- 旧 checkpoint 没有实例头时，应保持 `enable_inst_head=false` 加载。

常用命令：

```bash
python v2\main.py fit --data SFDataModule --config v2\configs\sf_csy.yaml
python v2\main.py test --data SFDataModule --config v2\configs\sf.yaml --ckpt_path output\checkpoints\model.ckpt
```

## 6. 可视化工具功能

涉及文件：

- `v2/utils/qt5_visualization.py`
- `v2/utils/instance_postprocess.py`

### 6.1 加载保护

可视化 GT 标签只允许加载完整版本标签 dict。必须包含：

- `labels`
- `labels_base`
- `version_id`
- `versions`
- `domains.geometry.face`

禁止直接加载：

- 纯 list 训练标签。
- MFInstSeg 原始标签。
- `labels_train` 训练快照。

这样可以避免可视化保存逻辑覆盖或污染原始标签。

### 6.2 展示能力

- GT / Prediction 双视口。
- 语义类别着色。
- 实例 ID 着色。
- GT 实例列表。
- Prediction 实例列表。
- 点击实例列表可选中对应 faces。
- 点击/框选 face 后同步列表选择。

### 6.3 编辑能力

GT 右键菜单支持：

- 修改 face 语义标签。
- 新建实例。
- 加入实例。
- 移出实例。
- 合并实例。
- 拆分实例。
- 修改实例类别。

修改实例类别时，可选择同步修改该实例内 face 的语义标签。

### 6.4 保存与版本

可视化保存规则：

- 面标签变化写入 `geometry.face` 和 `labels`。
- 实例变化写入 `domains.instance`。
- 本次会话修改会在切换样本或清空时聚合成一个版本记录。
- 版本记录可同时包含 face label change 和 `instance_change`。
- `domains.instance_base` 用于回放任意历史版本。

历史版本能力：

- 版本列表展示 current、base、历史版本。
- 可预览历史 face 标签和 instance 标签。
- 可回滚历史版本并保存成新的当前状态。

### 6.5 推理结果

若模型启用实例头：

- 推理输出包含 `seg_logits` 和 `inst_logits`。
- `inst_logits` 经 `postprocess_instance_logits` 后生成：

```json
{
  "face_instance": [-1, 0, 0, -1],
  "instances": [
    {"instance_id": 0, "class_id": 1, "face_indices": [1, 2]}
  ]
}
```

预测标签保存时，如果包含实例结果，会保存为完整版本标签。

## 7. 关键工具函数

`v2/utils/instance_label_utils.py`：

- `extract_instance_from_payload`
- `normalize_instance_payload`
- `face_instance_to_adj`
- `adj_to_instance_payload`
- `validate_instance_struct`
- `validate_inst_adj`
- `build_instance_change_ops`
- `rollback_instance_payload`

`v2/utils/instance_postprocess.py`：

- `postprocess_instance_logits`

## 8. 验证建议

语法检查：

```bash
python -m py_compile ^
  v2\utils\instance_label_utils.py ^
  v2\utils\instance_postprocess.py ^
  v2\utils\create_full_labels_from_train_labels.py ^
  v2\utils\export_train_labels.py ^
  v2\dataset\SFdataset.py ^
  v2\dataset\SFdatamodule.py ^
  v2\models\segmentors.py ^
  v2\utils\qt5_visualization.py
```

推荐 smoke test：

1. 用少量 MFInstSeg 标签生成 `labels_full`，确认 `domains.instance` 和 `domains.instance_base` 正确。
2. 打开可视化工具，只指向 `labels_full`，确认原始 MFInstSeg 或 `labels_train` 会被拒绝加载。
3. 编辑一个实例并保存，重新打开后检查实例归属一致。
4. 导出 `seg_inst`，确认输出无 `bottom`，`seg` 覆盖所有 face，`inst` 尺寸等于 face 数。
5. 用 `task_mode=seg_inst` 跑 1-2 个 batch，确认 loss 非 NaN。

## 9. 当前限制

- 尚未在当前环境中完成真实数据端到端训练验证。
- Qt 可视化交互需要在带 GUI 和 OpenCascade 环境中人工验收。
- instance 推理后处理当前使用阈值和连通分量，复杂相邻实例仍可能需要后续优化。
