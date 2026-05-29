# Instance Label 开发总结

## 1. 本次完成范围

本次按 `docs/instance_label_development_plan.md` 完成了 instance label 的核心训练闭环：

1. 新增 instance label 工具层。
2. 扩展训练标签导出，支持 `seg_inst`。
3. 扩展 SF 数据集加载，支持 `task_mode=seg_inst`。
4. 扩展 v2 Lightning 模型，支持可选 instance head。
5. 新增 instance 推理后处理工具。
6. 完成可视化工具中的 instance label 检查、修改、保存与版本记录。

现有 `seg_only` 路径默认不变；所有 instance 训练能力都需要显式开启。

## 2. 主要改动文件

### 2.1 标签工具

新增：

- `v2/utils/instance_label_utils.py`
- `v2/utils/instance_postprocess.py`

能力：

- `face_instance -> inst` 邻接矩阵转换。
- `inst -> face_instance + instances` 转换。
- instance payload 标准化。
- instance payload 一致性校验。
- instance 版本操作回放。
- instance logits 后处理为实例集合。

### 2.2 训练标签导出

修改：

- `v2/utils/export_train_labels.py`

新增参数：

```bash
--task-mode seg_only|seg_inst
--class-mapping path/to/mapping.json
--background-labels 0
```

`seg_only` 保持旧行为，继续导出纯 list：

```json
[0, 1, 1, 0]
```

`seg_inst` 导出：

```json
{
  "seg": [0, 1, 1, 0],
  "inst": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
}
```

### 2.3 数据加载

修改：

- `v2/dataset/SFdataset.py`
- `v2/dataset/SFdatamodule.py`

新增配置：

```yaml
data:
  task_mode: seg_inst
```

`seg_inst` batch 会额外返回：

```python
{
  "inst_labels": padded_adj,
  "inst_mask": padded_mask,
  "num_faces": num_faces_tensor
}
```

### 2.4 模型

修改：

- `v2/models/segmentors.py`

新增模型配置：

```yaml
model:
  enable_inst_head: true
  inst_loss_weight: 1.0
```

默认 `enable_inst_head=false`，旧训练配置不需要修改。

开启实例头后：

- `forward` 返回 `{"seg_logits": ..., "inst_logits": ...}`。
- 训练时计算 `CrossEntropyLoss + masked BCEWithLogitsLoss`。
- padding 区域通过 `inst_mask` 排除。
- 记录 `inst_acc` 和 `inst_f1`。

### 2.5 可视化实例编辑

修改：

- `v2/utils/qt5_visualization.py`

已完成：

- 兼容开启实例头模型的 dict 输出：取 `seg_logits` 显示面分割结果，取 `inst_logits` 后处理为实例集合。
- 加载 `domains.instance` 后可按实例着色，也可切换为语义类别着色。
- GT / Prediction 列表在实例着色模式下显示实例列表，点击实例可选中对应 face。
- GT 右键菜单支持新建、加入、移出、合并、拆分实例，以及修改实例类别。
- 保存 GT 修改时写入 `domains.instance` 与 `domains.instance_base`。
- 窗口切换或清空前会将本次会话修改汇总为版本记录。
- 版本记录可同时包含 `geometry.face` 修改与 `instance_change` 修改，回放时可恢复 instance label。

注意：

- 预测标签如果带实例结果，保存时会生成完整版本化标签。
- 若从历史版本预览状态继续编辑，会以预览版本作为本次编辑基线，避免把“预览切换”误记为用户修改。

## 3. 使用方式

### 3.1 导出 seg_inst 训练标签

```bash
python v2/utils/export_train_labels.py \
  --labels-full-dir C:\Data\SF-JSON\labels_full \
  --export-id dev_seg_inst \
  --task-mode seg_inst \
  --output-root C:\Data\SF-JSON\labels_train\dev_seg_inst
```

### 3.2 配置训练

```yaml
model:
  enable_inst_head: true
  inst_loss_weight: 1.0

data:
  label_dir: labels_train\dev_seg_inst
  task_mode: seg_inst
```

### 3.3 保持原面分割训练

不配置 `task_mode` 和 `enable_inst_head` 即可保持原行为：

```yaml
model:
  enable_inst_head: false

data:
  task_mode: seg_only
```

其中两项都有默认值，可以省略。

## 4. 验证结果

已完成：

```bash
python -m py_compile v2\utils\instance_label_utils.py v2\utils\instance_postprocess.py v2\utils\export_train_labels.py v2\dataset\SFdataset.py v2\dataset\SFdatamodule.py v2\models\segmentors.py v2\utils\qt5_visualization.py
```

已完成 instance 标签转换快速验证：

```text
face_instance -> inst -> face_instance
校验错误列表为空
```

已完成 instance 版本回放快速验证：

```text
instance version replay ok
```

未完成：

- 未运行真实数据训练，因为当前环境未提供目标 `labels_full` 数据路径。
- 未运行 Qt 可视化交互测试。

## 5. 后续建议

1. 用一小批真实 `labels_full` 跑一次 `seg_inst` 导出，检查 face 数、`seg` 长度、`inst` 矩阵尺寸是否完全一致。
2. 在真实样本上打开 Qt 工具做一次人工验收：实例着色、实例列表、右键编辑、保存、重新加载、版本回放。
3. 用 1-2 个 batch 跑一次 `task_mode=seg_inst` 训练 smoke test，确认 loss、metric、padding mask 都正常。
4. 后续如果需要更强的实例推理效果，再补充连通域过滤、最小实例面数、类别一致性等后处理策略。
