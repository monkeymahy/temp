# Instance Label 开发方案

## 1. 开发目标

基于 `docs/instance_label_requirements.md`，本方案将 instance label 能力拆成四条主线：

1. 标签数据结构与版本管理：完整标签支持 `domains.instance`，并能保存、回滚 instance 修改历史。
2. 训练标签导出与加载：从 `labels_full` 导出 `seg_inst` 训练快照，训练时加载 `seg` 和 `inst`。
3. 模型训练与推理：v2 模型支持可选实例头，训练时计算 masked instance loss，推理时输出实例集合。
4. 可视化检查与编辑：可视化工具支持实例着色、实例列表、实例编辑和版本保存。

优先保证现有 `seg_only` 训练、可视化和标签导出不受影响；instance 能力通过显式配置开启。

## 2. 总体设计

### 2.1 标签流转

推荐流程：

```text
labels_full/*.json
  -> 可视化检查/编辑/版本保存
  -> export_train_labels.py --task-mode seg_inst
  -> labels_train/<export_id>/labels/*.json
  -> SFDataset(task_mode=seg_inst)
  -> AAGNetSegmentor(enable_inst_head=true)
```

核心约束：

- `labels_full` 是可编辑、可回滚的完整标签。
- `labels_train/<export_id>` 是训练快照，不在训练过程中修改。
- 训练配置只指向 `labels_train/<export_id>`，不直接消费正在编辑的 `labels_full`。

### 2.2 instance 表达

完整标签使用两种互补表达：

- `face_instance`: 每个 face 对应一个 `instance_id`，背景为 `-1`。
- `instances`: 实例对象列表，用于 UI 展示和编辑。

训练标签使用一种矩阵表达：

- `inst`: `N x N` 同实例邻接矩阵，`inst[i][j] = 1` 表示 face `i` 和 face `j` 同实例。

实现时必须提供双向转换：

- `face_instance + instances -> inst`
- `inst + seg list -> face_instance + instances`

## 3. 模块拆分

### 3.1 标签工具层

目标文件：

- `v2/utils/data_utils.py`
- 新增可选：`v2/utils/instance_label_utils.py`
- 新增可选：`v2/utils/validate_instance_labels.py`

建议新增函数：

```python
def extract_instance_from_payload(payload: Any) -> Optional[dict]:
    ...

def normalize_instance_payload(payload: dict, num_faces: Optional[int] = None) -> dict:
    ...

def face_instance_to_adj(face_instance: list[int]) -> list[list[int]]:
    ...

def adj_to_face_instance(inst_adj: list[list[int]], seg_labels: list[int], background_labels: set[int]) -> dict:
    ...

def validate_instance_payload(payload: dict, num_faces: int) -> list[str]:
    ...

def rollback_instance_payload(payload: dict, target_version_id: int) -> dict:
    ...

def append_instance_version(payload: dict, author: str, ops: list[dict], timestamp: str) -> dict:
    ...
```

实现要点：

- `normalize_label_payload` 继续负责面标签兼容；新增 instance 解析不要破坏旧格式。
- `validate_instance_payload` 返回错误列表，调用方决定报错或 UI 提示。
- `rollback_instance_payload` 与现有 `rollback_payload` 类似，但回放 `instance_change`。
- 如果旧标签没有 `domains.instance`，标准化时生成空实例结构：

```json
{
  "face_instance": [-1, -1, ...],
  "instances": []
}
```

验收：

- 纯 list 标签、当前版本化标签、带 instance 的完整标签都能被标准化。
- `face_instance -> inst -> face_instance` 在合法输入上保持一致。
- 非法标签能给出明确错误，例如实例 ID 缺失、face 重复、矩阵非对称。

### 3.2 训练导出

目标文件：

- `v2/utils/export_train_labels.py`

新增 CLI 参数：

```bash
--task-mode seg_only|seg_inst
--class-mapping path/to/mapping.json
--background-labels 0
```

导出策略：

- `seg_only`：保持当前行为，导出纯 face label list。
- `seg_inst`：导出字典，其中 `seg` 使用 list 保存每个 face 的语义标签：

```json
{
  "seg": [0, 1, 1, 0],
  "inst": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
}
```

manifest 增加：

- `task_mode`
- `labels_full_dir`
- `class_mapping`
- `background_labels`
- `items[*].version_id`
- `items[*].label_path`
- `items[*].full_label_path`

实现要点：

- `--use-manifest-versions` 同时作用于面标签和实例标签。
- 如果选择 `seg_inst` 但样本没有 instance，应直接报错，不静默导出空矩阵。
- 导出前先跑一致性校验。

验收：

- 使用当前 `seg_only` 配置导出结果与旧版本一致。
- 使用 `seg_inst` 能生成 `seg + inst` 文件和 manifest。
- 训练快照导出后再次修改 `labels_full` 不影响已导出的文件。

### 3.3 数据加载

目标文件：

- `v2/dataset/SFdataset.py`
- `v2/dataset/SFdatamodule.py`

新增配置：

```yaml
data:
  task_mode: seg_inst
```

`SFDataset` 行为：

- `seg_only`：
  - 读取纯 list 或完整标签中的 face label。
  - 写入 `graph.ndata["y"]`。
- `seg_inst`：
  - 读取训练导出的 `seg` 和 `inst`。
  - `seg` 按 list 读取，直接转为 `graph.ndata["y"]`。
  - 写入 `graph.ndata["y"]`。
  - 样本字典增加 `inst_y`。

`SFDataModule._collate` 行为：

- 始终返回 `{"graph": batched_graph, "filename": ...}`。
- `seg_inst` 时额外返回：

```python
{
  "inst_labels": padded_adj,      # [B, max_faces, max_faces]
  "inst_mask": padded_mask,       # [B, max_faces, max_faces]
  "num_faces": num_faces_tensor
}
```

实现要点：

- padding 只在 batch 内按最大 face 数做。
- `inst_mask` 用于 loss 和指标过滤 padding。
- 单样本 face 数必须与 `seg`、`inst` 尺寸一致。

验收：

- `task_mode=seg_only` 原训练流程可跑通。
- `task_mode=seg_inst` dataloader 输出形状正确。
- batch 中不同 face 数样本能正确 padding 和 mask。

### 3.4 模型训练

目标文件：

- `v2/models/segmentors.py`

新增模型参数：

```yaml
model:
  enable_inst_head: true
  inst_loss_weight: 1.0
```

结构调整：

- 将 `forward` 拆成两步：
  - `_encode_graph(...) -> local_global_feat`
  - `forward(...) -> dict`
- `seg_head` 永远保留。
- `enable_inst_head=true` 时新增 `inst_head`，复用旧版 `models/inst_segmentors.py` 中 `InnerProductDecoder` 的思路。

推荐输出：

```python
{
  "seg_logits": seg_out,
  "inst_logits": inst_out,
}
```

损失：

```python
loss_seg = CrossEntropyLoss(seg_logits, seg_label)
loss_inst = BCEWithLogitsLoss(reduction="none")(inst_logits, inst_labels)
loss_inst = (loss_inst * inst_mask).sum() / inst_mask.sum().clamp_min(1)
loss = loss_seg + inst_loss_weight * loss_inst
```

指标：

- `seg_acc`
- `seg_iou`
- `inst_binary_acc`
- `inst_f1`

实现要点：

- 为了兼容旧代码，可让 `enable_inst_head=false` 时 `forward` 返回 `seg_logits`，或统一返回 dict 后同步改训练/验证/测试步骤。
- 建议统一返回 dict，降低后续推理分支复杂度。
- Lightning checkpoint 参数变化后，要注意老 checkpoint 的加载兼容；可视化工具加载 seg-only 模型时应关闭实例头。

验收：

- seg-only 配置训练无回归。
- seg-inst 配置能完成至少 1 个 epoch 的 train/val。
- `inst_mask` 覆盖后，padding 区域不影响 loss 和指标。

### 3.5 推理与后处理

目标文件：

- `v2/utils/qt5_visualization.py`
- 新增可选：`v2/utils/infer_instance.py`
- 新增可选：`v2/utils/instance_postprocess.py`

后处理函数：

```python
def postprocess_instance_logits(
    inst_logits,
    seg_logits,
    threshold: float,
    background_labels: set[int],
) -> dict:
    ...
```

输出：

```json
{
  "face_instance": [-1, 0, 0, -1],
  "instances": [
    {
      "instance_id": 0,
      "class_id": 1,
      "face_indices": [1, 2]
    }
  ]
}
```

后处理步骤：

1. `sigmoid(inst_logits) >= threshold` 得到邻接关系。
2. 强制对称化：`adj = adj | adj.T`。
3. 基于邻接矩阵做连通分量。
4. 每个连通分量内对 `seg_logits` 投票得到 `class_id`。
5. 背景类过滤。
6. 生成连续 `instance_id`。

验收：

- 单文件推理能生成 face label 和 instance label。
- 单面实例不会被默认丢弃。
- 推理结果能写入 `labels_full` 的 `domains.instance`。

### 3.6 可视化工具

目标文件：

- `v2/utils/qt5_visualization.py`

新增状态：

```python
self.current_gt_instances
self.current_gt_face_instance
self.current_pred_instances
self.current_pred_face_instance
self.instance_color_mode
self.selected_instance_ids_gt
self.selected_instance_ids_pred
```

新增 UI：

- 着色模式下拉：`语义类别` / `实例 ID`。
- GT 实例列表：显示 `instance_id`、类别名、面数量。
- Prediction 实例列表：同上。
- 实例右键菜单：新建、加入、移除、合并、拆分、修改类别。

编辑操作实现：

- 新建实例：
  - 找到当前最大 `instance_id + 1`。
  - 选中 face 写入该 ID。
  - 添加实例对象。
- 添加到实例：
  - 选中 face 原归属先移除。
  - 写入目标实例。
- 从实例移除：
  - 选中 face 写为 `-1`。
  - 实例为空则删除。
- 合并实例：
  - 目标 ID 保留，其他 ID 的 face 改为目标 ID。
- 拆分实例：
  - 选中 face 从旧实例拆出为新 ID。
- 修改实例类别：
  - 更新 `instances[*].class_id`。
  - 可选同步修改实例内 face 的语义标签。

版本保存：

- 面标签编辑继续生成 `face_label_change`。
- 实例编辑生成 `instance_change`。
- 保存前统一调用校验。

验收：

- 加载已有 instance 标签后默认按实例着色。
- 点击实例列表能高亮并选中该实例 face。
- 编辑后保存，重新打开仍一致。
- 版本列表可预览历史面标签和实例标签。

## 4. 开发顺序

### 阶段 1：标签工具和导出

改动：

- 新增 instance 标签解析、转换、校验工具。
- 扩展 `export_train_labels.py` 支持 `seg_inst`。

验收命令示例：

```bash
python v2/utils/export_train_labels.py \
  --labels-full-dir C:\Data\SF-JSON\labels_full \
  --export-id dev_seg_inst \
  --task-mode seg_inst \
  --output-root C:\Data\SF-JSON\labels_train\dev_seg_inst
```

### 阶段 2：数据加载

改动：

- `SFDataset` 增加 `task_mode`。
- `SFDataModule` collate 支持 instance padding 和 mask。

验收：

- 写一个小脚本加载 2 个不同 face 数样本，检查 `inst_labels`、`inst_mask` 形状。

### 阶段 3：模型实例头

改动：

- `AAGNetSegmentor` 增加实例头。
- training/validation/test step 增加 instance loss 和指标。

验收：

- `seg_only` 配置可训练。
- `seg_inst` 配置可训练，loss 不为 NaN。

### 阶段 4：可视化读写和编辑

改动：

- 可视化解析 `domains.instance`。
- 增加实例着色、实例列表、基本编辑。
- 保存 instance 版本记录。

验收：

- 打开样本，编辑实例，保存，重新打开和回滚均正常。

### 阶段 5：推理后处理

改动：

- 模型推理返回 `seg_logits + inst_logits`。
- 后处理生成 `face_instance + instances`。
- 可视化保存预测结果为完整标签。

验收：

- 单 STEP 推理后可直接进入可视化检查/编辑流程。

## 5. 测试计划

单元测试：

- `face_instance_to_adj`
- `adj_to_face_instance`
- `validate_instance_payload`
- `rollback_instance_payload`
- `export_train_labels --task-mode seg_inst`

集成测试：

- `labels_full -> labels_train -> SFDataset`。
- `SFDataset -> AAGNetSegmentor.training_step`。
- `qt5_visualization` 加载、编辑、保存、回滚。

回归测试：

- 现有 `seg_only` 数据加载不变。
- 现有 `labels_train` 最新目录解析不变。
- 现有可视化面标签编辑不变。

## 6. 风险与处理

- 老 checkpoint 与新增实例头不兼容：配置中显式关闭 `enable_inst_head`，可视化加载时按配置初始化。
- instance 版本回滚比 face label 更复杂：首版用 `face_instance` 作为回滚主状态，`instances` 由 `face_instance + class_id` 重建或随 op 同步记录。
- 同类相邻实例后处理容易合并：首版采用阈值连通分量，后续再补更强聚类策略。
- UI 编辑操作多且容易状态不同步：所有编辑最终都走统一的 `set_face_instance(...)` / `rebuild_instances(...)` / `validate(...)` 流程。

## 7. 推荐首个实现切片

第一轮先完成最小闭环：

1. 支持读取完整标签中的 `domains.instance`。
2. 支持导出 `seg_inst` 训练标签。
3. 支持 `SFDataset(task_mode=seg_inst)` 加载 `inst_labels`。
4. 支持可视化按实例着色和实例列表查看。

这个切片完成后，即使暂时不训练实例头，也可以先开始检查和修订 instance label。
