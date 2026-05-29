# Instance Label 功能需求分析

## 1. 目标
为了补齐自有数据的实例标签，已经通过外部数据预训练模型生成初始标签，并希望在自有数据上形成可检查、可修订、可训练、可回溯的 instance label 流程。
1. 在可视化工具中支持 instance label 的检查与修改。
2. 支持带 instance label 的训练和推理，包括数据加载、模型输出、推理后处理和结果保存。
3. 支持完整标签的版本管理，保留人工修订历史。
4. 支持训练时按指定版本导出和加载标签，保证训练实验可复现。

## 2. 术语

- 面语义标签：每个 face 的类别标签，当前训练主任务使用 `labels` 或 `domains.geometry.face` 表达。
- 实例标签：描述哪些 face 属于同一个加工特征实例。推荐同时支持实例 ID 列表和实例对象列表，训练时可派生为邻接矩阵。
- 完整标签：用于检查、编辑、版本管理的富结构标签，建议存放在 `labels_full`。
- 训练标签：从完整标签导出的不可变快照，建议存放在 `labels_train/<export_id>/labels`。
- 版本：完整标签文件内的一次逻辑修改记录，训练导出绑定到具体版本或导出快照。

## 3. 标签数据模型

### 3.1 完整标签格式

当前完整标签主要服务面语义标签，结构中没有可编辑、可回滚的 instance label。修改前可按如下结构理解：

```json
{
  "schema_version": 1,
  "sample_id": "graphs_00000001",
  "labels": [0, 2, 2, 0],
  "labels_base": [0, 1, 1, 0],
  "version_id": 2,
  "versions": [
    {
      "version_id": 2,
      "timestamp": "2026-05-28 10:30:00",
      "author": "user",
      "ops": [
        {
          "type": "face_label_change",
          "indices": [1, 2],
          "old_labels": [1, 1],
          "new_labels": [2, 2]
        }
      ]
    }
  ],
  "domains": {
    "geometry": {
      "face": [0, 2, 2, 0]
    }
  }
}
```

修改后建议将完整标签统一为 JSON 字典，面分割和实例标签共存在同一个样本文件中：

```json
{
  "schema_version": 2,
  "sample_id": "graphs_00000001",
  "labels": [0, 1, 1, 0],
  "labels_base": [0, 1, 1, 0],
  "version_id": 3,
  "versions": [
    {
      "version_id": 2,
      "timestamp": "2026-05-28 10:30:00",
      "author": "user",
      "ops": [
        {
          "type": "face_label_change",
          "indices": [1, 2],
          "old_labels": [2, 2],
          "new_labels": [1, 1]
        }
      ]
    },
    {
      "version_id": 3,
      "timestamp": "2026-05-28 10:40:00",
      "author": "user",
      "ops": [
        {
          "type": "instance_change",
          "indices": [1, 2],
          "old_instance_ids": [-1, -1],
          "new_instance_ids": [0, 0],
          "old_instances": [],
          "new_instances": [
            {
              "instance_id": 0,
              "class_id": 1,
              "face_indices": [1, 2]
            }
          ]
        }
      ]
    }
  ],
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
    }
  }
}
```

针对 instance label 的修改点：

- `domains.geometry.face` 是面语义标签的权威字段，`labels` 保留为兼容字段。
- `labels_base` 是版本回放的基准标签，`labels` 和 `domains.geometry.face` 是当前最新标签。
- `versions` 保存修改历史，不保存完整快照；回滚时从 `labels_base` 开始按 `ops` 依次重放到目标 `version_id`。
- 新增 `domains.instance` 域，用于承载所有 instance label 相关信息。
- 新增 `domains.instance.face_instance`，长度必须等于 face 数；每个位置表示对应 face 所属的 `instance_id`，背景或非特征面使用 `-1`。
- 新增 `domains.instance.instances`，用于保存实例对象列表，便于可视化工具展示和编辑。
- 新增 `instances[*].instance_id`，作为实例稳定 ID；同一样本内必须唯一。
- 新增 `instances[*].class_id`，表示该实例的语义类别；默认可由实例内 face 的语义标签投票得到，人工编辑后以实例对象为准。
- 新增 `instances[*].face_indices`，保存该实例包含的 face 索引列表。
- `face_instance` 与 `instances[*].face_indices` 必须双向一致：某 face 在 `face_instance` 中属于实例 `k`，则它必须出现在 `instance_id = k` 的 `face_indices` 中；反之亦然。
- 修改 face 语义标签时，不自动改变实例归属；修改实例类别时，可选择同步修改实例内 face 的语义标签。
- 保存版本时，除原有面标签变更外，需要记录 instance label 的变更，至少包括被影响 face 的旧/新 `instance_id`，必要时记录旧/新实例对象列表。

### 3.2 训练导出格式

训练导出应根据任务模式生成不同结构：

- `seg_only`：导出当前兼容格式，即纯面标签列表。
- `seg_inst`：导出包含 `seg` 与 `inst` 的字典，其中 `seg` 使用 list 保存每个 face 的语义标签。

训练导出单样本格式：

```json
{
  "seg": [0, 1, 1, 0],
  "inst": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
}
```

导出规则：

- `seg[i]` 表示 face `i` 的语义类别；长度必须等于 face 数。
- `inst[i][j] = 1` 表示 face `i` 和 face `j` 属于同一个实例。
- 同一个实例内对角线应为 `1`；背景面对应行列应为 `0`。
- 若实例只有单面，也应保留该 face 的对角线 `1`，否则训练时会被误认为背景。
- 导出 manifest 必须记录 `sample_id`、完整标签路径、使用版本、导出时间、导出模式、类别映射。

## 4. 可视化工具需求

### 4.1 加载与展示

可视化工具需要在现有 GT/Prediction 双视口基础上增加 instance label 视图能力：

- 加载 `labels_full` 时同时解析 `domains.geometry.face` 和 `domains.instance`。
- GT 视口支持按语义类别着色、按实例 ID 着色两种模式，默认使用实例着色
- Prediction 视口支持显示模型输出的实例结果。
- 实例列表显示 `instance_id`、类别名、面数量。
- 点击实例列表时，高亮该实例包含的所有 face，并进行选中
- 点击 face 时，侧栏显示该 face 的语义标签、实例 ID、所属实例信息。

### 4.2 编辑操作

首版需要支持以下编辑：

- 新建实例：选择一组 face，创建新 `instance_id`。
- 添加到实例：将选中 face 加入已有实例。
- 从实例移除：将选中 face 的 `face_instance` 置为 `-1`，并从实例对象移除。
- 合并实例：将多个实例合并为一个实例。
- 拆分实例：将选中 face 从原实例拆出为新实例。
- 修改实例类别：修改实例 `class_id`，并可选择同步修改实例内 face 的语义标签。
- 修改 face 语义标签：保留当前已有能力，同时检查是否导致实例内类别不一致。

编辑约束：

- 每个 face 同一时刻最多属于一个实例。
- 背景 face 的实例 ID 应为 `-1`。
- 实例对象不允许为空；如果移除后为空，应删除该实例。
- 保存前必须校验 `face_instance` 与 `instances` 一致。
- 修改后自动更新颜色、列表、脏状态和版本预览状态。

### 4.3 保存与版本

保存策略沿用完整标签内版本记录：

- 单次保存可聚合本次会话内的多次实例编辑，形成一个版本记录。
- 版本记录需要支持两类 op：`face_label_change` 和 `instance_change`。
- 支持查看任意历史版本的语义标签与实例标签。
- 支持回滚到任意历史版本并保存为新版本。
- 支持从预测结果保存为完整标签初始版本。

## 5. 训练需求

### 5.1 数据加载

v2 数据集加载层需要支持任务模式：

- `task_mode: seg_only`：保持当前 `graph.ndata["y"]` 行为。
- `task_mode: seg_inst`：额外返回 `inst_y` 或 batch 后的 `inst_labels`。

配置建议：

```yaml
data:
  root_dir: C:\Data\SF-JSON
  label_dir: labels_train\20260528_100000
  task_mode: seg_inst
  label_format: train_export
```

加载要求：

- 样本 face 数必须与 `seg`、`face_instance` 或 `inst` 一致。
- 支持从训练导出的 `inst` 邻接矩阵直接读取。
- 支持从 `face_instance` 动态生成 `inst` 邻接矩阵。
- batch 时按当前最大 face 数 padding 成 `[B, N, N]`，并提供有效 face mask，避免 padding 参与指标。
- `seg_only` 训练不得强依赖实例字段。

### 5.2 模型与损失

需要在 v2 模型中引入可配置多任务输出：

- `enable_seg_head: true`
- `enable_inst_head: true`

损失建议：

- 面语义：`CrossEntropyLoss`
- 实例邻接：`BCEWithLogitsLoss`，需要 mask 掉 padding 区域。

训练指标：

- `seg_acc`、`seg_iou`
- `inst_binary_acc`、`inst_f1`
- 可选 `instance_cluster_f1`，用于评估后处理后的实例级识别质量。

### 5.3 训练导出

训练前必须显式导出训练快照：

```bash
python v2/utils/export_train_labels.py \
  --labels-full-dir C:\Data\SF-JSON\labels_full \
  --export-id 20260528_seg_inst_v1 \
  --task-mode seg_inst \
  --output-root C:\Data\SF-JSON\labels_train\20260528_seg_inst_v1
```

导出 manifest 必须写入：

- `export_id`
- `task_mode`
- `labels_full_dir`
- `sample_id`
- `version_id`
- `label_path`
- `class_mapping`
- `generated_at`

训练配置不应直接指向会继续编辑的 `labels_full`，而应指向某个 `labels_train/<export_id>` 快照。

## 6. 推理需求

推理分为面分割推理和实例推理：

- `seg_only` 模型保持当前输出每面类别。
- `seg_inst` 模型输出 `seg_logits` 和 `inst_logits`。
- `inst_logits` 后处理为实例集合，保存为 `face_instance` 和 `instances`。
- 支持将推理结果保存为 `labels_full` 初始标签。

后处理最低要求：

- 对 `sigmoid(inst_logits)` 使用阈值生成邻接关系。
- 对邻接关系做连通分量聚类。
- 对每个实例内 face 的 `seg_logits` 投票得到 `class_id`。
- 过滤背景类实例。
- 单面实例按阈值和语义类别保留，不能默认丢弃。

## 7. 版本管理需求

完整标签版本管理需要满足：

- 每个样本独立维护版本。
- `version_id` 单调递增。
- `labels_base` 和 `domains.instance.base` 或等价字段能重建任意历史版本。
- 每个版本记录包含作者、时间、操作类型、变更摘要。
- 版本记录支持从旧格式迁移：纯 list 标签可迁移为 v1 完整标签，实例字段为空。
- 导出训练快照时可以选择最新版本或 manifest 中指定版本。
- 已导出的训练快照不可被后续编辑覆盖。

扩展 version op：

```json
{
  "version_id": 4,
  "timestamp": "2026-05-28 10:30:00",
  "author": "user",
  "ops": [
    {
      "type": "instance_change",
      "indices": [1, 2, 5],
      "old_instance_ids": [0, 0, -1],
      "new_instance_ids": [1, 1, 1],
      "old_instances": [],
      "new_instances": []
    }
  ]
}
```

## 8. 校验需求

保存、导出、训练加载前都应执行一致性校验：

- face 标签长度等于 AAG 节点数。
- `face_instance` 长度等于 AAG 节点数。
- `face_instance` 中非 `-1` 的 ID 必须存在于 `instances`。
- `instances[*].face_indices` 必须与 `face_instance` 一致。
- 同一 face 不能出现在多个实例中。
- `inst` 矩阵必须是方阵、对称矩阵，尺寸等于 face 数。
- 背景 face 对应 `inst` 行列应全 0。
- 非背景实例至少包含一个 face。
- 训练导出后的 manifest 中样本数量与 split 文件交集符合预期。
