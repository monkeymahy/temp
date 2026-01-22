# AAGNet 微调指南

本指南将帮助您使用 MFCAD++ 预训练的 AAGNet 模型在自己的数据集上进行微调，以实现面特征识别任务。

## 目录
1. [微调流程](#微调流程)
2. [数据集准备](#数据集准备)
3. [运行微调](#运行微调)
4. [参数说明](#参数说明)
5. [示例命令](#示例命令)
6. [注意事项](#注意事项)

## 微调流程

1. **准备预训练模型**：使用 MFCAD++ 数据集训练得到的模型权重
2. **准备目标数据集**：按照指定格式组织您自己的数据集
3. **配置微调参数**：设置学习率、批次大小等参数
4. **运行微调脚本**：执行 `finetune.py` 脚本
5. **评估微调结果**：使用测试集评估模型性能

## 数据集准备

您的数据集需要按照与 MFCAD2 相同的格式组织：

### 目录结构
```
your_dataset/
├── aag/                # 存储图结构数据
│   ├── 1.json          # 第一个样本的图结构
│   ├── 2.json          # 第二个样本的图结构
│   └── ...
├── labels/             # 存储标签数据
│   ├── 1.json          # 第一个样本的标签
│   ├── 2.json          # 第二个样本的标签
│   └── ...
├── train.txt           # 训练集样本ID列表
├── val.txt             # 验证集样本ID列表
└── test.txt            # 测试集样本ID列表
```

### 文件格式说明

#### 1. 图结构文件 (aag/*.json)
每个 JSON 文件包含一个样本的属性邻接图 (AAG) 数据，格式如下：
```json
[
  "sample_id",
  {
    "graph": {
      "edges": [...],
      "nodes": [...]
    }
  }
]
```

#### 2. 标签文件 (labels/*.json)
每个 JSON 文件包含对应样本中每个面的标签，格式为：
```json
[0, 1, 2, 0, ...]  // 每个数字代表对应面的类别ID
```

#### 3. 分割文件 (train.txt, val.txt, test.txt)
每个文件包含对应集合的样本ID，每行一个ID：
```
1
2
3
...
```

## 运行微调

### 前提条件
- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- DGL 1.0+

### 基本命令

```bash
python v2/finetune.py \
    --pretrained_ckpt /path/to/pretrained/model.ckpt \
    --data_root /path/to/your/dataset \
    --num_classes 3 \
    --batch_size 32 \
    --max_epochs 100 \
    --learning_rate 1e-5
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--pretrained_ckpt` | str | 必填 | 预训练模型权重路径 |
| `--data_root` | str | 必填 | 目标数据集根目录 |
| `--num_classes` | int | 必填 | 目标数据集的类别数 |
| `--batch_size` | int | 32 | 训练批次大小 |
| `--num_workers` | int | 4 | 数据加载线程数 |
| `--max_epochs` | int | 100 | 最大训练轮数 |
| `--learning_rate` | float | 1e-5 | 学习率（微调时通常较小） |
| `--save_dir` | str | ./finetune_checkpoints | 检查点保存目录 |
| `--freeze_backbone` | bool | False | 是否冻结主干网络 |
| `--use_3category` | bool | False | 是否使用3类别分类 |
| `--use_9category` | bool | False | 是否使用9类别分类 |

## 示例命令

### 示例 1: 基本微调（3类别）

```bash
python v2/finetune.py \
    --pretrained_ckpt ./checkpoints/mfcad_pretrained.ckpt \
    --data_root ./your_dataset \
    --num_classes 3 \
    --batch_size 16 \
    --max_epochs 50 \
    --learning_rate 5e-6
```

### 示例 2: 冻结主干网络微调

```bash
python v2/finetune.py \
    --pretrained_ckpt ./checkpoints/mfcad_pretrained.ckpt \
    --data_root ./your_dataset \
    --num_classes 3 \
    --batch_size 16 \
    --max_epochs 50 \
    --learning_rate 1e-4 \
    --freeze_backbone
```

## 注意事项

1. **数据集格式**：确保您的数据集严格按照 MFCAD2 的格式组织，特别是图结构和标签文件的格式。

2. **类别映射**：
   - 如果您的数据集只有3个类别，确保标签值为 0, 1, 2
   - 可以根据需要修改 `MFCAD2Dataset` 中的类别映射逻辑

3. **学习率选择**：
   - 微调时学习率通常比从头训练小10-100倍
   - 建议从 1e-5 开始尝试，根据验证集性能调整

4. **批次大小**：
   - 根据您的 GPU 内存调整批次大小
   - 内存不足时可减小批次大小

5. **冻结与解冻**：
   - 对于小数据集，建议冻结主干网络，只训练分类头
   - 对于较大数据集，可以解冻全部层进行微调

6. **预训练模型**：
   - 确保预训练模型是在 MFCAD++ 数据集上训练的
   - 预训练模型应包含完整的网络权重

7. **评估**：
   - 微调完成后，模型会自动在测试集上进行评估
   - 评估指标包括准确率和 IoU（交并比）

## 常见问题

### Q: 运行时出现 "Graph has no edges" 错误
A: 检查您的数据集中是否存在没有边的图结构，这可能是数据生成过程中的问题。

### Q: 微调后模型性能不佳
A: 尝试以下方法：
- 调整学习率（增大或减小）
- 增加训练轮数
- 不冻结主干网络
- 确保数据集标注质量

### Q: 内存不足错误
A: 减小批次大小，或使用 `--num_workers 0` 减少内存使用。