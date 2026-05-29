import torch
from torch import nn
import lightning.pytorch as L
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassJaccardIndex,
)

import models.encoders as encoders
from .layers import MLP


class InnerProductDecoder(nn.Module):
    """按图拆分节点特征后计算每个样本内部的同实例 logits。"""

    def __init__(self, wq: nn.Module, wk: nn.Module) -> None:
        super().__init__()
        self.wq = wq
        self.wk = wk

    def forward(self, batched_graph, batched_h):
        batch_num_nodes = batched_graph.batch_num_nodes().tolist()
        hidden_list = torch.split(batched_h, batch_num_nodes, dim=0)
        padded_hidden = pad_sequence(hidden_list, batch_first=True)
        q = self.wq(padded_hidden)
        k = self.wk(padded_hidden)
        return torch.bmm(q, k.transpose(1, 2))


###############################################################################
# Segmentation model
# AAG-Net solid face segmentation model for B-rep (Boundary Representation) data
###############################################################################


class AAGNetSegmentor(L.LightningModule):
    """
    AAG-Net solid face segmentation model
    基于 Attributed Adjacency Graph (AAG) 的实体面分割模型，用于 B-rep 数据的面分类任务

    模型结构：
    1. 节点(面)属性编码器：将输入的面属性映射到嵌入空间
    2. 节点(面)网格编码器：处理2D UV网格特征
    3. 边属性编码器：将输入的边属性映射到嵌入空间
    4. 图编码器：通过消息传递计算节点(面)嵌入和全局图嵌入
    5. 分割头：将节点嵌入映射到分类概率

    训练流程：
    1. 前向传播：计算每个面的分类logits
    2. 损失计算：使用交叉熵损失函数
    3. 指标计算：计算准确率和IOU
    4. 优化器更新：使用AdamW优化器和余弦退火学习率调度器
    """

    def __init__(
        self,
        num_classes,
        arch,
        edge_attr_dim,
        node_attr_dim,
        edge_attr_emb,
        node_attr_emb,
        edge_grid_dim,
        node_grid_dim,
        edge_grid_emb,
        node_grid_emb,
        num_layers,
        delta,
        mlp_ratio=4,
        drop=0.0,
        drop_path=0.0,
        head_hidden_dim=256,
        conv_on_edge=True,
        use_uv_gird=True,
        use_edge_attr=True,
        use_face_attr=True,
        lr=1e-4,
        weight_decay=0,
        n_epochs=200,
        enable_inst_head=False,
        inst_loss_weight=1.0,
    ):
        """
        Initialize the AAG-Net solid face segmentation model

        Args:
            num_classes (int): 每个面要输出的类别数量
            arch (str): 图编码器的架构名称
            edge_attr_dim (int): 边属性的输入维度
            node_attr_dim (int): 节点属性的输入维度
            edge_attr_emb (int): 边属性的嵌入维度
            node_attr_emb (int): 节点属性的嵌入维度
            edge_grid_dim (int): 边UV网格的输入通道数
            node_grid_dim (int): 节点UV网格的输入通道数
            edge_grid_emb (int): 边UV网格的嵌入维度
            node_grid_emb (int): 节点UV网格的嵌入维度
            num_layers (int): 图编码器的层数
            delta (float): 图编码器中的delta参数
            mlp_ratio (int, optional): MLP扩展比例. Defaults to 4.
            drop (float, optional): Dropout率. Defaults to 0.0.
            drop_path (float, optional): DropPath率. Defaults to 0.0.
            head_hidden_dim (int, optional): 分类头的隐藏层维度. Defaults to 256.
            conv_on_edge (bool, optional): 是否在边上执行卷积操作. Defaults to True.
            use_uv_gird (bool, optional): 是否使用UV网格特征. Defaults to True.
            use_edge_attr (bool, optional): 是否使用边属性. Defaults to True.
            use_face_attr (bool, optional): 是否使用面属性. Defaults to True.
            lr (float, optional): 学习率. Defaults to 1e-4.
            weight_decay (float, optional): 权重衰减. Defaults to 0.
            n_epochs (int, optional): 训练轮数. Defaults to 200.
        """
        super().__init__()
        # 模型配置参数
        self.use_uv_gird = use_uv_gird
        self.use_edge_attr = use_edge_attr
        self.use_face_attr = use_face_attr
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.enable_inst_head = enable_inst_head
        self.inst_loss_weight = float(inst_loss_weight)

        # 损失函数：使用交叉熵损失函数
        # 适用于多分类任务，自动处理类别不平衡问题
        self.seg_loss = nn.CrossEntropyLoss()
        self.inst_loss = nn.BCEWithLogitsLoss(reduction="none")

        # 节点(面)属性编码器：将输入的面属性映射到嵌入空间
        # 包含线性层和层归一化，用于提取面属性的特征表示
        self.node_attr_encoder = nn.Sequential(
            nn.Linear(node_attr_dim, node_attr_emb),  # 线性变换到嵌入维度
            nn.LayerNorm(node_attr_emb),  # 层归一化，加速训练和提高稳定性
        )

        # 节点(面)网格编码器：处理2D UV网格特征
        # 当node_grid_dim > 0时，使用卷积神经网络提取网格特征
        if node_grid_dim:
            self.node_grid_encoder = nn.Sequential(
                # 第一层卷积：输入通道数 -> 输出通道数(node_grid_emb // 4)
                nn.Conv2d(
                    node_grid_dim,
                    node_grid_emb // 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(node_grid_emb // 4),  # 批归一化
                nn.Mish(),  # Mish激活函数，比ReLU有更好的性能
                # 第二层卷积：通道数翻倍
                nn.Conv2d(
                    node_grid_emb // 4,
                    node_grid_emb // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(node_grid_emb // 2),  # 批归一化
                nn.Mish(),  # Mish激活函数
                # 第三层卷积：通道数翻倍到目标嵌入维度
                nn.Conv2d(
                    node_grid_emb // 2,
                    node_grid_emb,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(node_grid_emb),  # 批归一化
                nn.Mish(),  # Mish激活函数
                nn.AdaptiveAvgPool2d(1),  # 全局平均池化，将特征图转为向量
                nn.Flatten(1),  # 展平特征，去除空间维度
            )
        else:
            # 当node_grid_dim为0时，不使用网格编码器
            self.node_grid_encoder = None

        # 边属性编码器：将输入的边属性映射到嵌入空间
        # 包含线性层和层归一化，用于提取边属性的特征表示
        self.edge_attr_encoder = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_attr_emb),  # 线性变换到嵌入维度
            nn.LayerNorm(edge_attr_emb),  # 层归一化
        )

        # 边网格编码器：处理1D UV网格特征（预留功能，尚未实现）
        if edge_grid_dim:
            # TODO: 实现边网格编码器
            pass

        # 计算最终的节点和边嵌入维度
        # 节点嵌入 = 属性嵌入 + 网格嵌入
        node_emb = node_attr_emb + node_grid_emb
        # 边嵌入 = 属性嵌入 + 网格嵌入
        edge_emb = edge_attr_emb + edge_grid_emb

        # 图编码器：基于指定架构初始化图神经网络
        # 通过动态获取编码器类，实现不同图编码器的灵活切换
        encoder = getattr(encoders, arch)  # 动态获取编码器类
        self.graph_encoder = encoder(
            node_dim=node_emb,  # 节点嵌入维度
            edge_dim=edge_emb,  # 边嵌入维度
            num_layers=num_layers,  # 编码器层数
            delta=delta,  # delta参数
            mlp_ratio=mlp_ratio,  # MLP扩展比例
            drop=drop,  # Dropout率
            drop_path=drop_path,  # DropPath率
            conv_on_edge=conv_on_edge,  # 是否在边上执行卷积
        )

        # 计算最终输出嵌入维度（节点嵌入 + 全局图嵌入）
        # 将全局图嵌入与每个节点的局部嵌入拼接，提供更丰富的上下文信息
        final_out_emb = 2 * node_emb

        # 分割头：将节点嵌入映射到分类概率
        # 使用多层感知机(MLP)实现，包含隐藏层和输出层
        self.seg_head = MLP(
            num_layers=2,  # MLP层数
            input_dim=final_out_emb,  # 输入维度
            hidden_dim=head_hidden_dim,  # 隐藏层维度
            output_dim=num_classes,  # 输出维度（类别数）
            norm=nn.LayerNorm,  # 归一化层
            act=nn.Mish,  # 激活函数
        )

        self.inst_head = None
        if self.enable_inst_head:
            wq = MLP(
                num_layers=2,
                input_dim=final_out_emb,
                hidden_dim=head_hidden_dim,
                output_dim=head_hidden_dim,
                norm=nn.LayerNorm,
                last_norm=True,
                act=nn.Mish,
            )
            wk = MLP(
                num_layers=2,
                input_dim=final_out_emb,
                hidden_dim=head_hidden_dim,
                output_dim=head_hidden_dim,
                norm=nn.LayerNorm,
                last_norm=True,
                act=nn.Mish,
            )
            self.inst_head = InnerProductDecoder(wq, wk)

        # 初始化评估指标
        # 为训练集、验证集和测试集分别初始化准确率和IOU指标
        self._init_metrics(num_classes=num_classes)

    def _init_metrics(self, num_classes):
        """
        初始化各类评估指标
        为训练集、验证集和测试集分别初始化准确率和IOU指标

        Args:
            num_classes (int): 类别数量，用于初始化多分类评估指标
        """
        # 训练集指标
        self.tra_seg_acc = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",  # 宏平均：计算所有类的指标并取平均
        )
        self.tra_seg_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",  # 宏平均：计算所有类的IOU并取平均
        )
        self.tra_seg_acc_per_class = MulticlassAccuracy(
            num_classes=num_classes,
            average=None,  # None：返回每个类的指标
        )
        self.tra_seg_iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes,
            average=None,  # None：返回每个类的IOU
        )

        # 验证集指标
        self.val_seg_acc = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",  # 宏平均
        )
        self.val_seg_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",  # 宏平均
        )
        self.val_seg_acc_per_class = MulticlassAccuracy(
            num_classes=num_classes,
            average=None,  # 返回每个类的指标
        )
        self.val_seg_iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes,
            average=None,  # 返回每个类的IOU
        )

        # 测试集指标
        self.tst_seg_acc = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",  # 宏平均
        )
        self.tst_seg_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",  # 宏平均
        )
        self.tst_seg_acc_per_class = MulticlassAccuracy(
            num_classes=num_classes,
            average=None,  # 返回每个类的指标
        )
        self.tst_seg_iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes,
            average=None,  # 返回每个类的IOU
        )
        if self.enable_inst_head:
            self.tra_inst_acc = BinaryAccuracy()
            self.tra_inst_f1 = BinaryF1Score()
            self.val_inst_acc = BinaryAccuracy()
            self.val_inst_f1 = BinaryF1Score()
            self.tst_inst_acc = BinaryAccuracy()
            self.tst_inst_f1 = BinaryF1Score()

    def _encode_graph(self, batched_graph):
        """编码图并返回每个节点拼接全局上下文后的特征。"""
        # 获取输入特征
        # 根据配置决定是否使用面属性
        input_node_attr = (
            batched_graph.ndata["x"]
            if self.use_face_attr
            else torch.zeros_like(batched_graph.ndata["x"])  # 如果不使用面属性，则置为0
        )
        # 根据配置决定是否使用UV网格
        input_node_grid = (
            batched_graph.ndata["grid"]
            if self.use_uv_gird
            else torch.zeros_like(batched_graph.ndata["grid"])  # 如果不使用UV网格，则置为0
        )
        # 根据配置决定是否使用边属性
        input_edge_attr = (
            batched_graph.edata["x"]
            if self.use_edge_attr
            else torch.zeros_like(batched_graph.edata["x"])  # 如果不使用边属性，则置为0
        )

        # 计算节点(面)隐藏特征
        # 通过属性编码器处理面属性
        node_feat = self.node_attr_encoder(input_node_attr)
        # 如果使用网格编码器，则处理网格特征并与属性特征拼接
        if self.node_grid_encoder:
            assert input_node_grid.numel() > 0, "输入网格特征不能为空"
            node_grid_feat = self.node_grid_encoder(input_node_grid)
            node_feat = torch.concat([node_feat, node_grid_feat], dim=1)  # 拼接属性特征和网格特征

        # 计算边隐藏特征
        # 通过边属性编码器处理边属性
        edge_feat = self.edge_attr_encoder(input_edge_attr)
        # Message pass and compute per-face(node) and global embeddings
        node_emb, graph_emb = self.graph_encoder(batched_graph, node_feat, edge_feat)
        # concatenated to the per-node embeddings
        num_nodes_per_graph = batched_graph.batch_num_nodes()
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0)

        # 拼接局部节点嵌入和全局图嵌入
        # 结合局部特征和全局上下文，提高分类性能
        return torch.cat((node_emb, graph_emb), dim=1)

    def _forward_outputs(self, batched_graph):
        local_global_feat = self._encode_graph(batched_graph)

        # 映射到分类logits
        # 通过分割头将嵌入向量映射到类别概率
        seg_out = self.seg_head(local_global_feat)
        outputs = {"seg_logits": seg_out}
        if self.enable_inst_head and self.inst_head is not None:
            outputs["inst_logits"] = self.inst_head(batched_graph, local_global_feat)
        return outputs

    def forward(self, batched_graph):
        """
        Forward pass 前向传播。默认保持旧行为：seg-only 模式返回分割logits；
        开启实例头时返回包含 seg_logits / inst_logits 的字典。
        """
        outputs = self._forward_outputs(batched_graph)
        if self.enable_inst_head:
            return outputs
        return outputs["seg_logits"]

    def _compute_step_loss_and_logs(self, batch: dict, stage: str):
        graphs = batch["graph"]
        seg_label = graphs.ndata["y"]
        outputs = self._forward_outputs(batched_graph=graphs)
        seg_pred = outputs["seg_logits"]
        loss_seg = self.seg_loss(seg_pred, seg_label)
        loss = loss_seg

        metric_prefix = {
            "tra": (self.tra_seg_acc, self.tra_seg_iou, self.tra_seg_acc_per_class, self.tra_seg_iou_per_class),
            "val": (self.val_seg_acc, self.val_seg_iou, self.val_seg_acc_per_class, self.val_seg_iou_per_class),
            "tst": (self.tst_seg_acc, self.tst_seg_iou, self.tst_seg_acc_per_class, self.tst_seg_iou_per_class),
        }
        seg_acc, seg_iou, seg_acc_per_class_metric, seg_iou_per_class_metric = metric_prefix[stage]
        seg_acc(seg_pred, seg_label)
        seg_iou(seg_pred, seg_label)
        seg_acc_per_class = seg_acc_per_class_metric(seg_pred, seg_label)
        seg_iou_per_class = seg_iou_per_class_metric(seg_pred, seg_label)

        log_dict = {
            f"{stage}_loss": loss.item(),
            f"{stage}_seg_loss": loss_seg.item(),
            f"{stage}_seg_acc_avg": seg_acc,
            f"{stage}_seg_iou_avg": seg_iou,
        }

        if self.enable_inst_head:
            if "inst_logits" not in outputs:
                raise ValueError("enable_inst_head=True but model did not return inst_logits")
            if "inst_labels" not in batch or "inst_mask" not in batch:
                raise ValueError("seg_inst training requires inst_labels and inst_mask in batch")
            inst_logits = outputs["inst_logits"]
            inst_labels = batch["inst_labels"].to(device=inst_logits.device, dtype=inst_logits.dtype)
            inst_mask = batch["inst_mask"].to(device=inst_logits.device, dtype=inst_logits.dtype)
            if inst_logits.shape != inst_labels.shape:
                raise ValueError(
                    f"inst logits shape {tuple(inst_logits.shape)} != labels {tuple(inst_labels.shape)}"
                )
            loss_inst_raw = self.inst_loss(inst_logits, inst_labels)
            loss_inst = (loss_inst_raw * inst_mask).sum() / inst_mask.sum().clamp_min(1.0)
            loss = loss + self.inst_loss_weight * loss_inst
            log_dict[f"{stage}_loss"] = loss.item()
            log_dict[f"{stage}_inst_loss"] = loss_inst.item()

            valid = inst_mask > 0
            inst_metric_map = {
                "tra": (self.tra_inst_acc, self.tra_inst_f1),
                "val": (self.val_inst_acc, self.val_inst_f1),
                "tst": (self.tst_inst_acc, self.tst_inst_f1),
            }
            inst_acc, inst_f1 = inst_metric_map[stage]
            inst_acc(inst_logits[valid], inst_labels[valid].long())
            inst_f1(inst_logits[valid], inst_labels[valid].long())
            log_dict[f"{stage}_inst_acc"] = inst_acc
            log_dict[f"{stage}_inst_f1"] = inst_f1

        return loss, seg_label, seg_acc_per_class, seg_iou_per_class, log_dict

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        """
        训练步骤

        Args:
            batch (dict): 包含图数据的批次
            batch_idx (int): 批次索引

        Returns:
            torch.tensor: 损失值
        """
        loss, seg_label, seg_acc_per_class, seg_iou_per_class, _dic = self._compute_step_loss_and_logs(
            batch=batch,
            stage="tra",
        )
        LABEL_NAMES = self.trainer.train_dataloader.dataset.label_names
        for i, (_acc, _iou) in enumerate(zip(seg_acc_per_class, seg_iou_per_class)):
            _dic[f"tra_seg_acc{i}({LABEL_NAMES[i]})"] = _acc
            _dic[f"tra_seg_iou{i}({LABEL_NAMES[i]})"] = _iou

        # 记录指标
        self.log_dict(_dic, on_step=False, on_epoch=True, batch_size=seg_label.numel())

        return loss

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        """
        验证步骤

        Args:
            batch (dict): 包含图数据的批次
            batch_idx (int): 批次索引
        """
        loss, seg_label, seg_acc_per_class, seg_iou_per_class, _dic = self._compute_step_loss_and_logs(
            batch=batch,
            stage="val",
        )
        LABEL_NAMES = self.trainer.val_dataloaders.dataset.label_names
        for i, (_acc, _iou) in enumerate(zip(seg_acc_per_class, seg_iou_per_class)):
            _dic[f"val_seg_acc{i}({LABEL_NAMES[i]})"] = _acc
            _dic[f"val_seg_iou{i}({LABEL_NAMES[i]})"] = _iou

        # 记录指标
        self.log_dict(_dic, on_step=False, on_epoch=True, batch_size=seg_label.numel())

    def test_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        """
        测试步骤

        Args:
            batch (dict): 包含图数据的批次
            batch_idx (int): 批次索引
        """
        loss, seg_label, seg_acc_per_class, seg_iou_per_class, _dic = self._compute_step_loss_and_logs(
            batch=batch,
            stage="tst",
        )
        LABEL_NAMES = self.trainer.test_dataloaders.dataset.label_names
        for i, (_acc, _iou) in enumerate(zip(seg_acc_per_class, seg_iou_per_class)):
            _dic[f"tst_seg_acc{i}({LABEL_NAMES[i]})"] = _acc
            _dic[f"tst_seg_iou{i}({LABEL_NAMES[i]})"] = _iou

        # 记录指标
        self.log_dict(_dic, on_step=False, on_epoch=True, batch_size=seg_label.numel())

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器

        Returns:
            tuple: (优化器列表, 学习率调度器列表)
        """
        # 定义优化器：使用AdamW优化器
        # AdamW是Adam的变体，对权重衰减的处理更有效
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,  # 初始学习率
            weight_decay=self.weight_decay,  # 权重衰减，用于正则化
        )

        # 定义学习率调度器：余弦退火调度器
        # 学习率从初始值逐渐降低到0，形成余弦曲线
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.n_epochs,  # 最大迭代次数
            eta_min=0,  # 最小学习率
        )

        return [optimizer], [scheduler]
