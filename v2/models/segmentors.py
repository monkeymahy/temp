import torch
from torch import nn
import lightning.pytorch as L
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

import models.encoders as encoders
from .layers import MLP
from v2.constant import LABEL_NAMES


###############################################################################
# Segmentation model
###############################################################################


class AAGNetSegmentor(L.LightningModule):
    """
    AAG-Net solid face segmentation model
    """

    # 20251029
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
    ):
        """
        Initialize the AAG-Net solid face segmentation model

        Args:
            num_classes (int): Number of classes to output per-face
            crv_in_channels (int, optional): Number of input channels for the 1D edge UV-grids
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the whole graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        # A linear network to encode B-rep face attributes
        self.use_uv_gird = use_uv_gird
        self.use_edge_attr = use_edge_attr
        self.use_face_attr = use_face_attr
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.seg_loss = nn.CrossEntropyLoss()

        self.node_attr_encoder = nn.Sequential(
            nn.Linear(node_attr_dim, node_attr_emb),
            nn.LayerNorm(node_attr_emb),
        )

        if node_grid_dim:
            self.node_grid_encoder = nn.Sequential(
                nn.Conv2d(
                    node_grid_dim,
                    node_grid_emb // 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(node_grid_emb // 4),
                nn.Mish(),
                nn.Conv2d(
                    node_grid_emb // 4,
                    node_grid_emb // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(node_grid_emb // 2),
                nn.Mish(),
                nn.Conv2d(
                    node_grid_emb // 2,
                    node_grid_emb,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(node_grid_emb),
                nn.Mish(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
            )
        else:
            self.node_grid_encoder = None

        # A linear network to encode B-rep edge attributes
        self.edge_attr_encoder = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_attr_emb),
            nn.LayerNorm(edge_attr_emb),
        )
        if edge_grid_dim:
            # TODO 这是原作者准备优化的地方？
            pass
        node_emb = node_attr_emb + node_grid_emb
        edge_emb = edge_attr_emb + edge_grid_emb
        # A graph neural network that message passes face and edge features
        encoder = getattr(encoders, arch)  # todo 简化
        self.graph_encoder = encoder(
            node_dim=node_emb,
            edge_dim=edge_emb,
            num_layers=num_layers,
            delta=delta,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            conv_on_edge=conv_on_edge,
        )
        final_out_emb = 2 * node_emb
        # A non-linear classifier that maps face embeddings to face logits
        self.seg_head = MLP(
            num_layers=2,
            input_dim=final_out_emb,
            hidden_dim=head_hidden_dim,
            output_dim=num_classes,
            norm=nn.LayerNorm,
            act=nn.Mish,
        )

        self._init_metrics(num_classes=num_classes)

    def _init_metrics(self, num_classes):
        """
        初始化各类指标
        """
        # 训练集指标
        self.tra_seg_acc = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",
        )
        self.tra_seg_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
        )
        self.tra_seg_acc_per_class = MulticlassAccuracy(
            num_classes=num_classes,
            average=None,
        )
        self.tra_seg_iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes,
            average=None,
        )

        # 验证集指标
        self.val_seg_acc = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",
        )
        self.val_seg_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
        )
        self.val_seg_acc_per_class = MulticlassAccuracy(
            num_classes=num_classes,
            average=None,
        )
        self.val_seg_iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes,
            average=None,
        )

        # 测试集论文
        self.tst_seg_acc = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",
        )
        self.tst_seg_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
        )
        self.tst_seg_acc_per_class = MulticlassAccuracy(
            num_classes=num_classes,
            average=None,
        )
        self.tst_seg_iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes,
            average=None,
        )

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor:
                Logits (total_nodes_in_batch x num_classes)
                Bottom Logits (total_nodes_in_batch x 1)
            list [torch.tensor]:
                Face adjacency graph (num_graph_per_batch, num_faces x num_faces)
        """
        # Input features
        input_node_attr = (
            batched_graph.ndata["x"]
            if self.use_face_attr
            else torch.zeros_like(batched_graph.ndata["x"])
        )
        input_node_grid = (
            batched_graph.ndata["grid"]
            if self.use_uv_gird
            else torch.zeros_like(batched_graph.ndata["grid"])
        )
        input_edge_attr = (
            batched_graph.edata["x"]
            if self.use_edge_attr
            else torch.zeros_like(batched_graph.edata["x"])
        )
        # input_edge_grid = batched_graph.edata["grid"]
        # Compute hidden face features
        node_feat = self.node_attr_encoder(input_node_attr)
        if self.node_grid_encoder:
            assert input_node_grid.numel() > 0
            node_grid_feat = self.node_grid_encoder(input_node_grid)
            node_feat = torch.concat([node_feat, node_grid_feat], dim=1)
        # Compute hidden edge features
        edge_feat = self.edge_attr_encoder(input_edge_attr)
        # Message pass and compute per-face(node) and global embeddings
        node_emb, graph_emb = self.graph_encoder(batched_graph, node_feat, edge_feat)
        # concatenated to the per-node embeddings
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(
            graph_emb.device
        )
        local_global_feat = torch.cat((node_emb, graph_emb), dim=1)
        # Map to logits
        seg_out = self.seg_head(local_global_feat)

        return seg_out

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        graphs = batch["graph"]
        seg_label = graphs.ndata["y"]

        seg_pred = self.forward(batched_graph=graphs)
        loss = self.seg_loss(seg_pred, seg_label)

        self.tra_seg_acc(seg_pred, seg_label)
        self.tra_seg_iou(seg_pred, seg_label)
        seg_acc_per_class = self.tra_seg_acc_per_class(seg_pred, seg_label)
        seg_iou_per_class = self.tra_seg_iou_per_class(seg_pred, seg_label)

        _dic = {
            "tra_loss": loss.item(),
            "tra_seg_acc_avg": self.tra_seg_acc,
            "tra_seg_iou_avg": self.tra_seg_iou,
        }
        for i, (_acc, _iou) in enumerate(zip(seg_acc_per_class, seg_iou_per_class)):
            _dic[f"tra_seg_acc{i}({LABEL_NAMES[i]})"] = _acc
            _dic[f"tra_seg_iou{i}({LABEL_NAMES[i]})"] = _iou

        self.log_dict(
            _dic,
            on_step=False,
            on_epoch=True,
            batch_size=seg_label.shape[0],  # TODO 此处并非batch size，后续需要注意
        )

        return loss

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        graphs = batch["graph"]
        seg_label = graphs.ndata["y"]

        seg_pred = self.forward(batched_graph=graphs)
        loss = self.seg_loss(seg_pred, seg_label)

        self.val_seg_acc(seg_pred, seg_label)
        self.val_seg_iou(seg_pred, seg_label)
        seg_acc_per_class = self.val_seg_acc_per_class(seg_pred, seg_label)
        seg_iou_per_class = self.val_seg_iou_per_class(seg_pred, seg_label)

        _dic = {
            "val_loss": loss.item(),
            "val_seg_acc_avg": self.val_seg_acc,
            "val_seg_iou_avg": self.val_seg_iou,
        }
        for i, (_acc, _iou) in enumerate(zip(seg_acc_per_class, seg_iou_per_class)):
            _dic[f"val_seg_acc{i}({LABEL_NAMES[i]})"] = _acc
            _dic[f"val_seg_iou{i}({LABEL_NAMES[i]})"] = _iou
        self.log_dict(
            _dic,
            on_step=False,
            on_epoch=True,
            batch_size=seg_label.shape[0],  # TODO 此处并非batch size，后续需要注意
        )

    def configure_optimizers(self):
        # include trainable NaN fill parameters in predictor optimizer

        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.n_epochs,
            eta_min=0,
        )

        return [optimizer], [scheduler]
