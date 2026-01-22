import torch
from .segmentors import AAGNetSegmentor


class PretrainedAAGNetSegmentor(AAGNetSegmentor):
    """
    预训练的 AAGNet 分割模型
    用于从预训练权重初始化并适应新的类别数
    """
    
    def __init__(
        self,
        pretrained_ckpt,
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
        初始化预训练模型
        
        Args:
            pretrained_ckpt (str): 预训练模型权重路径
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
        # 首先加载预训练模型（仅用于加载权重），25分类头，传入给对象pretrained_model留着备用
        pretrained_model = AAGNetSegmentor.load_from_checkpoint(pretrained_ckpt)
        
        # 使用传入的参数初始化，本模型初始化3分类头
        super().__init__(
            num_classes=num_classes,
            arch=arch,
            edge_attr_dim=edge_attr_dim,
            node_attr_dim=node_attr_dim,
            edge_attr_emb=edge_attr_emb,
            node_attr_emb=node_attr_emb,
            edge_grid_dim=edge_grid_dim,
            node_grid_dim=node_grid_dim,
            edge_grid_emb=edge_grid_emb,
            node_grid_emb=node_grid_emb,
            num_layers=num_layers,
            delta=delta,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            head_hidden_dim=head_hidden_dim,
            conv_on_edge=conv_on_edge,
            use_uv_gird=use_uv_gird,
            use_edge_attr=use_edge_attr,
            use_face_attr=use_face_attr,
            lr=lr,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
        )
        
        # 加载预训练权重（除分类头外），加载刚才备用的pretrained_model的权重传入pretrained_state_dict
        pretrained_state_dict = pretrained_model.state_dict()
        
        # 移除分类头的权重
        new_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if not key.startswith('seg_head'):
                new_state_dict[key] = value
        
        # 本模型加载权重
        self.load_state_dict(new_state_dict, strict=False)