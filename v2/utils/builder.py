from argparse import Namespace
from datetime import datetime
import lightning.pytorch as pl
from swanlab.integration.pytorch_lightning import SwanLabLogger
import os.path as osp
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging


from v2.dataset.SFdatamodule import SFDataModule
from v2.models.segmentors import AAGNetSegmentor
from v2.utils.io import ensure_directories_exist
from v2.callbacks.EMA import EMACallback


def build_model(args):
    """
    构造模型
    """
    assert isinstance(args, Namespace)

    if args.model_name == "AAGNet":
        model = AAGNetSegmentor(
            num_classes=args.num_classes,
            arch="AAGNetGraphEncoder",
            edge_attr_dim=args.edge_attr_dim,
            node_attr_dim=args.node_attr_dim,
            edge_attr_emb=args.edge_attr_emb,
            node_attr_emb=args.node_attr_emb,
            edge_grid_dim=args.edge_grid_dim,
            node_grid_dim=args.node_grid_dim,
            edge_grid_emb=args.edge_grid_emb,
            node_grid_emb=args.node_grid_emb,
            num_layers=args.num_layers,
            delta=args.delta,
            mlp_ratio=args.mlp_ratio,
            drop=args.drop,
            drop_path=args.drop_path,
            head_hidden_dim=args.head_hidden_dim,
            conv_on_edge=args.conv_on_edge,
            use_uv_gird=args.use_uv_gird,
            use_edge_attr=args.use_edge_attr,
            use_face_attr=args.use_face_attr,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_epochs=args.epochs,
        )

    return model


def build_datamodule(args):
    """
    构造数据模块
    """
    assert isinstance(args, Namespace)

    dm = SFDataModule(
        root_dir=args.data_root,
        normalize=False,
        center_and_scale=False,
        random_rotate=False,
        transform=None,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        prefetch_factor=4,
    )

    return dm


def build_trainer(args: Namespace):
    """
    构建并返回 PyTorch Lightning 的 Trainer 实例。

    参数:
        config (Namespace): 包含训练相关参数的命名空间对象。
        custom_callbacks (list): 可选，用户自定义回调列表。

    返回:
        pl.Trainer: 配置好的训练器对象。
    """
    assert isinstance(args, Namespace)
    # assert isinstance(custom_callbacks, list) or custom_callbacks is None

    # 当前时间字符串，用于唯一标识实验
    run_name = datetime.now().strftime("%m-%d-%H-%M-%S")

    CKPT_DIR = osp.join(args.output_dir, args.ckpt_folder, run_name)
    SWANLOG_DIR = osp.join(args.output_dir, args.swanlog_dir, run_name)
    ensure_directories_exist([SWANLOG_DIR])

    # 检查点回调：保存验证集上性能最优的模型
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,  # 监控的指标名称
        mode=args.monitor_mode,  # 指标越大越好
        save_top_k=1,  # 只保留最优的一个模型
        dirpath=CKPT_DIR,  # 检查点保存目录
        filename="{epoch}-{" + args.monitor + ":.2f}",  # 文件名格式
        save_last=True,  # 同时保存最后一个epoch的模型
    )
    swa_callback = StochasticWeightAveraging(swa_epoch_start=10, swa_lrs=0.01)
    ema_callback = EMACallback(decay=0.5 ** (1 / 14))  # todo 硬编码

    callbacks = [checkpoint_callback, swa_callback, ema_callback]

    # # 早停回调：若指标多次未提升则提前终止训练
    # early_stopping_callback = EarlyStopping(
    #     monitor=config.monitor,  # 监控的指标名称
    #     patience=5,  # 容忍轮数
    #     mode=config.monitor_mode,  # 指标越大越好
    #     verbose=config.verbose,  # 输出早停信息
    # )

    # 日志记录器：用于记录训练过程
    swanlab_logger = SwanLabLogger(
        project=args.proj_name,
        experiment_name=run_name,
        logdir=SWANLOG_DIR,
    )

    # 构建 Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,  # 最大训练轮数
        callbacks=callbacks,  # 回调列表
        accelerator="gpu",  # 使用GPU训练
        logger=swanlab_logger,  # 日志记录器
        enable_checkpointing=True,  # 启用检查点
        log_every_n_steps=50,  # 每50步记录一次日志
        deterministic=True,  # 保证可复现性
        # check_val_every_n_epoch=args.check_val_every_n_epoch,  # 验证集检查频率
        num_sanity_val_steps=0,  # 跳过验证集完整性检查
        # limit_train_batches=5,  # 限制训练批次数量（用于debug）
        # limit_val_batches=5,  # 限制验证批次数量（用于debug）
        # limit_test_batches=100,  # 限制测试批次数量（用于debug）
        # gradient_clip_val=1.0,                      # 可选：梯度裁剪
        # accumulate_grad_batches=1,                  # 可选：梯度累积
    )

    return trainer
