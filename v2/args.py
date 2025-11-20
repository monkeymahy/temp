import argparse


def get_args():
    parser = argparse.ArgumentParser(description="AAGNet arguments")

    parser.add_argument("--edge_attr_dim", type=int, default=12)
    parser.add_argument("--node_attr_dim", type=int, default=10)
    parser.add_argument("--edge_attr_emb", type=int, default=64)
    parser.add_argument("--node_attr_emb", type=int, default=64)
    parser.add_argument("--edge_grid_dim", type=int, default=0)
    parser.add_argument("--node_grid_dim", type=int, default=7)
    parser.add_argument("--edge_grid_emb", type=int, default=0)
    parser.add_argument("--node_grid_emb", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--mlp_ratio", type=int, default=2)
    parser.add_argument("--drop", type=float, default=0.25)
    parser.add_argument("--drop_path", type=float, default=0.25)
    parser.add_argument("--head_hidden_dim", type=int, default=64)
    parser.add_argument(
        "--conv_on_edge",
        default=False,
        help="Whether to apply convolution on edges (True/False)",
    )
    parser.add_argument(
        "--use_uv_gird",
        default=True,
        help="Whether to use UV grid (True/False)",
    )
    parser.add_argument(
        "--use_edge_attr",
        default=True,
        help="Whether to use edge attributes (True/False)",
    )
    parser.add_argument(
        "--use_face_attr",
        default=True,
        help="Whether to use face attributes (True/False)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="AAGNet")
    parser.add_argument("--data_root", type=str, default=r"C:\Data\SF-JSON")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ema_decay_per_epoch", type=float, default=1.0 / 2.0)
    parser.add_argument("--num_classes", type=int, default=6)

    # 日志相关参数
    parser.add_argument("--output_dir", type=str, default=r"D:\Projects\AAGNet\output")

    # 检查点相关参数
    parser.add_argument("--ckpt_folder", type=str, default="checkpoints")

    # swanlab相关参数
    parser.add_argument("--proj_name", type=str, default="AAGNet_PL")
    parser.add_argument("--swanlog_dir", type=str, default="swanlog")

    # 监控指标相关参数
    parser.add_argument("--monitor", type=str, default="val_seg_acc_avg")
    parser.add_argument("--monitor_mode", type=str, default="max")

    return parser.parse_args()
