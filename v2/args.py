import argparse

"""
"edge_attr_dim": 12,
"node_attr_dim": 10,
"edge_attr_emb": 64,  # recommend: 64
"node_attr_emb": 64,  # recommend: 64
"edge_grid_dim": 0,
"node_grid_dim": 7,
"edge_grid_emb": 0,
"node_grid_emb": 64,  # recommend: 64
"num_layers": 3,  # recommend: 3
"delta": 2,  # obsolete
"mlp_ratio": 2,
"drop": 0.25,
"drop_path": 0.25,
"head_hidden_dim": 64,
"conv_on_edge": False,
"use_uv_gird": True,
"use_edge_attr": True,
"use_face_attr": True,
"seed": 42,
"num_workers": 1,
"device": "cuda",
"model_name": "AAGNet",
"dataset": dataset_name,
# "dataset": r"D:\Projects\AAGNet\training_data\MFCAD2",
"dataset": r"C:\Data\SF-JSON",
"epochs": 200,  # option: 100e for MFCAD2; 350e for MFCAD
"lr": 1e-2,
"weight_decay": 1e-2,
"batch_size": 32,
"ema_decay_per_epoch": 1.0 / 2.0,
"""


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
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="AAGNet")
    parser.add_argument("--data_root", type=str, default=r"C:\Data\SF-JSON")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ema_decay_per_epoch", type=float, default=1.0 / 2.0)
    parser.add_argument("--num_classes", type=int, default=6)

    return parser.parse_args()
