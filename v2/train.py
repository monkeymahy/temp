# 训练脚本
import os
import sys
from argparse import Namespace
import lightning.pytorch as pl

parent_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
    )
)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from v2.args import get_args
from v2.utils.builder import build_model, build_datamodule, build_trainer


def train(args):
    assert isinstance(args, Namespace)

    model = build_model(args=args)
    datamodule = build_datamodule(args=args)
    trainer = build_trainer(args=args, show_progress_bar=True)

    # 开始训练
    trainer.fit(model, datamodule=datamodule)

    # 获取最佳验证分数和模型路径
    best_score = trainer.checkpoint_callback.best_model_score
    best_path = trainer.checkpoint_callback.best_model_path

    print(f">>> 最佳验证分数: {best_score}")
    print(f">>> 最佳模型路径: {best_path}")

    pass


def test(args):
    pass


if __name__ == "__main__":
    args = get_args()
    pl.seed_everything(args.seed, workers=True)  # 设置随机种子，保证可复现

    train(args)
    pass
