"""v2 训练入口（LightningCLI）。

说明：LightningCLI 在启动时会尝试预加载 `--ckpt_path` 来解析配置。
在部分环境/ckpt 组合下，这一步会使用 `torch.load(..., weights_only=True)` 并失败。
对本项目来说没必要在 CLI 阶段加载 ckpt，因此这里禁用该行为，交由 Trainer 在 fit/test 阶段加载。
"""

import os
import sys

from lightning.pytorch.cli import LightningCLI

parent_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
    )
)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from v2.models.segmentors import AAGNetSegmentor
from v2.dataset.SFdatamodule import SFDataModule


class AAGNetCLI(LightningCLI):
    def _parse_ckpt_path(self) -> None:
        # Do not load checkpoint in CLI stage; Trainer will handle loading.
        return


if __name__ == "__main__":
    cli = AAGNetCLI(
        model_class=AAGNetSegmentor,
        datamodule_class=SFDataModule,
        save_config_kwargs={
            "overwrite": True,  # TODO 临时设置：禁用默认的配置保存行为，避免在相同log路径下生成config.yaml
            # "save_to_log_dir": False,  #  disable the standard behavior of saving the config to the log_dir # todo ValueError: `save_to_log_dir=False` only makes sense when subclassing SaveConfigCallback to implement `save_config` and it is desired to disable the standard behavior of saving to log_dir.
        },
    )
