"""v2 训练入口（LightningCLI）。"""

import os  # 标准库：路径处理（abspath/join/dirname 等）
import sys  # 标准库：运行时模块搜索路径 sys.path

from lightning.pytorch.cli import LightningCLI  # Lightning：命令行训练/测试入口封装

# v2 位于子目录：直接运行 `python v2/train.py ...` 时，可能找不到 `v2.*` 模块。
# 这里将项目根目录（本文件父目录）加入 sys.path，保证导入稳定。
project_root = os.path.abspath(  # 项目根目录绝对路径
    os.path.join(  # 拼接路径
        os.path.dirname(__file__),  # .../v2
        os.pardir,  # 上一级目录
    )  # 结束 join
)  # 结束 abspath
if project_root not in sys.path:  # 避免重复追加
    sys.path.append(project_root)  # 追加到尾部：降低覆盖环境同名包的风险

from v2.models.segmentors import AAGNetSegmentor  # 项目代码：模型（LightningModule/封装）
from v2.dataset.SFdatamodule import SFDataModule  # 项目代码：数据模块（LightningDataModule）


class AAGNetCLI(LightningCLI):
    """AAGNet 的 LightningCLI 定制封装。

    目的：禁用 CLI 解析阶段的 ckpt 预加载，避免 weights-only 反序列化失败。
    """

    def _parse_ckpt_path(self) -> None:
        # 禁用 CLI 阶段 ckpt 预加载：让 Trainer 在 fit/test 阶段按需加载。
        return  # 不调用父类实现


if __name__ == "__main__":
    _cli = AAGNetCLI(  # 初始化 CLI：构造函数会解析参数并执行 fit/test/predict/validate 等流程
        model_class=AAGNetSegmentor,  # 模型类
        datamodule_class=SFDataModule,  # 数据模块类
        save_config_kwargs={  # 配置保存相关参数
            "overwrite": True,  # 覆盖同名 config.yaml：规避 LightningCLI 已知问题（https://github.com/Lightning-AI/pytorch-lightning/issues/17168）
        },
    )
