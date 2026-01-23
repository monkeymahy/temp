"""v2训练入口（LightningCLI）。"""

import os  # 标准库：路径处理（abspath/join/dirname等）
import sys  # 标准库：运行时模块搜索路径sys.path
from lightning.pytorch.cli import LightningCLI  # Lightning：命令行训练/测试入口封装

# v2位于子目录：直接运行`python v2/main.py ...`时，可能找不到`v2.*`模块  # 场景说明
# 这里将项目根目录（本文件父目录）加入sys.path，保证导入稳定  # 目的说明
project_root = os.path.abspath(  # 计算项目根目录的绝对路径
    os.path.join(  # 拼接路径
        os.path.dirname(__file__),  # 当前文件所在目录（.../v2）
        os.pardir,  # 上一级目录（项目根）
    )  # 结束join
)  # 结束abspath
if project_root not in sys.path:  # 避免重复追加路径
    sys.path.append(project_root)  # 追加到尾部：降低覆盖环境同名包风险

from v2.models import AAGNetSegmentor  # 项目代码：模型（LightningModule封装）
from v2.dataset import MFCAD2DataModule, SFDataModule  # 项目代码：数据模块（LightningDataModule）


class AAGNetCLI(LightningCLI):  # CLI封装类
    """AAGNet的LightningCLI定制封装。"""  # 类说明

    # 目的：禁用CLI解析阶段的ckpt预加载，避免weights-only反序列化失败  # 设计目的
    def _parse_ckpt_path(self) -> None:  # 覆写父类方法
        return  # 直接返回：不调用父类实现


if __name__ == "__main__":  # 入口保护
    _cli = AAGNetCLI(  # 初始化CLI：构造函数会解析参数并执行fit/test/predict/validate等流程
        model_class=AAGNetSegmentor,  # 模型类
        # datamodule_class=MFCAD2DataModule,  # 数据模块类
        # NOTE 此处不再硬编码数据集，只需CLI启动时增加'--data [DataModule类名]'指定数据集  # 使用说明
        # 例如：python v2/main.py fit --data SFDataModule -c v2/configs/sf_csy.yaml  # 示例
        save_config_kwargs={  # 配置保存相关参数
            "overwrite": True,  # 覆盖同名config.yaml：规避LightningCLI已知问题
        },  # 结束save_config_kwargs
    )  # 结束AAGNetCLI初始化
