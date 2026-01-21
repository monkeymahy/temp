# SF 数据集：负责从磁盘读取 AAG 图数据与对应标签，并按需做标准化/增强

import pathlib  # 使用 Path 处理路径（更稳健、跨平台）
import json  # 读取 *.json 数据文件
import torch  # 将标签转为 torch.Tensor
import numpy as np  # 读取 split 列表（txt）等
import os  # 处理脚本所在目录等
import sys  # 动态追加工程根目录到 sys.path 便于导入
from torch.utils.data import Dataset  # 引入 PyTorch Dataset 基类

# # 计算工程根目录（v2 的上两级），用于支持直接运行本文件时也能正确 import
# parent_dir = os.path.abspath(  # 得到规范化后的绝对路径
#     os.path.join(  # 拼接路径片段
#         os.path.dirname(__file__),  # 当前文件所在目录
#         os.pardir,  # 上一级
#         os.pardir,  # 再上一级
#     )
# )
# if parent_dir not in sys.path:  # 避免重复插入 sys.path
#     sys.path.append(parent_dir)  # 将工程根目录加入模块搜索路径

# 从工程工具库导入数据处理相关函数
from v2.utils.data_utils import (  # noqa: E402（允许在 sys.path 调整后再 import）
    load_one_graph,  # 从 json 数据构建单个图对象
    load_statistics,  # 读取用于标准化的统计量
    standardization,  # 对特征做标准化
    center_and_scale as center_and_scale_fn,  # 避免与参数/属性同名造成歧义
    get_random_rotation,  # 生成离散随机旋转（90°倍数）
    rotate_uvgrid,  # 对 uv grid 做旋转增强
    filter_filenames_by_ids_9s,  # 根据 id 列表筛选文件路径（并校验）
)


class SFDataset(Dataset):  # 显式继承 Dataset，增强兼容性与可读性
    """SF 数据集读取类（PyTorch 风格：实现 __len__ / __getitem__）。"""

    def label_names(self) -> list[str]:
        """返回分类类别名称列表。"""

        if not hasattr(self, "_label_names"):
            # self._label_names = [  # 原始类别名称列表
            #     "slot groove",
            #     "pocket",
            #     "planar",
            #     "hole",
            #     "notch,boss,wedm,chamfer",
            #     "other",
            # ]
            self._label_names = [  # 类别缩减后的类别名称列表
                "other",
                "slot groove",
                "hole",
            ]
        return self._label_names  # 返回常量中的类别名称列表

    def label_mapping(self) -> dict[int, int]:
        """返回原始标签到训练标签的映射。"""
        # mapping = {
        #     0: 1,  # slot groove -> slot groove
        #     1: 0,  # pocket -> other
        #     2: 0,  # planar -> other
        #     3: 2,  # hole -> hole
        #     4: 0,  # notch,boss,wedm,chamfer -> other
        #     5: 0,  # other -> other
        # }
        if not hasattr(self, "lut"):
            self.lut = np.asarray([1, 0, 0, 2, 0, 0], dtype=np.int32)

        return self.lut  # 返回常量中的标签映射字典

    def __init__(
        self,
        root_dir: str,  # 数据集根目录（包含 aag/ labels/ train.txt 等）
        split: str = "train",  # 数据划分：train/val/test
        normalize: bool = True,  # 是否启用标准化
        center_and_scale: bool = True,  # 是否启用居中与缩放（几何归一）
        random_rotate: bool = False,  # 是否做随机旋转增强
        transform=None,  # 可选的外部 transform（对 graph 做额外变换）
    ) -> None:
        """
        从 root_dir 加载 SF 数据集。

        Args:
            root_dir: 数据集根路径。
            split: 选择 train / val / test。
            normalize: 是否使用 attr_stat.json 对特征标准化。
            center_and_scale: 是否对几何做中心化与缩放。
            random_rotate: 是否对网格做 90°倍数随机旋转增强。
            transform: 用户自定义的图变换函数（输入 graph，输出 graph）。
        """
        assert isinstance(root_dir, str)  # root_dir 必须是字符串
        assert isinstance(split, str) and split in (
            "train",
            "val",
            "test",
        )  # split 取值校验
        assert isinstance(normalize, bool)  # normalize 类型校验
        assert isinstance(center_and_scale, bool)  # center_and_scale 类型校验
        assert isinstance(random_rotate, bool)  # random_rotate 类型校验
        assert (
            callable(transform) or transform is None
        )  # transform 必须可调用或为空

        self.root_dir = pathlib.Path(
            root_dir
        )  # 将根目录转为 Path，便于 joinpath 等操作
        self.split = split  # 记录当前 split
        self.normalize = normalize  # 是否标准化
        self.do_center_and_scale = (
            center_and_scale  # 用更明确的属性名，避免与函数名混淆
        )
        self.random_rotate = random_rotate  # 是否随机旋转
        self.transform = transform  # 额外变换函数

        # 扫描 aag 目录下所有 json（注意：后续会按 split 再过滤）
        self.graph_json_paths = list(
            (self.root_dir / "aag").rglob("*.json")
        )  # 预扫描所有（1859个）图文件
        print(
            f">>> 扫描到{len(self.graph_json_paths)}个文件。"
        )  # 输出扫描数量便于核对

        # 若启用标准化，则加载统计量（只加载一次，避免每次 __getitem__ 读取）
        self.stat = None  # 默认无统计量
        if self.normalize:  # 需要标准化才加载
            stat_path = self.root_dir.joinpath(
                "aag/attr_stat.json"
            )  # 统计量文件路径
            self.stat = load_statistics(stat_path=stat_path)  # 读取统计量到内存

        # 读取 split 对应的 id 列表（例如：train.txt）
        split_file = self.root_dir.joinpath(f"{split}.txt")  # split 文件路径
        split_ids = np.loadtxt(
            str(split_file), dtype=str
        )  # 读取所有 id（可能为 1 行或多行）
        split_ids = np.atleast_1d(
            split_ids
        ).tolist()  # 统一转为 list（避免单行时变成标量导致 set 出错）
        split_ids_set = set(split_ids)  # 转为 set，加速后续筛选与查找

        # 根据 split 的 id 列表筛选出对应的文件名子序列（并断言全部存在）
        print(f">>> 加载{split}数据...")  # 输出当前加载的 split
        self.graph_json_paths = (
            filter_filenames_by_ids_9s(  # 按 ids 对文件列表过滤
                filenames=self.graph_json_paths,  # 原始文件列表
                ids=split_ids_set,  # 当前 split 的 id 集合
                index_width=8,  # id 的数字宽度（例如 00000001）
                prefix="graphs_",  # 文件名前缀
                suffix=".json",  # 文件名后缀
            )
        )
        print(
            f">>> 过滤得到{len(self.graph_json_paths)}个文件用于'{split}'数据划分。"
        )  # 输出过滤后数量

    def __len__(self) -> int:
        """返回样本数量。"""
        return len(self.graph_json_paths)  # 直接返回过滤后的文件数

    def _load_one_graph_with_label(self, fn: str, data: dict) -> dict:
        """
        加载单个图，并附加节点标签到 graph.ndata["y"]。

        Args:
            fn: 图样本标识（用于定位 labels/<fn>.json）。
            data: 图结构/属性数据（来自 aag json）。

        Returns:
            one_graph: 包含 "graph" 等字段的样本字典。
        """
        one_graph = load_one_graph(
            fn=fn, data=data
        )  # 调用通用构图逻辑得到 sample/one_graph

        # label 文件与 aag 文件分离存放：labels/<fn>.json
        label_path = self.root_dir.joinpath("labels").joinpath(
            f"{fn}.json"
        )  # 构造标签文件路径
        with open(
            label_path, "r", encoding="utf-8"
        ) as f:  # 显式指定编码，避免跨环境问题
            labels_data = json.load(f)  # 读取标签列表（通常为每个节点一个 int）

        labels_np = np.asarray(
            labels_data, dtype=np.int32
        )  # 转为 int32 的 numpy 数组（更省内存）
        labels_np = self.label_mapping()[
            labels_np  # NOTE `如果使用6类，注释掉这行`
        ]  # 应用标签映射（numpy 索引更高效）
        one_graph["graph"].ndata["y"] = torch.from_numpy(
            labels_np
        ).long()  # 转为 torch.long 作为分类标签

        return one_graph  # 返回已附加标签的样本

    def __getitem__(self, idx: int) -> dict:
        """
        按索引读取样本（读取 json -> 构图 -> 加载标签 -> 增强/标准化）。

        Args:
            idx: 样本索引。

        Returns:
            one_graph: 字典，至少包含 one_graph["graph"]。
        """
        graph_path = self.graph_json_paths[
            idx
        ]  # 取出对应的 json 路径（Path 对象）

        # aag json 内容格式为 [fn, data]（即一个二元结构）
        with open(graph_path, "r", encoding="utf-8") as f:  # 读取图文件
            item = json.load(f)  # 解析 json
            fn, data = item  # 解包得到样本标识与图数据

        one_graph = self._load_one_graph_with_label(
            fn=fn, data=data
        )  # 构图并附加标签

        # 若图没有边，则训练/推理通常无法进行，直接抛出异常以便定位数据问题
        if one_graph["graph"].edata["x"].size(0) == 0:  # 边特征为空意味着无边
            raise ValueError(
                f"Graph has no edges: {graph_path}"
            )  # 增加路径信息便于排查

        # 数据预处理/增强：尽量只在需要时执行，避免不必要的开销
        if self.normalize:  # 是否启用特征标准化
            one_graph = standardization(
                data=one_graph, stat=self.stat
            )  # 使用预加载的统计量做标准化

        if self.do_center_and_scale:  # 是否启用几何中心化与缩放
            one_graph = center_and_scale_fn(
                data=one_graph
            )  # 调用工具函数（避免与属性同名）

        if self.random_rotate:  # 是否启用随机旋转增强
            rotation = get_random_rotation()  # 获取随机旋转（离散角度）
            one_graph["graph"].ndata["grid"] = rotate_uvgrid(  # 旋转节点的 grid
                one_graph["graph"].ndata["grid"],
                rotation,
            )
            if (
                "grid" in one_graph["graph"].edata
            ):  # 若边也包含 grid，则同步旋转
                one_graph["graph"].edata["grid"] = (
                    rotate_uvgrid(  # 旋转边的 grid
                        one_graph["graph"].edata["grid"],
                        rotation,
                    )
                )

        if (
            self.transform is not None
        ):  # 若提供额外 transform，则应用到 graph 上
            one_graph["graph"] = self.transform(
                one_graph["graph"]
            )  # 保持与外部 transform 约定一致

        return one_graph  # 返回样本字典


if __name__ == "__main__":
    # 仅用于本文件单独运行时做快速自检（不影响被 import 使用）
    dataset = SFDataset(  # 构建数据集实例
        root_dir=r"C:\Data\SF-JSON",  # 数据集路径（按你的实际路径调整）
        split="train",  # 选择训练集
        normalize=False,  # 关闭标准化（自检时可更快）
        center_and_scale=False,  # 关闭几何归一（自检时可更快）
        random_rotate=False,  # 关闭随机旋转
        transform=None,  # 不使用额外 transform
    )
    print(dataset[0]["graph"].ndata["y"])  # 打印第一个样本的节点标签
