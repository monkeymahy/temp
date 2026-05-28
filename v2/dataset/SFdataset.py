# SF数据集：负责从磁盘读取AAG图数据与对应标签，并按需做标准化/增强

import pathlib  # 使用Path处理路径（更稳健、跨平台）
import json  # 读取*.json数据文件
import torch  # 将标签转为torch.Tensor
import numpy as np  # 读取split列表（txt）等
from torch.utils.data import Dataset  # 引入PyTorch Dataset基类
import os  # 处理脚本所在目录等
import sys  # 动态追加工程根目录到 sys.path 便于导入

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


from v2.utils.data_utils import (
    extract_labels_from_payload,
    load_json_or_pkl,
    resolve_label_file,
    resolve_training_label_dir,
)


class SFDataset(Dataset):  # 显式继承Dataset，增强兼容性与可读性
    """SF数据集读取类"""  # 类说明

    def __init__(  # 初始化数据集
        self,  # 实例本身
        root_dir: str,  # 数据集根目录（包含aag/labels/train.txt等）
        split: str = "train",  # 数据划分：train/val/test
        label_dir: str = "labels",  # 标签目录（相对 root_dir）
        normalize: bool = True,  # 是否启用标准化
        center_and_scale: bool = True,  # 是否启用居中与缩放（几何归一）
        random_rotate: bool = False,  # 是否做随机旋转增强
        transform=None,  # 可选的外部transform（对graph做额外变换）
        label_names: list[str] = None,  # 类别名称列表
    ) -> None:
        """
        从root_dir加载SF数据集。

        Args:
            root_dir: 数据集根路径。
            split: 选择 train / val / test。
            label_dir: 标签目录（相对 root_dir，如 labels / labels_mapped）。
            normalize: 是否使用 attr_stat.json 对特征标准化。
            center_and_scale: 是否对几何做中心化与缩放。
            random_rotate: 是否对网格做 90°倍数随机旋转增强。
            transform: 用户自定义的图变换函数（输入 graph，输出 graph）。
        """
        assert isinstance(root_dir, str)  # root_dir必须是字符串
        assert isinstance(split, str) and split in ("train", "val", "test")  # split取值校验
        assert isinstance(label_dir, str) and len(label_dir.strip()) > 0  # label_dir非空字符串
        assert isinstance(normalize, bool)  # normalize类型校验
        assert isinstance(center_and_scale, bool)  # center_and_scale类型校验
        assert isinstance(random_rotate, bool)  # random_rotate类型校验
        assert callable(transform) or transform is None  # transform必须可调用或为空
        assert isinstance(label_names, list)  # label_names列表类型校验

        self.root_dir = pathlib.Path(root_dir)  # 将根目录转为Path便于joinpath
        self.split = split  # 记录当前 split
        self.label_dir = label_dir  # 标签目录（相对 root_dir）
        self.normalize = normalize  # 是否标准化
        self.do_center_and_scale = center_and_scale  # 用更明确的属性名避免与函数名混淆
        self.random_rotate = random_rotate  # 是否随机旋转
        self.transform = transform  # 额外变换函数
        self.label_names = label_names  # 类别名称列表

        self.label_root = resolve_training_label_dir(self.root_dir, self.label_dir)
        if not self.label_root.exists():
            raise FileNotFoundError(f"label dir not found: {self.label_root}")
        print(f">>> Using label dir: {self.label_root}")
        self.graph_json_paths = list((self.root_dir / "aag").rglob("*.json"))  # 预扫描图文件
        print(f">>> 扫描到{len(self.graph_json_paths)}个文件。")  # 输出扫描数量便于核对

        # 若启用标准化，则加载统计量（只加载一次，避免每次__getitem__读取）
        self.stat = None  # 默认无统计量
        if self.normalize:  # 需要标准化才加载
            stat_path = self.root_dir.joinpath("aag/attr_stat.json")  # 统计量文件路径
            self.stat = load_statistics(stat_path=stat_path)  # 读取统计量到内存

        # 读取split对应的id列表（例如：train.txt）
        split_file = self.root_dir.joinpath("split").joinpath(f"{split}.txt")  # split 文件路径
        split_ids = np.loadtxt(str(split_file), dtype=str)  # 读取所有id（可能为1行或多行）
        split_ids = np.atleast_1d(split_ids).tolist()  # 统一转为list避免单行标量
        split_ids_set = set(split_ids)  # 转为 set，加速后续筛选与查找

        # 根据split的id列表筛选出对应的文件名子序列（并断言全部存在）
        print(f">>> 加载{split}数据...")  # 输出当前加载的split
        self.graph_json_paths = filter_filenames_by_ids_9s(  # 按ids对文件列表过滤
            filenames=self.graph_json_paths,  # 原始文件列表
            ids=split_ids_set,  # 当前split的id集合
            index_width=8,  # id数字宽度（例如00000001）
            prefix="graphs_",  # 文件名前缀
            suffix=".json",  # 文件名后缀
        )
        print(f">>> 过滤得到{len(self.graph_json_paths)}个文件用于'{split}'数据划分。")  # 输出过滤后数量

    def __len__(self) -> int:
        """返回样本数量。"""  # 方法说明
        return len(self.graph_json_paths)  # 返回过滤后的文件数

    def _load_one_graph_with_label(self, fn: str, data: dict) -> dict:
        """
        加载单个图，并附加节点标签到graph.ndata["y"]。

        Args:
            fn: 图样本标识（用于定位labels/<fn>.json）。
            data: 图结构/属性数据（来自aag json）。

        Returns:
            one_graph: 包含"graph"等字段的样本字典。
        """
        one_graph = load_one_graph(fn=fn, data=data)  # 调用通用构图逻辑

        # label文件与aag文件分离存放：<label_dir>/<fn>.json
        label_path = resolve_label_file(self.label_root, fn)
        label_data = load_json_or_pkl(label_path)
        labels = extract_labels_from_payload(label_data)
        if labels is None:
            raise ValueError(f"label payload does not contain face labels: {label_path}")
        labels_np = np.asarray(labels, dtype=np.int32)
        one_graph["graph"].ndata["y"] = torch.from_numpy(labels_np).long()  # 转为torch.long分类标签

        return one_graph  # 返回已附加标签的样本

    def __getitem__(self, idx: int) -> dict:
        """
        按索引读取样本（读取json->构图->加载标签->增强/标准化）。

        Args:
            idx: 样本索引。

        Returns:
            one_graph: 字典，至少包含one_graph["graph"]。
        """
        graph_path = self.graph_json_paths[idx]  # 取出对应json路径（Path对象）

        # aag json 内容格式为 [fn, data]（即一个二元结构）
        with open(graph_path, "r", encoding="utf-8") as f:  # 读取图文件
            item = json.load(f)  # 解析json
            fn, data = item  # 解包得到样本标识与图数据

        one_graph = self._load_one_graph_with_label(fn=fn, data=data)  # 构图并附加标签

        # 若图没有边，则训练/推理通常无法进行，直接抛出异常以便定位数据问题
        if one_graph["graph"].edata["x"].size(0) == 0:  # 边特征为空意味着无边
            raise ValueError(f"Graph has no edges: {graph_path}")  # 增加路径信息便于排查

        # 数据预处理/增强：尽量只在需要时执行，避免不必要的开销
        if self.normalize:  # 是否启用特征标准化
            one_graph = standardization(data=one_graph, stat=self.stat)  # 使用预加载统计量

        if self.do_center_and_scale:  # 是否启用几何中心化与缩放
            one_graph = center_and_scale_fn(data=one_graph)  # 调用工具函数

        if self.random_rotate:  # 是否启用随机旋转增强
            rotation = get_random_rotation()  # 获取随机旋转（离散角度）
            one_graph["graph"].ndata["grid"] = rotate_uvgrid(  # 旋转节点grid
                one_graph["graph"].ndata["grid"],
                rotation,
            )
            if "grid" in one_graph["graph"].edata:  # 若边包含grid则同步旋转
                one_graph["graph"].edata["grid"] = rotate_uvgrid(  # 旋转边grid
                    one_graph["graph"].edata["grid"],
                    rotation,
                )

        if self.transform is not None:  # 若提供额外transform则应用到graph上
            one_graph["graph"] = self.transform(one_graph["graph"])  # 保持与外部transform约定一致

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
