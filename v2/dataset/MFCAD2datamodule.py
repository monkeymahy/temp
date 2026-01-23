import dgl
import torch
import pathlib
import json
import numpy as np
import lightning.pytorch as L
from torch.utils.data import DataLoader

import sys
import os

# # 添加项目根目录到Python路径
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# )

from v2.utils.data_utils import load_one_graph, load_statistics
from v2.utils.data_utils import standardization, center_and_scale
from v2.utils.data_utils import get_random_rotation, rotate_uvgrid
from v2.utils.data_utils import filter_filenames_by_ids_9s

from torch.utils.data import Dataset


class MFCAD2Dataset(Dataset):
    """
    原始的MFCAD2数据集类，用于加载和处理MFCAD2数据集
    """

    def label_names(self) -> list[str]:
        """返回分类类别名称列表。"""

        if not hasattr(self, "_label_names"):
            # self._label_names = [  # 原始类别名称列表
            # "Chamfer",
            # "Through hole",
            # "Triangular passage",
            # "Rectangular passage",
            # "6-sided passage",
            # "Triangular through slot",
            # "Rectangular through slot",
            # "Circular through slot",
            # "Rectangular through step",
            # "2-sided through step",
            # "Slanted through step",
            # "O-ring",
            # "Blind hole",
            # "Triangular pocket",
            # "Rectangular pocket",
            # "6-sided pocket",
            # "Circular end pocket",
            # "Rectangular blind slot",
            # "Vertical circular end blind slot",
            # "Horizontal circular end blind slot",
            # "Triangular blind step",
            # "Circular blind step",
            # "Rectangular blind step",
            # "Round",
            # "Stock",
            # ]
            self._label_names = [  # 类别缩减后的类别名称列表
                "other",
                "hole",
                "slot",
            ]
        return self._label_names  # 返回常量中的类别名称列表

    def label_mapping(self) -> dict[int, int]:
        """返回原始标签到训练标签的映射。"""
        # mapping = {
        #     0: 0,
        #     1: 1,
        #     2: 0,
        #     3: 0,
        #     4: 0,
        #     5: 2,
        #     6: 2,
        #     7: 2,
        #     8: 0,
        #     9: 0,
        #     10: 0,
        #     11: 0,
        #     12: 1,
        #     13: 0,
        #     14: 0,
        #     15: 0,
        #     16: 0,
        #     17: 2,
        #     18: 2,
        #     19: 2,
        #     20: 0,
        #     21: 0,
        #     22: 0,
        #     23: 0,
        # }

        # 9类别映射
        # mapping = {
        #     0: 0,   # other -> other
        #     1: 1,   # Through hole -> Through hole
        #     2: 0,   # other -> other
        #     3: 0,   # other -> other
        #     4: 0,   # other -> other
        #     5: 3,   # Triangular through slot -> Triangular through slot
        #     6: 4,   # Rectangular through slot -> Rectangular through slot
        #     7: 5,   # Circular through slot -> Circular through slot
        #     8: 0,   # other -> other
        #     9: 0,   # other -> other
        #     10: 0,  # other -> other
        #     11: 0,  # other -> other
        #     12: 2,  # Blind hole -> Blind hole
        #     13: 0,  # other -> other
        #     14: 0,  # other -> other
        #     15: 0,  # other -> other
        #     16: 0,  # other -> other
        #     17: 6,  # Rectangular blind slot -> Rectangular blind slot
        #     18: 7,  # Vertical circular end blind slot -> Vertical circular end blind slot
        #     19: 8,  # Horizontal circular end blind slot -> Horizontal circular end blind slot
        #     20: 0,  # other -> other
        #     21: 0,  # other -> other
        #     22: 0,  # other -> other
        #     23: 0,  # other -> other
        #     24: 0,  # other -> other
        # }
        # 9类别查找表（注释掉）
        # self.lut = np.asarray([0, 1, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0], dtype=np.int32)

        if not hasattr(self, "lut"):
            # 根据映射关系创建查找表
            # 映射规则：0:other, 1:hole, 2:slot
            self.lut = np.asarray(
                [
                    0,
                    1,
                    0,
                    0,
                    0,
                    2,
                    2,
                    2,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    2,
                    2,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=np.int32,
            )

        return self.lut  # 返回常量中的标签映射字典

    def __init__(
        self,
        root_dir,
        graphs=None,
        split="train",
        normalize=False,
        center_and_scale=False,
        random_rotate=False,
        num_train_data=-1,
        transform=None,
        use_3category=False,
        use_9category=False,
    ):
        """
        初始化MFCAD2数据集

        Args:
            root_dir (str): 数据集根路径
            graphs (list, optional): 图数据列表
            split (str, optional): 数据分割，可选值: "train", "val", "test"
            normalize (bool, optional): 是否标准化数据
            center_and_scale (bool, optional): 是否居中并缩放实体
            random_rotate (bool, optional): 是否应用随机旋转
            num_train_data (int, optional): 训练数据数量，默认-1表示使用所有数据
            transform (callable, optional): 数据变换
        """
        assert isinstance(root_dir, str)
        assert isinstance(graphs, (type(None), list))
        assert split in ("train", "val", "test")
        assert isinstance(normalize, bool)
        assert isinstance(center_and_scale, bool)
        assert isinstance(random_rotate, bool)
        assert isinstance(num_train_data, int) and num_train_data >= -1
        assert callable(transform) or transform is None

        self.root_dir = root_dir
        self.graphs = graphs
        self.split = split
        self.normalize = normalize
        self.center_and_scale = center_and_scale
        self.random_rotate = random_rotate
        self.num_train_data = num_train_data
        self.transform = transform
        root_dir = pathlib.Path(root_dir)
        self.root_dir = root_dir
        self.use_3category = use_3category
        self.use_9category = use_9category

        # 1. 扫描所有图结构数据文件，已经分割后的
        #  - self.filenames: 存储所有找到的JSON文件的完整路径列表，如 [Path("E:/AAGnetV2/MFCAD2/aag/1.json"), ...]
        self.filenames = list((root_dir / "aag").rglob("*.json"))
        print(">>> Done scanning {} files".format(len(self.filenames)))

        # 2. 如果需要标准化，加载统计信息
        if self.normalize:
            print(
                f"Normalize is True, trying to load stat file from {root_dir.joinpath('aag/attr_stat.json')}"
            )
            self.stat = load_statistics(
                stat_path=root_dir.joinpath("aag/attr_stat.json")
            )
        else:
            print("Normalize is False, skipping loading stat file")

        # 3. 加载当前split对应的文件ID列表
        #  - files_id: 字典，存储不同split的文件ID集合
        #  - _file: 当前split的ID文件路径，如 "E:/AAGnetV2/MFCAD2/train.txt"
        #  - data: NumPy数组，存储当前split的所有文件ID字符串，如 array(['1', '2', '3', ...], dtype='<U3')
        #  - files_id[split]: 将ID数组存储到字典中，键为split名称（如'train'）
        files_id = {}
        _file = str(root_dir.joinpath(f"{split}.txt"))
        data = np.loadtxt(_file, dtype=str)
        files_id[split] = data

        # 4. 如果是训练集，限制训练数据的数量
        #  - num_train_data: 整数，限制的训练数据数量（-1表示使用所有数据）
        #  - files_id['train'][:num_train_data]: 只有当num_train_data != -1时，才截取前num_train_data个ID
        if split == "train":
            if num_train_data != -1:
                files_id[split] = files_id[split][:num_train_data]
        # 5. 转换ID列表为集合以提高查找效率
        #  - files_id[split] = set(...): 将ID数组转换为集合，如 {'1', '2', '3', ...}
        print(f">>> Loading {split} data...")
        files_id[split] = set(files_id[split])

        # 6. 根据当前split的ID集合筛选对应的文件名得到文件路径，放到self.filenames。并记缺失的文件ID。
        #  - filter_filenames_by_ids_9s: 筛选函数，根据ID集合匹配文件名
        #  - filenames: 所有JSON文件路径列表
        #  - ids: 当前split的ID集合
        #  - index_width: ID的填充宽度（当前未使用）
        #  - prefix: 文件名前缀（设置为空字符串）
        #  - suffix: 文件名后缀（.json）
        #  - 返回值: 筛选后的文件路径列表，如 [Path("E:/AAGnetV2/MFCAD2/aag/1.json"), ...]
        self.filenames = filter_filenames_by_ids_9s(
            filenames=self.filenames,
            ids=files_id[split],
            index_width=8,
            prefix="",
            suffix=".json",
        )
        print(f">>> Filtered {len(self.filenames)} files for split '{split}'.")

    def _collate(self, batch):
        """
        将一批数据样本合并为一个批次

        Args:
            batch (List[dict]): 数据样本列表

        Returns:
            dict: 合并后的批次数据
        """
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        return {"graph": batched_graph, "filename": batched_filenames}

    def load_one_graph(self, fn, data):
        """
        加载单个图数据

        Args:
            fn (str): 文件名
            data (dict): 文件数据

        Returns:
            dict: 加载的图数据
        """
        # 使用base class方法加载图
        sample = load_one_graph(fn=fn, data=data)
        # 加载标签并存储为节点数据
        label_file = self.root_dir.joinpath("labels").joinpath(fn + ".json")
        with open(str(label_file), "r") as read_file:
            labels_data = json.load(read_file)

        labels_np = np.asarray(
            labels_data, dtype=np.int32
        )  # 转为 int32 的 numpy 数组（更省内存）
        labels_np = self.label_mapping()[
            labels_np
        ]  # NOTE `如果使用6类，注释掉这行`

        sample["graph"].ndata["y"] = torch.tensor(labels_np).long()

        return sample

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本

        Args:
            idx (int): 样本索引

        Returns:
            dict: 数据样本
        """
        # 检索json文件的路径
        filename = self.filenames[idx]
        # 读取json文件的内容：["0-0-0-0-0-23",{"graph":{"edges":[[0,0,0,0,0,0
        with open(filename, "r") as read_file:
            item = json.load(read_file)
            fn, data = item  # 文件id和数据值

        one_graph = self.load_one_graph(fn=fn, data=data)

        if one_graph["graph"].edata["x"].size(0) == 0:
            # 捕获没有边的图
            raise ValueError("Graph has no edges")

        # 数据增强
        if self.normalize:  # 激活
            one_graph = standardization(data=one_graph, stat=self.stat)
        if self.center_and_scale:  # 未激活
            one_graph = center_and_scale(data=one_graph)

        if self.random_rotate:  # 未激活
            rotation = get_random_rotation()
            one_graph["graph"].ndata["grid"] = rotate_uvgrid(
                one_graph["graph"].ndata["grid"], rotation
            )
            if "grid" in one_graph["graph"].edata.keys():
                one_graph["graph"].edata["grid"] = rotate_uvgrid(
                    one_graph["graph"].edata["grid"], rotation
                )
        if self.transform:  # None
            one_graph["graph"] = self.transform(one_graph["graph"])

        return one_graph


class MFCAD2DataModule(L.LightningDataModule):
    """
    用于MFCAD2数据集的LightningDataModule，负责数据的加载、划分与批处理。
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 32,
        normalize: bool = False,
        center_and_scale: bool = False,
        random_rotate: bool = False,
        num_train_data: int = -1,
        transform=None,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        prefetch_factor=4,
        use_3category=False,
        use_9category=False,
    ):
        """
        初始化MFCAD2数据模块。

        参数:
            root_dir (str): 数据集根路径
            batch_size (int, optional): 批次大小，默认32
            normalize (bool, optional): 是否标准化数据
            center_and_scale (bool, optional): 是否居中并缩放实体
            random_rotate (bool, optional): 是否应用随机旋转
            num_train_data (int, optional): 训练数据数量，默认-1表示使用所有数据
            transform (callable, optional): 数据变换
            shuffle (bool, optional): 是否打乱数据
            drop_last (bool, optional): 是否丢弃最后一个不完整的批次
            num_workers (int, optional): DataLoader加载数据的线程数
            prefetch_factor (int, optional): 预取批次的数量
        """
        # 参数校验，确保健壮性
        assert isinstance(root_dir, str)
        assert isinstance(normalize, bool)
        assert isinstance(center_and_scale, bool)
        assert isinstance(random_rotate, bool)
        assert callable(transform) or transform is None
        assert isinstance(num_train_data, int) and num_train_data >= -1

        super().__init__()

        # 成员变量赋值
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.center_and_scale = center_and_scale
        self.random_rotate = random_rotate
        self.num_train_data = num_train_data
        self.transform = transform
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = prefetch_factor  # 预取更多批次以提高数据加载效率
        self.pin_memory = True if torch.cuda.is_available() else False
        self.ds_train = None
        self.ds_valid = None
        self.ds_test = None
        self.use_3category = use_3category
        self.use_9category = use_9category

    def setup(self, stage: str = None):
        """
        根据不同阶段(stage)初始化数据集。

        参数:
            stage (str, optional): 指定数据集初始化的阶段，可选值为 "fit"（训练/验证）、"test"（测试）或 None（默认，初始化训练和验证集）。
        """
        assert stage is None or stage in ("fit", "test")

        # 仅在训练阶段或未指定阶段时加载训练/验证集
        if stage == "fit" or stage is None:
            # 训练集
            self.ds_train = MFCAD2Dataset(
                root_dir=self.root_dir,
                split="train",
                normalize=self.normalize,
                center_and_scale=self.center_and_scale,
                random_rotate=self.random_rotate,
                num_train_data=self.num_train_data,
                transform=self.transform,
                use_3category=self.use_3category,
                use_9category=self.use_9category,
            )
            # [:5]
            # 验证集
            self.ds_valid = MFCAD2Dataset(
                root_dir=self.root_dir,
                split="val",
                normalize=self.normalize,
                center_and_scale=self.center_and_scale,
                random_rotate=False,  # 验证集不使用随机旋转
                transform=self.transform,
                use_3category=self.use_3category,
                use_9category=self.use_9category,
            )
        elif stage == "test":
            self.ds_test = MFCAD2Dataset(
                root_dir=self.root_dir,
                split="test",
                normalize=self.normalize,
                center_and_scale=self.center_and_scale,
                random_rotate=False,  # 测试集不使用随机旋转
                transform=self.transform,
                use_3category=self.use_3category,
                use_9category=self.use_9category,
            )
        else:
            raise NotImplementedError("仅支持训练/验证阶段的数据加载。")

    @staticmethod
    def _collate(batch):
        """
        Collate a batch of data samples together into a single batch.

        Args:
            batch (List[dict]): List of data samples.

        Returns:
            dict: Batched data.
        """
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]

        return {"graph": batched_graph, "filename": batched_filenames}

    def train_dataloader(self):
        """
        构建训练集的DataLoader。
        """
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,  # 训练集需要打乱
            num_workers=self.num_workers,
            collate_fn=self._collate,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        """
        构建验证集的DataLoader。
        """
        return DataLoader(
            self.ds_valid,
            batch_size=self.batch_size,
            shuffle=False,  # 验证集不需要打乱
            num_workers=self.num_workers,
            collate_fn=self._collate,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        """
        构建测试集的DataLoader。
        """
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,  # 测试集不需要打乱
            num_workers=self.num_workers,
            collate_fn=self._collate,
            persistent_workers=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
        )

    def predict_dataloader(self):
        """
        构建预测集的DataLoader。
        """
        return DataLoader(
            self.ds_test,  # 使用测试集进行预测
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            persistent_workers=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
        )


# if __name__ == "__main__":
#     # 测试MFCAD2DataModule
#     data_module = MFCAD2DataModule(
#         root_dir=r"E:\AAGnetV2\aagnet\MFCAD2",
#         batch_size=32,
#         normalize=False,
#         center_and_scale=True,
#     )

#     # 必须先调用setup()方法初始化数据集
#     data_module.setup(stage="fit")

#     # 获取训练数据加载器
#     train_loader = data_module.train_dataloader()

#     # 迭代一个批次
#     for batch in train_loader:
#         print("Batch graph nodes:", batch["graph"].num_nodes())
#         print("Batch graph edges:", batch["graph"].num_edges())
#         print("Batch node labels:", batch["graph"].ndata["y"])
#         break  # 只迭代一个批次
