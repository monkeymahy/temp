import dgl
import torch
import lightning.pytorch as L
from torch.utils.data import DataLoader

from v2.dataset.SFdataset import SFDataset


class SFDataModule(L.LightningDataModule):
    """
    用于私有数据集的LightningDataModule，负责数据的加载、划分与批处理。
    """

    def __init__(
        self,
        root_dir=r"C:\Data\SF-JSON",
        normalize=False,
        center_and_scale=False,
        random_rotate=False,
        transform=None,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        prefetch_factor=4,
    ):
        """
        初始化SeasFire数据模块。

        参数:
            processed_data_dir (str): 数据存储目录。
            raw_zarr_path (str): 原始Zarr数据路径。
            history_steps (int): 输入序列的历史步数。
            future_steps (int): 预测序列的未来步数。
            num_folds (int): K折交叉验证的折数。
            test_fold_index (int): 测试集所在折的索引。
            batch_size (int): 批量大小。
            num_workers (int): DataLoader加载数据的线程数。
            label_threshold (float): 标签阈值。
            target_key (str): 目标变量键。
            valid_mask_key (str): 有效时间掩码键。
            verbose (bool): 是否打印详细信息。
        """
        # 参数校验，确保健壮性
        assert isinstance(root_dir, str)
        assert isinstance(normalize, bool)
        assert isinstance(center_and_scale, bool)
        assert isinstance(random_rotate, bool)
        assert callable(transform) or transform is None

        super().__init__()

        # 成员变量赋值
        self.root_dir = root_dir
        self.normalize = normalize
        self.center_and_scale = center_and_scale
        self.random_rotate = random_rotate
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = prefetch_factor  # 预取更多批次以提高数据加载效率
        self.pin_memory = True if torch.cuda.is_available() else False

    # 20251102
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
            self.ds_train = SFDataset(
                root_dir=self.root_dir,
                split="train",
                normalize=self.normalize,
                center_and_scale=self.center_and_scale,
                random_rotate=self.random_rotate,
                transform=self.transform,
            )
            # 验证集
            self.ds_valid = SFDataset(
                root_dir=self.root_dir,
                split="val",
                normalize=self.normalize,
                center_and_scale=self.center_and_scale,
                random_rotate=self.random_rotate,
                transform=self.transform,
            )
        elif stage == "test":
            self.ds_test = SFDataset(
                root_dir=self.root_dir,
                split="test",
                normalize=self.normalize,
                center_and_scale=self.center_and_scale,
                random_rotate=self.random_rotate,
                transform=self.transform,
            )
        else:
            raise NotImplementedError("仅支持训练/验证阶段的数据加载。")

    def train_dataloader(self):
        """
        构建训练集的DataLoader。
        """
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
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
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
        )

    # 20251102
    def test_dataloader(self):
        """
        复用验证集来构建测试集的DataLoader。
        由于数据划分方式相同，测试集与验证集一致。
        """
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            persistent_workers=False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
        )

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
