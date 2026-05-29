import dgl  # 图深度学习库：用于图批处理
import torch  # PyTorch：张量与设备判断
import lightning.pytorch as L  # Lightning：训练框架基类
from torch.utils.data import DataLoader  # PyTorch数据加载器

from v2.dataset.SFdataset import SFDataset  # 项目数据集实现


class SFDataModule(L.LightningDataModule):  # Lightning数据模块封装
    """用于私有数据集的LightningDataModule，负责数据加载、划分与批处理。"""  # 类说明

    def __init__(  # 初始化数据模块
        self,  # 实例本身
        root_dir,  # 数据集根目录
        label_dir="labels",  # 标签目录（相对 root_dir）
        normalize=False,  # 是否标准化
        center_and_scale=False,  # 是否居中缩放
        random_rotate=False,  # 是否随机旋转
        transform=None,  # 可选变换函数
        batch_size=32,  # 批大小
        drop_last=False,  # 是否丢弃最后不满批
        num_workers=4,  # 数据加载线程数
        prefetch_factor=4,  # 预取批次数
        label_names=None,  # 类别名称列表
        task_mode="seg_only",  # 任务模式：seg_only / seg_inst
    ):
        """初始化SF数据模块。"""  # 简要说明
        # 参数校验，确保健壮性
        assert isinstance(root_dir, str)  # root_dir必须是字符串
        assert isinstance(label_dir, str) and len(label_dir.strip()) > 0  # label_dir非空字符串
        assert isinstance(normalize, bool)  # normalize必须是bool
        assert isinstance(center_and_scale, bool)  # center_and_scale必须是bool
        assert isinstance(random_rotate, bool)  # random_rotate必须是bool
        assert callable(transform) or transform is None  # transform需可调用或为空
        assert isinstance(batch_size, int) and batch_size > 0  # batch_size正整数
        assert isinstance(drop_last, bool)  # drop_last必须是bool
        assert isinstance(num_workers, int) and num_workers >= 0  # num_workers非负整数
        assert isinstance(prefetch_factor, int) and prefetch_factor > 0  # prefetch_factor正整数
        assert isinstance(label_names, list)  # label_names列表类型校验
        assert isinstance(task_mode, str) and task_mode in ("seg_only", "seg_inst")

        super().__init__()  # 初始化父类

        # 成员变量赋值
        self.root_dir = root_dir  # 数据根目录
        self.label_dir = label_dir  # 标签目录（相对 root_dir）
        self.normalize = normalize  # 标准化开关
        self.center_and_scale = center_and_scale  # 居中缩放开关
        self.random_rotate = random_rotate  # 随机旋转开关
        self.transform = transform  # 变换函数
        self.batch_size = batch_size  # 批大小
        self.drop_last = drop_last  # 丢弃最后不满批
        self.num_workers = num_workers  # 线程数
        self.persistent_workers = self.num_workers > 0  # 多进程持久化开关
        self.prefetch_factor = prefetch_factor  # 预取批次数
        self.pin_memory = True if torch.cuda.is_available() else False  # CUDA时启用pin_memory
        self.label_names = label_names  # 类别名称列表
        self.task_mode = task_mode  # 任务模式

    def setup(self, stage: str = None):  # 按阶段构建数据集
        """根据stage初始化数据集。"""  # 简要说明
        assert stage is None or stage in ("fit", "test")  # stage取值校验

        # 仅在训练阶段或未指定阶段时加载训练/验证集
        if stage == "fit" or stage is None:  # 训练/验证阶段
            # 训练集
            self.ds_train = SFDataset(  # 创建训练集
                root_dir=self.root_dir,  # 数据根目录
                split="train",  # 训练划分
                label_dir=self.label_dir,  # 标签目录
                normalize=self.normalize,  # 标准化开关
                center_and_scale=self.center_and_scale,  # 居中缩放开关
                random_rotate=self.random_rotate,  # 随机旋转开关
                transform=self.transform,  # 变换函数
                label_names=self.label_names,  # 类别名称列表
                task_mode=self.task_mode,  # 任务模式
            )
            # 验证集
            self.ds_valid = SFDataset(  # 创建验证集
                root_dir=self.root_dir,  # 数据根目录
                split="val",  # 验证划分
                label_dir=self.label_dir,  # 标签目录
                normalize=self.normalize,  # 标准化开关
                center_and_scale=self.center_and_scale,  # 居中缩放开关
                random_rotate=self.random_rotate,  # 随机旋转开关
                transform=self.transform,  # 变换函数
                label_names=self.label_names,  # 类别名称列表
                task_mode=self.task_mode,  # 任务模式
            )
        elif stage == "test":  # 测试阶段
            self.ds_test = SFDataset(  # 创建测试集
                root_dir=self.root_dir,  # 数据根目录
                split="test",  # 测试划分
                label_dir=self.label_dir,  # 标签目录
                normalize=self.normalize,  # 标准化开关
                center_and_scale=self.center_and_scale,  # 居中缩放开关
                random_rotate=self.random_rotate,  # 随机旋转开关
                transform=self.transform,  # 变换函数
                label_names=self.label_names,  # 类别名称列表
                task_mode=self.task_mode,  # 任务模式
            )
        else:  # 非法stage
            raise NotImplementedError("仅支持训练/验证阶段的数据加载。")  # 明确错误

    def train_dataloader(self):  # 训练集加载器
        """构建训练集DataLoader。"""  # 简要说明
        return DataLoader(  # 返回DataLoader
            self.ds_train,  # 训练集
            batch_size=self.batch_size,  # 批大小
            shuffle=True,  # 训练集打乱
            num_workers=self.num_workers,  # 线程数
            collate_fn=self._collate,  # 批处理拼接函数
            persistent_workers=self.persistent_workers,  # 持久化worker
            pin_memory=self.pin_memory,  # 是否使用锁页内存
            prefetch_factor=self.prefetch_factor,  # 预取批次数
            drop_last=self.drop_last,  # 是否丢弃不满批
        )

    def val_dataloader(self):  # 验证集加载器
        """构建验证集DataLoader。"""  # 简要说明
        return DataLoader(  # 返回DataLoader
            self.ds_valid,  # 验证集
            batch_size=self.batch_size,  # 批大小
            shuffle=False,  # 验证集不打乱
            num_workers=self.num_workers,  # 线程数
            collate_fn=self._collate,  # 批处理拼接函数
            persistent_workers=self.persistent_workers,  # 持久化worker
            pin_memory=self.pin_memory,  # 是否使用锁页内存
            prefetch_factor=self.prefetch_factor,  # 预取批次数
            drop_last=self.drop_last,  # 是否丢弃不满批
        )

    def test_dataloader(self):  # 测试集加载器
        """构建测试集DataLoader。"""  # 简要说明
        return DataLoader(  # 返回DataLoader
            self.ds_test,  # 测试集
            batch_size=self.batch_size,  # 批大小
            shuffle=False,  # 测试集不打乱
            num_workers=self.num_workers,  # 线程数
            collate_fn=self._collate,  # 批处理拼接函数
            persistent_workers=False,  # 测试阶段关闭持久化
            pin_memory=self.pin_memory,  # 是否使用锁页内存
            prefetch_factor=self.prefetch_factor,  # 预取批次数
            drop_last=self.drop_last,  # 是否丢弃不满批
        )

    def _collate(self, batch):  # 自定义批处理函数
        """将样本列表合并为批次。"""  # 简要说明
        batched_graph = dgl.batch([sample["graph"] for sample in batch])  # 合并图  # 取出每个样本的graph
        batched_filenames = [sample["filename"] for sample in batch]  # 收集文件名  # 取出每个样本的filename

        result = {"graph": batched_graph, "filename": batched_filenames}  # 返回批次字典
        if self.task_mode == "seg_inst":
            num_faces = [sample["inst_y"].shape[0] for sample in batch]
            max_faces = max(num_faces)
            inst_labels = torch.zeros(len(batch), max_faces, max_faces, dtype=torch.float32)
            inst_mask = torch.zeros(len(batch), max_faces, max_faces, dtype=torch.float32)
            for batch_idx, sample in enumerate(batch):
                size = sample["inst_y"].shape[0]
                inst_labels[batch_idx, :size, :size] = sample["inst_y"]
                inst_mask[batch_idx, :size, :size] = 1.0
            result["inst_labels"] = inst_labels
            result["inst_mask"] = inst_mask
            result["num_faces"] = torch.tensor(num_faces, dtype=torch.long)
        return result
