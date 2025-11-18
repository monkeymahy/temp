# 私有数据集

import pathlib
import json
import torch
import numpy as np
import os
import sys

parent_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
    )
)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from v2.utils.data_utils import (
    load_one_graph,
    load_statistics,
    standardization,
    center_and_scale,
    get_random_rotation,
    rotate_uvgrid,
    filter_filenames_by_ids_9s,
)


class SFDataset:

    @staticmethod
    def num_classes():  # TODO 是否可以非硬编码
        return 6  # SF数据

    # 20251104
    def __init__(
        self,
        root_dir,
        split="train",
        normalize=True,
        center_and_scale=True,
        random_rotate=False,
        transform=None,
    ):
        """
        Load the MFInstSeg Dataset from the root directory.

        Args:
            root_dir (str): Root path of the dataset.
            split (str, optional): Data split to load. Defaults to "train".
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
            transform (callable, optional): Transformation to apply to the data.
        """
        assert isinstance(root_dir, str)
        assert isinstance(split, str) and split in ("train", "val", "test")
        assert isinstance(normalize, bool)
        assert isinstance(center_and_scale, bool)
        assert isinstance(random_rotate, bool)
        assert callable(transform) or transform is None

        root_dir = pathlib.Path(root_dir)
        self.root_dir = root_dir
        self.split = split
        self.normalize = normalize
        self.center_and_scale = center_and_scale
        self.random_rotate = random_rotate
        self.transform = transform

        self.filenames = list((root_dir / "aag").rglob("*.json"))  # 59455文件
        print(">>> Done scanning {} files".format(len(self.filenames)))

        if self.normalize:
            self.stat = load_statistics(
                stat_path=root_dir.joinpath("aag/attr_stat.json")
            )

        files_id = {}
        _file = str(root_dir.joinpath(f"{split}.txt"))
        data = np.loadtxt(_file, dtype=str)
        files_id[split] = data

        # Load graphs
        print(f">>> Loading {split} data...")
        files_id[split] = set(files_id[split])

        # 根据 split 的 id 列表筛选出对应的文件名子序列（并断言全部存在）
        self.filenames = filter_filenames_by_ids_9s(
            filenames=self.filenames,
            ids=files_id[split],
            index_width=8,
            prefix="graphs_",
            suffix=".json",
        )
        print(f">>> Filtered {len(self.filenames)} files for split '{split}'.")

    # 20251104
    def load_one_graph(self, fn, data):
        """
        Load the data for a single file.

        Args:
            fn (str): Filename.
            data (dict): Data for the file.

        Returns:
            dict: Data for the file.
        """
        # Load the graph using base class method
        sample = load_one_graph(fn=fn, data=data)
        # Additionally load the label and store it as node data
        label_file = self.root_dir.joinpath("labels").joinpath(fn + ".json")
        with open(str(label_file), "r") as read_file:
            labels_data = json.load(read_file)
        labels_data = np.array(labels_data, dtype=np.int32)
        sample["graph"].ndata["y"] = torch.tensor(labels_data).long()

        return sample

    def __len__(self):
        return len(self.filenames)

    # 20251104
    def __getitem__(self, idx):
        # 检索json文件名
        filename = self.filenames[idx]
        # 读取json内容
        with open(filename, "r") as read_file:
            item = json.load(read_file)
            fn, data = item

        one_graph = self.load_one_graph(fn=fn, data=data)

        if one_graph["graph"].edata["x"].size(0) == 0:
            # Catch the case of graphs with no edges
            raise ValueError("Graph has no edges")

        # use data augmentation
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


if __name__ == "__main__":
    dataset = SFDataset(
        root_dir=r"C:\Data\SF-JSON",
        split="train",
        normalize=False,
        center_and_scale=False,
        random_rotate=False,
        transform=None,
    )
    print(dataset[0]["graph"].ndata["y"])
