from pathlib import Path
import pathlib
import json
import torch
import dgl
import numpy as np

from .base import BaseDataset
from utils.data_utils import load_one_graph, load_statistics, load_json_or_pkl
from utils.data_utils import standardization, center_and_scale
from utils.data_utils import get_random_rotation, rotate_uvgrid


class MFCAD2Dataset(BaseDataset):

    @staticmethod
    def num_classes():
        return 25

    # 20251029
    def __init__(
        self,
        root_dir,
        graphs=None,
        split="train",
        normalize=True,
        center_and_scale=True,
        random_rotate=False,
        num_train_data=-1,
        transform=None,
        num_threads=0,  # todo 无用参数
    ):
        """
        Load the MFInstSeg Dataset from the root directory.

        Args:
            root_dir (str): Root path of the dataset.
            graphs (list, optional): List of graph data.
            split (str, optional): Data split to load. Defaults to "train".
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
            num_train_data (int, optional): Number of training examples to use. Defaults to -1 (all training examples will be used).
            transform (callable, optional): Transformation to apply to the data.
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
        path = pathlib.Path(root_dir)
        self.path = path

        # todo 硬编码
        SPLIT_DIR = Path("C:\Data\MFCAD2_split")
        self.filenames = list(SPLIT_DIR.rglob("*.json"))
        print("Done scanning {} files".format(len(self.filenames)))

        if self.normalize:
            self.stat = load_statistics(stat_path=path.joinpath("aag/attr_stat.json"))

        filelist = {}
        _file = str(path.joinpath(f"{split}.txt"))
        data = np.loadtxt(_file, dtype=str)
        filelist[split] = data

        if split == "train":
            split_filelist = filelist["train"][:num_train_data]
        elif split == "val":
            split_filelist = filelist["val"]
        else:
            split_filelist = filelist["test"]

        # Load graphs
        print(f"Loading {split} data...")
        split_filelist = set(split_filelist)

    # 20251029
    def load_graphs(  # todo 无用函数
        self,
        file_path,
        graphs=None,
        split_file_list=None,
        center_and_scale_grid=True,
        normalization_attribute=True,
        num_threads=4,
    ):
        assert isinstance(file_path, Path)
        assert isinstance(graphs, (type(None), list))
        assert isinstance(split_file_list, set)
        assert isinstance(center_and_scale_grid, bool)
        assert isinstance(normalization_attribute, bool)
        assert isinstance(num_threads, int) and num_threads > 0

        self.filenames = []

        if graphs:
            self.dataset = graphs
        else:  # todo 替换
            self.dataset = load_json_or_pkl(json_path=file_path.joinpath("graphs.json"))
        if normalization_attribute:
            stat = load_statistics(stat_path=file_path.joinpath("attr_stat.json"))

    def _collate(self, batch):
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

    # 20251029
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
        label_file = self.path.joinpath("labels").joinpath(fn + ".json")
        with open(str(label_file), "r") as read_file:
            labels_data = json.load(read_file)
        labels_data = np.array(labels_data, dtype=np.int32)
        sample["graph"].ndata["y"] = torch.tensor(labels_data).long()

        return sample

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 检索json文件名
        filename = self.filenames[idx]
        # 读取json内容
        with open(filename, "r") as read_file:
            item = json.load(read_file)
            fn, data = item

        one_graph = self.load_one_graph(fn, data)

        if one_graph["graph"].edata["x"].size(0) == 0:
            # Catch the case of graphs with no edges
            raise ValueError("Graph has no edges")

        # use data augmentation
        if self.normalize:
            one_graph = standardization(data=one_graph, stat=self.stat)
        if self.center_and_scale:
            one_graph = center_and_scale(data=one_graph)

        # one_graph = self.filenames[idx]# todo 添加处理好的graph
        if self.random_rotate:
            rotation = get_random_rotation()
            one_graph["graph"].ndata["grid"] = rotate_uvgrid(
                one_graph["graph"].ndata["grid"], rotation
            )
            if "grid" in one_graph["graph"].edata.keys():
                one_graph["graph"].edata["grid"] = rotate_uvgrid(
                    one_graph["graph"].edata["grid"], rotation
                )
        if self.transform:
            one_graph["graph"] = self.transform(one_graph["graph"])

        return one_graph


if __name__ == "__main__":
    dataset = MFCAD2Dataset(
        root_dir="E:\\MFCAD2",
        split="train",
        center_and_scale=True,
        normalize=False,
    )
    print(dataset[0]["graph"].ndata["y"])
