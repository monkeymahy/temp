from pathlib import Path
import random
import os.path as osp
import json
import pathlib
from typing import List, Union
import numpy as np
import torch
import dgl
from scipy.spatial.transform import Rotation
from OCC.Core.STEPControl import STEPControl_Reader


def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)


def bounding_box_uvgrid(inp: torch.Tensor):
    pts = inp[..., :3].reshape((-1, 3))
    mask = inp[..., 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    return bounding_box_pointcloud(pts)


def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):
    inp = inp.transpose(1, 3)  # channel last
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[..., :3] -= center
    inp[..., :3] *= scale
    inp = inp.transpose(1, 3)  # channel first
    if return_center_scale:
        return inp, center, scale
    return


def center_and_scale(data: torch.Tensor):
    data["graph"].ndata["grid"], center, scale = center_and_scale_uvgrid(
        data["graph"].ndata["grid"], return_center_scale=True
    )
    if "grid" in data["graph"].edata.keys():
        egrid = data["graph"].edata["grid"]
        egrid = egrid.transpose(1, 2)  # channel last
        egrid[..., :3] -= center
        egrid[..., :3] *= scale
        egrid = egrid.transpose(1, 2)  # channel first
        data["graph"].edata["grid"] = egrid
    return data


# 20251029
def standardization(data, stat):
    data["graph"].ndata["x"] -= stat["mean_face_attr"]
    data["graph"].ndata["x"] /= stat["std_face_attr"]
    data["graph"].edata["x"] -= stat["mean_edge_attr"]
    data["graph"].edata["x"] /= stat["std_edge_attr"]
    return data


def get_random_rotation():
    """
    Get a random rotation in 90 degree increments along the canonical axes
    """
    axes = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]
    angles = [0.0, 90.0, 180.0, 270.0]
    axis = random.choice(axes)
    angle_radians = np.radians(random.choice(angles))
    return Rotation.from_rotvec(angle_radians * axis)


def rotate_uvgrid(inp, rotation):
    """
    Rotate the node features in the graph by a given rotation
    """
    inp = inp.transpose(1, 3)  # channel last
    Rmat = torch.tensor(rotation.as_matrix()).float()
    orig_size = inp[..., :3].size()
    inp[..., :3] = torch.mm(inp[..., :3].reshape(-1, 3), Rmat).reshape(
        orig_size
    )  # Points
    inp[..., 3:6] = torch.mm(inp[..., 3:6].reshape(-1, 3), Rmat).reshape(
        orig_size
    )  # Normals/tangents
    inp = inp.transpose(1, 3)  # channel first
    return inp


def load_body_from_step(step_file):
    """
    Load the body from the step file.
    We expect only one body in each file
    """
    assert pathlib.Path(step_file).suffix in [".step", ".stp", ".STEP", ".STP"]
    reader = STEPControl_Reader()
    reader.ReadFile(str(step_file))
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


# 20251029
def load_json_or_pkl(json_path):  # todo 此函数非常占用内存，峰值80G，稳定在45G
    assert isinstance(json_path, Path)

    # try to load dataset from pickel first
    pkl_path = str(json_path).split(".")[0] + ".pkl"
    if osp.exists(pkl_path):
        return torch.load(pkl_path)
    else:  # if no pkl exists, load from json
        with open(json_path, "r") as fp:
            return json.load(fp)


# 20251104
def load_one_graph(fn, data):
    # Create the graph using the edges and number of nodes
    edges = tuple(data["graph"]["edges"])  # 图的边
    num_nodes = data["graph"]["num_nodes"]  # 图的节点数
    dgl_graph = dgl.graph(data=edges, num_nodes=num_nodes)

    # Convert node attributes to PyTorch tensors and add them to the graph
    node_attributes = data["graph_face_attr"]  # TODO 这里到底存什么？
    node_attributes = np.array(node_attributes)  # (n_nodes, feat_dim)
    node_attributes = torch.from_numpy(node_attributes).type(torch.float32)
    dgl_graph.ndata["x"] = node_attributes

    # Convert and add node grid attributes if they are present
    node_grid_attributes = data["graph_face_grid"]  # TODO 这里到底存什么？
    if len(node_grid_attributes) > 0:
        node_grid_attributes = np.array(node_grid_attributes)
        node_grid_attributes = torch.from_numpy(node_grid_attributes).type(
            torch.float32
        )  # shape = (n_nodes, grid_dim, grid_u, grid_v) (33,7,5,5)
        dgl_graph.ndata["grid"] = node_grid_attributes

    # Convert edge attributes to PyTorch tensors and add them to the graph
    edge_attributes = data["graph_edge_attr"]  # TODO 这里到底存什么？
    edge_attributes = np.array(edge_attributes)
    edge_attributes = torch.from_numpy(edge_attributes).type(torch.float32)
    dgl_graph.edata["x"] = edge_attributes

    # Convert and add edge grid attributes if they are present
    edge_grid_attributes = data["graph_edge_grid"]  # TODO 这里到底存什么？
    if len(edge_grid_attributes) > 0:  # todo 存在不存在的情况
        edge_grid_attributes = np.array(edge_grid_attributes)
        edge_grid_attributes = torch.from_numpy(edge_grid_attributes).type(
            torch.float32
        )
        dgl_graph.edata["grid"] = edge_grid_attributes

    sample = {"graph": dgl_graph, "filename": fn}

    return sample


# 20251029
def load_statistics(stat_path):  # TODO 统计值哪里来的？
    assert isinstance(stat_path, Path)

    stat = load_json_or_pkl(stat_path)
    mean_face_attr = np.array(stat["mean_face_attr"])
    std_face_attr = np.array(stat["std_face_attr"])
    mean_edge_attr = np.array(stat["mean_edge_attr"])
    std_edge_attr = np.array(stat["std_edge_attr"])

    stat["mean_face_attr"] = torch.from_numpy(mean_face_attr)
    stat["std_face_attr"] = torch.from_numpy(std_face_attr)
    stat["mean_edge_attr"] = torch.from_numpy(mean_edge_attr)
    stat["std_edge_attr"] = torch.from_numpy(std_edge_attr)

    # TODO 是否合理
    # if the std is 0, we set the std to 1
    eps = 1e-8
    stat["std_face_attr"][stat["std_face_attr"] < eps] = 1.0
    stat["std_edge_attr"][stat["std_edge_attr"] < eps] = 1.0

    return stat


# 20251103 陈守玉
def filter_filenames_by_ids(
    filenames: List[Path],
    ids: set,
    index_width: int = 8,
    prefix: str = "graphs_",
    suffix: str = ".json",
) -> List[Path]:
    """
    根据给定的 id 列表，从文件名列表中过滤出对应的文件按 id 顺序返回；
    同时断言所有 id 都能在文件名中找到对应文件。

    约定：目标文件名形如 f"{prefix}{id.zfill(index_width)}{suffix}"。
    """
    assert isinstance(filenames, list)
    assert isinstance(ids, set)
    assert isinstance(index_width, int) and index_width > 0
    assert isinstance(prefix, str) and prefix is not None
    assert isinstance(suffix, str) and suffix is not None

    # 建立 name -> Path 的快速索引
    name_to_path = {p.name: p for p in filenames}

    def to_padded_id(x: Union[str, int]) -> str:
        # 允许传入数字或数字字符串
        assert isinstance(x, np.str_)

        s = str(x).strip()
        assert s.isdigit()

        return s.zfill(index_width)

    selected: List[Path] = []
    missing: List[str] = []
    for _id in ids:
        pid = to_padded_id(_id)
        fname = f"{prefix}{pid}{suffix}"
        path = name_to_path.get(fname)
        if path is None:
            missing.append(str(_id))
        else:
            selected.append(path)

    print(
        f">>> There are {len(missing)} ids not found in filenames, examples: {missing[:10]}"
    )
    # 训练集 len(selected)=41382 len(missing)=384
    # 验证集 len(selected)=8877 len(missing)=73
    return selected
