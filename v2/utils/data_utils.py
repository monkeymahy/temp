from pathlib import Path  # 使用 Path 表达文件路径（更稳健、跨平台）
import random  # 用于随机旋转的数据增强
from typing import List, Union, Iterable, Optional, Dict, Any  # 类型标注（提升可读性与 IDE 体验）
import numpy as np  # 数值处理（均值方差、数组转换等）
import torch  # 张量计算（与 DGL 图特征存储一致）
import dgl  # DGL 图构建与图数据容器
from scipy.spatial.transform import Rotation  # 用于生成/表示三维旋转
import json  # 读取 json 文件
import os.path as osp  # 兼容性更好的路径判断（exists 等）


def map_sf_labels(labels: Union[List[int], np.ndarray], lut: Optional[np.ndarray] = None) -> np.ndarray:
    """将 SF 原始标签映射到训练标签空间，返回 np.int32 数组。"""
    labels_np = np.asarray(labels, dtype=np.int32)
    if lut is None:
        lut = np.asarray([2, 0, 0, 1, 0, 0], dtype=np.int32)
    else:
        lut = np.asarray(lut, dtype=np.int32)
    return lut[labels_np]


def load_one_graph(fn: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """从单个样本的 json 数据构建 DGLGraph，并填充节点/边特征。"""
    # 读取边（DGL 需要 (src, dst) 两个序列或等价结构）
    edges = tuple(data["graph"]["edges"])  # 图的边（期望形如 [src_list, dst_list]）
    # 读取节点数（用于 DGLGraph 初始化）
    num_nodes = int(data["graph"]["num_nodes"])  # 图的节点数（显式转 int 更稳健）
    # 构建 DGL 图对象（仅包含拓扑结构）
    dgl_graph = dgl.graph(data=edges, num_nodes=num_nodes)  # 创建 DGLGraph

    # ---- 节点属性：face attr ----
    node_attributes = data["graph_face_attr"]  # 节点属性（每个 face 的特征向量） # TODO 这里存什么？
    node_attributes = np.asarray(node_attributes)  # 转为 numpy（形如 (n_nodes, feat_dim)）
    node_attributes = torch.from_numpy(node_attributes).to(dtype=torch.float32)  # 转为 float32 Tensor
    dgl_graph.ndata["x"] = node_attributes  # 写入节点特征（约定键名为 "x"）

    # ---- 节点网格：face grid（可选）----
    # 如果存在节点网格特征，则进行转换并添加
    node_grid_attributes = data["graph_face_grid"]  # 节点网格特征（可能为空） # TODO 这里存什么？
    if len(node_grid_attributes) > 0:  # 若存在网格特征则加载（避免空列表转 tensor）
        node_grid_attributes = np.asarray(node_grid_attributes)  # 转为 numpy
        node_grid_attributes = torch.from_numpy(node_grid_attributes).to(dtype=torch.float32)  # 转为 float32
        dgl_graph.ndata["grid"] = node_grid_attributes  # 写入节点网格（约定键名为 "grid"）

    # ---- 边属性：edge attr ----
    edge_attributes = data["graph_edge_attr"]  # 边属性特征（每条边的特征向量）# TODO 这里存什么？
    edge_attributes = np.asarray(edge_attributes)  # 转为 numpy
    edge_attributes = torch.from_numpy(edge_attributes).to(dtype=torch.float32)  # 转为 float32 Tensor
    dgl_graph.edata["x"] = edge_attributes  # 写入边特征（约定键名为 "x"）

    # ---- 边网格：edge grid（可选）----
    edge_grid_attributes = data["graph_edge_grid"]  # 边网格特征（可能为空）# TODO 这里存什么？
    if len(edge_grid_attributes) > 0:  # 仅当存在时才写入（避免键存在但为空导致后续误用）# TODO 这里存什么？
        edge_grid_attributes = np.asarray(edge_grid_attributes)  # 转为 numpy
        edge_grid_attributes = torch.from_numpy(edge_grid_attributes).to(dtype=torch.float32)  # 转为 float32
        dgl_graph.edata["grid"] = edge_grid_attributes  # 写入边网格（约定键名为 "grid"）

    # 将图与文件名封装成统一 sample（与 Dataset 侧逻辑保持一致）
    sample = {"graph": dgl_graph, "filename": fn}  # 统一返回结构

    return sample  # 返回一个样本字典（包含 graph/filename）


def load_statistics(stat_path: Path) -> Dict[str, torch.Tensor]:
    """加载标准化所需统计量（均值/方差），并转换为 torch.Tensor。"""
    if not isinstance(stat_path, Path):  # 显式类型检查（避免 assert 在 -O 下失效）
        raise TypeError(f"stat_path 必须是 pathlib.Path，但得到：{type(stat_path)}")  # 抛出明确错误

    stat = load_json_or_pkl(stat_path)  # 读取统计量（json 或 pkl）

    # 将 numpy 数组转为 torch.Tensor（后续可直接用于张量计算）
    mean_face_attr = np.asarray(stat["mean_face_attr"])  # 节点特征均值
    std_face_attr = np.asarray(stat["std_face_attr"])  # 节点特征标准差
    mean_edge_attr = np.asarray(stat["mean_edge_attr"])  # 边特征均值
    std_edge_attr = np.asarray(stat["std_edge_attr"])  # 边特征标准差

    stat["mean_face_attr"] = torch.from_numpy(mean_face_attr)  # 转 tensor（默认 CPU）
    stat["std_face_attr"] = torch.from_numpy(std_face_attr)  # 转 tensor（默认 CPU）
    stat["mean_edge_attr"] = torch.from_numpy(mean_edge_attr)  # 转 tensor（默认 CPU）
    stat["std_edge_attr"] = torch.from_numpy(std_edge_attr)  # 转 tensor（默认 CPU）

    # 防止 std 为 0 造成除零（数值更稳定）
    eps = 1e-8  # 小阈值（用于判断“接近 0”） # TODO 是否合理
    stat["std_face_attr"][stat["std_face_attr"] < eps] = 1.0  # 节点 std 过小则置 1
    stat["std_edge_attr"][stat["std_edge_attr"] < eps] = 1.0  # 边 std 过小则置 1

    return stat  # 返回统计量字典（tensor 版本）


def standardization(data: Dict[str, Any], stat: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """对节点/边特征做标准化：x = (x - mean) / std（原地操作，减少额外内存）。"""
    # 节点特征标准化（原地减/除，避免额外分配）
    data["graph"].ndata["x"].sub_(stat["mean_face_attr"])  # x -= mean（in-place）
    data["graph"].ndata["x"].div_(stat["std_face_attr"])  # x /= std（in-place）
    # 边特征标准化（原地减/除）
    data["graph"].edata["x"].sub_(stat["mean_edge_attr"])  # x -= mean（in-place）
    data["graph"].edata["x"].div_(stat["std_edge_attr"])  # x /= std（in-place）
    return data  # 返回同一个 data（便于链式调用）


def bounding_box_pointcloud(pts: torch.Tensor) -> torch.Tensor:
    """计算点云 AABB（轴对齐包围盒），返回形如 [[xmin,ymin,zmin],[xmax,ymax,zmax]]。"""
    # 取出 x/y/z 分量（便于分别求 min/max）
    x = pts[:, 0]  # x 坐标
    y = pts[:, 1]  # y 坐标
    z = pts[:, 2]  # z 坐标
    # 构建包围盒（Python list 组织，随后转 tensor）
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]  # AABB 两个角点
    return pts.new_tensor(box)  # 用 new_tensor 保持 device/dtype（避免 CPU/GPU 混用报错）


def bounding_box_uvgrid(inp: torch.Tensor) -> torch.Tensor:
    """从 uvgrid 中提取有效点并计算 AABB（约定 mask 通道位于第 7 维，即索引 6）。"""
    # 将所有点展平为 (N, 3)，只取前三个通道作为点坐标
    pts = inp[..., :3].reshape((-1, 3))  # (N, 3)
    # 取出 mask（约定第 7 个通道为 mask）
    mask = inp[..., 6].reshape(-1)  # (N,)
    # 选出 mask==1 的点（表示“在面内”的采样点）
    point_indices_inside_faces = mask == 1  # bool 索引
    pts = pts[point_indices_inside_faces, :]  # 过滤无效点
    return bounding_box_pointcloud(pts)  # 计算过滤后点云的 AABB


def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale: bool = False):
    """对 uvgrid 做中心化与缩放，使其落在近似 [-1, 1] 范围（返回变换后的 inp）。"""
    # 将通道维从 (C, U, V) 转为 (U, V, C) 形式以便处理坐标（保持原逻辑）
    inp = inp.transpose(1, 3)  # channel last
    # 计算 AABB（用于得到中心与尺度）
    bbox = bounding_box_uvgrid(inp)  # [[min],[max]]
    # 包围盒对角线长度（用于估计尺度）
    diag = bbox[1] - bbox[0]  # (3,)
    # 使用最大边长做统一缩放（避免各向异性缩放）
    scale = 2.0 / max(diag[0], diag[1], diag[2])  # 标准化尺度（使最大边长约为 2）
    # 中心点（AABB 中心）
    center = 0.5 * (bbox[0] + bbox[1])  # (3,)
    # 对坐标通道做中心化与缩放（仅影响前三维坐标）
    inp[..., :3] -= center  # 平移到原点附近
    inp[..., :3] *= scale  # 缩放到期望尺度
    # 还原通道维布局（保持与模型侧约定一致）
    inp = inp.transpose(1, 3)  # channel first
    # 如果需要，也返回 center/scale 供边网格同步变换
    if return_center_scale:  # 需要返回中心与尺度
        return inp, center, scale  # 返回三元组
    return inp  # 必须返回 inp（原实现返回 None 会导致调用方出错）


def center_and_scale(data: Dict[str, Any]) -> Dict[str, Any]:
    """对样本中的节点/边 grid 做一致的中心化与缩放（保证节点/边在同一坐标系）。"""
    # 节点 grid 必须存在，否则这里会 KeyError；上层通常通过开关控制是否调用
    data["graph"].ndata["grid"], center, scale = center_and_scale_uvgrid(  # 节点网格归一化
        data["graph"].ndata["grid"],  # 节点 grid
        return_center_scale=True,  # 需要 center/scale 用于边同步
    )
    # 边 grid 可选：存在则做同样的平移/缩放
    if "grid" in data["graph"].edata.keys():  # 判断边是否含 grid
        egrid = data["graph"].edata["grid"]  # 取出边 grid
        egrid = egrid.transpose(1, 2)  # 调整维度布局（保持原作者逻辑）
        egrid[..., :3] -= center  # 使用相同 center 平移
        egrid[..., :3] *= scale  # 使用相同 scale 缩放
        egrid = egrid.transpose(1, 2)  # 还原维度布局
        data["graph"].edata["grid"] = egrid  # 写回边 grid
    return data  # 返回样本字典（便于链式调用）


def get_random_rotation() -> Rotation:
    """获取一个 90°倍数的随机旋转（绕 x/y/z 轴之一）。"""
    axes = [  # 三个规范轴
        np.array([1, 0, 0]),  # x 轴
        np.array([0, 1, 0]),  # y 轴
        np.array([0, 0, 1]),  # z 轴
    ]
    angles = [0.0, 90.0, 180.0, 270.0]  # 离散角度集合（单位：度）
    axis = random.choice(axes)  # 随机选轴
    angle_radians = np.radians(random.choice(angles))  # 随机选角并转弧度
    return Rotation.from_rotvec(angle_radians * axis)  # 轴角表示构造 Rotation


def rotate_uvgrid(inp: torch.Tensor, rotation: Rotation) -> torch.Tensor:
    """对 uvgrid 中的坐标/法向等向量通道施加旋转（保持 dtype/device 一致）。"""
    # 将通道维转到末尾，方便把向量通道当作最后一维处理
    inp = inp.transpose(1, 3)  # channel last
    # 将旋转矩阵转为与 inp 同设备/同 dtype 的张量（避免 CPU/GPU 混用错误）
    rmat_np = rotation.as_matrix().astype(np.float32)  # (3, 3) numpy float32
    Rmat = torch.from_numpy(rmat_np).to(device=inp.device, dtype=inp.dtype)  # (3, 3) tensor
    # 记录原始形状（用于 reshape 回来）
    orig_size = inp[..., :3].size()  # (..., 3) 的形状
    # 旋转坐标向量（前三维）
    inp[..., :3] = torch.mm(inp[..., :3].reshape(-1, 3), Rmat).reshape(orig_size)  # Points
    # 旋转法向/切向（假设 3:6 为向量通道；与原实现一致）
    inp[..., 3:6] = torch.mm(inp[..., 3:6].reshape(-1, 3), Rmat).reshape(orig_size)  # Normals/tangents
    # 恢复通道维布局（与模型侧约定一致）
    inp = inp.transpose(1, 3)  # channel first
    return inp  # 返回旋转后的 grid


def filter_filenames_by_ids_9s(
    filenames: List[Path],  # 全量文件路径列表（通常来自 rglob）
    ids: Iterable[Union[str, int]],  # 目标 id 列表/集合（建议传 list 以保持顺序）
    index_width: int = 8,  # 数字 id 的补零宽度（例如 1 -> 00000001）
    prefix: str = "graphs_",  # 文件名前缀（例如 graphs_00000001.json）
    suffix: str = ".json",  # 文件名后缀（例如 .json）
    strict: bool = False,  # 是否严格校验：有缺失则直接抛错（默认兼容当前“只打印不报错”）
) -> List[Path]:
    """（兼容九韶提供的数据）
    根据给定的 id 列表，从文件名列表中过滤出对应的文件并返回。

    兼容规则（按优先级）：
    1) 若 id 本身已包含后缀（如 xxx.json），直接用它当作文件名；
    2) 若 id 以 prefix 开头（如 graphs_00000001），拼接 suffix；
    3) 否则若 id 为纯数字字符串/数字，按 index_width 补零并加 prefix+suffix。
    """
    # ---- 参数检查：用显式异常替代 assert（更稳定）----
    if not isinstance(filenames, list):  # filenames 必须为 list[Path]
        raise TypeError(f"filenames 必须是 list，但得到：{type(filenames)}")  # 明确错误信息
    if not isinstance(index_width, int) or index_width <= 0:  # index_width 必须为正整数
        raise ValueError(f"index_width 必须是 >0 的 int，但得到：{index_width!r}")  # 明确错误信息
    if not isinstance(prefix, str) or prefix is None:  # prefix 必须为字符串
        raise TypeError("prefix 必须是 str 且不能为 None")  # 明确错误信息
    if not isinstance(suffix, str) or suffix is None:  # suffix 必须为字符串
        raise TypeError("suffix 必须是 str 且不能为 None")  # 明确错误信息

    # 建立 name -> Path 的索引（O(1) 查找；比反复遍历 filenames 更快）
    name_to_path = {p.name: p for p in filenames}  # 以文件名为键（不含目录）

    def _id_to_filename(x: Union[str, int]) -> str:
        """将各种形式的 id 统一映射到最终文件名（含 suffix）。"""
        s = str(x).strip()  # 转字符串并去空白
        if not s:  # 空字符串属于非法输入
            raise ValueError("id 不能为空")  # 明确提示
        if s.endswith(suffix):  # 情况1：已包含后缀
            return s  # 直接使用
        if s.startswith(prefix):  # 情况2：已包含前缀
            return f"{s}{suffix}"  # 直接补后缀
        if s.isdigit():  # 情况3：纯数字 id
            s = s.zfill(index_width)  # 补零到固定宽度
            return f"{prefix}{s}{suffix}"  # 组装完整文件名
        # 情况4：非纯数字但也不含 prefix/suffix，则按“原样+suffix”兜底（更兼容）
        return f"{s}{suffix}"  # 兼容 id 已经是完整 basename 但不带后缀的情况

    # 将 ids 转为 list（若传入 set，可排序以获得确定性输出）
    ids_list = list(ids)  # 统一物化一次（避免迭代器被消耗）
    if isinstance(ids, set):  # 若是 set，则原生无序
        ids_list = sorted(ids_list, key=lambda v: str(v))  # 排序保证输出稳定（便于复现实验）

    selected: List[Path] = []  # 存放命中的 Path
    missing: List[str] = []  # 存放缺失的 id（便于诊断）
    for _id in ids_list:  # 按 ids_list 的顺序筛选
        fname = _id_to_filename(_id)  # 将 id 映射到文件名（含后缀）
        path = name_to_path.get(fname)  # 通过索引快速定位 Path
        if path is None:  # 没找到对应文件
            missing.append(str(_id))  # 记录缺失 id
        else:
            selected.append(path)  # 记录命中文件

    # 输出缺失统计（保留你原来的行为：打印而不是强制报错）
    print(f">>> 有{len(missing)}个 id 未在文件名中找到，示例：{missing[:10]}")  # 仅打印前 10 个样例

    # 可选严格模式：用于你希望“强一致性”时尽早失败定位问题
    if strict and len(missing) > 0:  # 若开启 strict 且存在缺失
        raise FileNotFoundError(f"有 {len(missing)} 个 id 未找到对应文件，例如：{missing[:10]}")  # 抛出明确异常

    return selected  # 返回命中的 Path 列表（顺序与 ids_list 对齐）


def load_json_or_pkl(json_path: Path) -> Any:  # todo 此函数非常占用内存，峰值80G，稳定在45G
    """优先从同名 .pkl 加载，否则从 .json 加载（减少重复解析 json 的开销）。"""
    if not isinstance(json_path, Path):  # 显式类型检查（避免 assert 在 -O 下失效）
        raise TypeError(f"json_path必须是pathlib.Path，但得到：{type(json_path)}")  # 明确错误信息

    # 推导同名 pkl 路径（比 split('.') 更稳健：不会被多重后缀干扰）
    pkl_path = json_path.with_suffix(".pkl")  # 形如 xxx.pkl
    if osp.exists(str(pkl_path)):  # 若 pkl 存在则优先加载
        return torch.load(str(pkl_path), map_location="cpu")  # 强制加载到 CPU（避免无意占用 GPU 显存）
    # 若不存在 pkl，则回退读取 json（注意：大文件会慢且占内存）
    with open(json_path, "r", encoding="utf-8") as fp:  # 显式编码更稳健
        return json.load(fp)  # 解析并返回 Python 对象
