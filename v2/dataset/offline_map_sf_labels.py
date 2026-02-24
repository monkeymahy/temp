"""离线执行 SF 标签映射。

用法示例：
    python v2\dataset\offline_map_sf_labels.py --root_dir C:\Data\SF-JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
import os
import sys
import numpy as np

# 计算工程根目录（v2 的上两级），用于支持直接运行本文件时也能正确 import
parent_dir = os.path.abspath(  # 得到规范化后的绝对路径
    os.path.join(  # 拼接路径片段
        os.path.dirname(__file__),  # 当前文件所在目录
        os.pardir,  # 上一级
        os.pardir,  # 再上一级
    )
)
if parent_dir not in sys.path:  # 避免重复插入 sys.path
    sys.path.append(parent_dir)  # 将工程根目录加入模块搜索路径


from v2.utils.data_utils import map_sf_labels


def iter_label_files(labels_dir: Path) -> Iterable[Path]:
    """返回 labels 目录下所有 json 标签文件。"""
    return sorted(labels_dir.rglob("*.json"))


def process_one_file(src_path: Path, dst_path: Path, lut: np.ndarray) -> int:
    """处理单个标签文件，返回标签数量。"""
    with open(src_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    mapped = map_sf_labels(labels_data, lut=lut)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(mapped.tolist(), f, ensure_ascii=False)

    return int(mapped.size)


def main() -> None:
    parser = argparse.ArgumentParser(description="离线执行SF标签映射")
    parser.add_argument("--root_dir", type=str, required=True, help="数据根目录（包含 labels/）")
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="labels",
        help="标签目录相对路径（相对于 root_dir，默认 labels）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="labels_mapped",
        help="输出目录相对路径（相对于 root_dir，默认 labels_mapped）",
    )
    parser.add_argument(
        "--lut",
        type=int,
        nargs="+",
        default=[2, 0, 0, 1, 0, 0],
        help="标签映射查找表，例如 --lut 2 0 0 1 0 0",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    labels_dir = root_dir / args.labels_dir

    if not labels_dir.exists():
        raise FileNotFoundError(f"labels 目录不存在: {labels_dir}")

    dst_root = root_dir / args.output_dir

    lut = np.asarray(args.lut, dtype=np.int32)

    files = list(iter_label_files(labels_dir))
    if len(files) == 0:
        raise FileNotFoundError(f"未在目录中找到任何 json 标签文件: {labels_dir}")

    total_files = 0
    total_labels = 0

    for src in files:
        rel = src.relative_to(labels_dir)
        dst = dst_root / rel
        total_labels += process_one_file(src_path=src, dst_path=dst, lut=lut)
        total_files += 1

    print("=== Offline SF Label Mapping Done ===")
    print(f"root_dir    : {root_dir}")
    print(f"source_dir  : {labels_dir}")
    print(f"target_dir  : {dst_root}")
    print(f"files       : {total_files}")
    print(f"label_count : {total_labels}")
    print(f"lut         : {lut.tolist()}")


if __name__ == "__main__":
    main()
