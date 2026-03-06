from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="读取目标目录下的 JSON 文件，按比例划分 train/val/test，并输出对应 txt 名单。"
	)
	parser.add_argument(
		"input_dir",
		type=Path,
		help="包含 json 文件的目标目录",
	)
	parser.add_argument(
		"output_dir",
		type=Path,
		help="保存 train.txt / val.txt / test.txt 的目录",
	)
	parser.add_argument(
		"--train-ratio",
		type=float,
		default=0.8,
		help="训练集比例，默认 0.7",
	)
	parser.add_argument(
		"--val-ratio",
		type=float,
		default=0.1,
		help="验证集比例，默认 0.15",
	)
	parser.add_argument(
		"--test-ratio",
		type=float,
		default=0.1,
		help="测试集比例，默认 0.15",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="随机种子，默认 42",
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="是否递归扫描子目录中的 json 文件",
	)
	return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
	ratios = (train_ratio, val_ratio, test_ratio)
	if any(r < 0 for r in ratios):
		raise ValueError("比例不能为负数")
	total = sum(ratios)
	if abs(total - 1.0) > 1e-8:
		raise ValueError(f"比例之和必须为 1.0，当前为 {total:.6f}")


def collect_json_filenames(input_dir: Path, recursive: bool = False) -> List[str]:
	if not input_dir.exists() or not input_dir.is_dir():
		raise FileNotFoundError(f"输入目录不存在或不是目录: {input_dir}")

	pattern = "**/*.json" if recursive else "*.json"
	files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
	return [p.name for p in files]


def split_names(
	names: Sequence[str],
	train_ratio: float,
	val_ratio: float,
	test_ratio: float,
	seed: int,
) -> Tuple[List[str], List[str], List[str]]:
	validate_ratios(train_ratio, val_ratio, test_ratio)

	all_names = list(names)
	random.Random(seed).shuffle(all_names)

	total = len(all_names)
	train_count = int(total * train_ratio)
	val_count = int(total * val_ratio)
	test_count = total - train_count - val_count

	train_names = all_names[:train_count]
	val_names = all_names[train_count:train_count + val_count]
	test_names = all_names[train_count + val_count:train_count + val_count + test_count]

	return train_names, val_names, test_names


def write_lines(path: Path, lines: Sequence[str]) -> None:
	content = "\n".join(lines)
	if content:
		content += "\n"
	path.write_text(content, encoding="utf-8")


def main() -> None:
	args = parse_args()

	names = collect_json_filenames(args.input_dir, recursive=args.recursive)
	if not names:
		raise RuntimeError(f"未在目录中找到 json 文件: {args.input_dir}")

	train_names, val_names, test_names = split_names(
		names=names,
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		seed=args.seed,
	)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	write_lines(args.output_dir / "train.txt", train_names)
	write_lines(args.output_dir / "val.txt", val_names)
	write_lines(args.output_dir / "test.txt", test_names)

	print(f"总 json 数量: {len(names)}")
	print(f"train: {len(train_names)} -> {(args.output_dir / 'train.txt')}")
	print(f"val  : {len(val_names)} -> {(args.output_dir / 'val.txt')}")
	print(f"test : {len(test_names)} -> {(args.output_dir / 'test.txt')}")


if __name__ == "__main__":
	main()
