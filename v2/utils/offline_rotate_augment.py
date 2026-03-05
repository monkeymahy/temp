import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from data_utils import filter_filenames_by_ids_9s


def parse_splits(value: str) -> List[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


RotationSpec = Tuple[str, float, Rotation]


def build_discrete_rotations() -> List[RotationSpec]:
    rotations: List[RotationSpec] = []
    for axis, angle in [("x", 90), ("x", 180), ("x", 270), ("y", 90), ("y", 180), ("y", 270), ("z", 90), ("z", 180), ("z", 270)]:
        rotations.append((axis, float(angle), rotation_from_axis_angle(axis, float(angle))))
    return rotations


def rotation_from_axis_angle(axis: str, angle_deg: float) -> Rotation:
    axis_map = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    vec = axis_map[axis]
    return Rotation.from_rotvec(np.radians(float(angle_deg)) * vec)


def build_random_rotations(
    num_aug: int,
    angle_min: float,
    angle_max: float,
    axis: str,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> List[RotationSpec]:
    rotations: List[RotationSpec] = []
    for _ in range(num_aug):
        angle = rng.uniform(angle_min, angle_max)
        if axis in {"x", "y", "z"}:
            axis_tag = axis
            rotation = rotation_from_axis_angle(axis, angle)
        else:
            axis_vec = np_rng.normal(size=3).astype(np.float32)
            norm = np.linalg.norm(axis_vec)
            if norm == 0.0:
                axis_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                axis_vec = axis_vec / norm
            axis_tag = "any"
            rotation = Rotation.from_rotvec(np.radians(float(angle)) * axis_vec)
        rotations.append((axis_tag, float(angle), rotation))
    return rotations


def rotate_uvgrid_np(grid: np.ndarray, rotation: Rotation) -> np.ndarray:
    if grid.size == 0:
        return grid
    if grid.ndim != 4:
        raise ValueError(f"Expected grid with 4 dims (N, C, U, V), got shape {grid.shape}")

    grid = np.transpose(grid, (0, 2, 3, 1))
    rmat = rotation.as_matrix().astype(np.float32)
    orig_size = grid[..., :3].shape

    coords = grid[..., :3].reshape(-1, 3) @ rmat
    normals = grid[..., 3:6].reshape(-1, 3) @ rmat

    grid[..., :3] = coords.reshape(orig_size)
    grid[..., 3:6] = normals.reshape(orig_size)
    grid = np.transpose(grid, (0, 3, 1, 2))

    return grid


def load_split_ids(split_file: Path) -> List[str]:
    text = split_file.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def save_split_ids(split_file: Path, ids: Iterable[str]) -> None:
    content = "\n".join(ids) + "\n"
    split_file.write_text(content, encoding="utf-8")


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)


def merge_ids_keep_order(existing_ids: List[str], new_ids: List[str]) -> List[str]:
    seen = set(existing_ids)
    merged = list(existing_ids)
    for item in new_ids:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.write_bytes(src.read_bytes())


def augment_one_file(
    graph_path: Path,
    label_dir: Path,
    out_aag_dir: Path,
    out_label_dir: Path,
    rotations: List[RotationSpec],
    copy_original: bool,
) -> List[str]:
    with open(graph_path, "r", encoding="utf-8") as f:
        fn, data = json.load(f)

    label_path = label_dir / f"{fn}.json"
    if not label_path.exists():
        return []

    out_ids: List[str] = []

    if copy_original:
        out_graph_path = out_aag_dir / f"{fn}.json"
        out_graph_path.write_text(json.dumps([fn, data]), encoding="utf-8")

        out_label_path = out_label_dir / f"{fn}.json"
        out_label_path.write_bytes(label_path.read_bytes())

        out_ids.append(fn)

    face_grid = data.get("graph_face_grid", [])
    edge_grid = data.get("graph_edge_grid", [])

    face_grid_np = np.asarray(face_grid, dtype=np.float32)
    edge_grid_np = np.asarray(edge_grid, dtype=np.float32)

    for idx, (axis, angle, rotation) in enumerate(rotations):

        new_data = dict(data)
        if face_grid_np.size > 0:
            new_face_grid = rotate_uvgrid_np(face_grid_np.copy(), rotation)
            new_data["graph_face_grid"] = new_face_grid.tolist()
        if edge_grid_np.size > 0:
            new_edge_grid = rotate_uvgrid_np(edge_grid_np.copy(), rotation)
            new_data["graph_edge_grid"] = new_edge_grid.tolist()

        angle_tag = f"{angle:.1f}".replace(".", "p")
        new_fn = f"{fn}__rot_{axis}{angle_tag}_{idx:02d}"
        out_graph_path = out_aag_dir / f"{new_fn}.json"
        out_graph_path.write_text(json.dumps([new_fn, new_data]), encoding="utf-8")

        out_label_path = out_label_dir / f"{new_fn}.json"
        out_label_path.write_bytes(label_path.read_bytes())

        out_ids.append(new_fn)

    return out_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline rotation augmentation for SF AAG data.")
    parser.add_argument("--input-root", required=True, help="Dataset root directory (has aag/ and labels/).")
    parser.add_argument("--output-root", required=True, help="Output directory for augmented dataset.")
    parser.add_argument("--splits", default="train", help="Comma-separated list: train,val,test")
    parser.add_argument("--mode", choices=["random-angle", "discrete", "all"], default="random-angle")
    parser.add_argument("--num-aug", type=int, default=3, help="Number of rotations per sample (random-angle/discrete).")
    parser.add_argument("--angle-min", type=float, default=0.0, help="Min angle in degrees for random-angle mode.")
    parser.add_argument("--angle-max", type=float, default=360.0, help="Max angle in degrees for random-angle mode.")
    parser.add_argument("--axis", choices=["x", "y", "z", "any"], default="any", help="Rotation axis for random-angle.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-original", action="store_true", default=True)
    parser.add_argument("--no-copy-original", dest="copy_original", action="store_false")
    parser.add_argument(
        "--overwrite",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Whether to overwrite existing output directory. Supports --overwrite or --overwrite true/false.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append augmented files/split ids into an existing output directory instead of requiring a fresh one.",
    )
    parser.add_argument("--prefix", default="graphs_", help="File prefix for numeric ids.")
    parser.add_argument("--index-width", type=int, default=8)
    parser.add_argument("--suffix", default=".json")

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if input_root.resolve() == output_root.resolve():
        raise ValueError("input_root and output_root must be different to avoid overwriting.")

    out_aag_dir = output_root / "aag"
    out_label_dir = output_root / "labels"

    if args.append:
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        ensure_empty_dir(output_root, args.overwrite)
    out_aag_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    copy_if_exists(input_root / "aag" / "attr_stat.json", out_aag_dir / "attr_stat.json")

    split_names = parse_splits(args.splits)

    all_rotations = build_discrete_rotations()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    aag_dir = input_root / "aag"
    label_dir = input_root / "labels"

    for split in split_names:
        split_file = input_root / "split" / f"{split}.txt"
        if not split_file.exists():
            continue

        split_ids = load_split_ids(split_file)
        graph_paths = list(aag_dir.rglob("*.json"))
        graph_paths = filter_filenames_by_ids_9s(
            filenames=graph_paths,
            ids=set(split_ids),
            index_width=args.index_width,
            prefix=args.prefix,
            suffix=args.suffix,
        )

        new_ids: List[str] = []
        skipped_no_label = 0
        for graph_path in graph_paths:
            if args.mode == "all":
                rotations = all_rotations
            elif args.mode == "discrete":
                if args.num_aug <= len(all_rotations):
                    rotations = rng.sample(all_rotations, args.num_aug)
                else:
                    rotations = [rng.choice(all_rotations) for _ in range(args.num_aug)]
            else:
                rotations = build_random_rotations(
                    num_aug=args.num_aug,
                    angle_min=args.angle_min,
                    angle_max=args.angle_max,
                    axis=args.axis,
                    rng=rng,
                    np_rng=np_rng,
                )

            generated_ids = augment_one_file(
                graph_path=graph_path,
                label_dir=label_dir,
                out_aag_dir=out_aag_dir,
                out_label_dir=out_label_dir,
                rotations=rotations,
                copy_original=args.copy_original,
            )
            if not generated_ids:
                skipped_no_label += 1
                continue
            new_ids.extend(generated_ids)

        out_split_dir = output_root / "split"
        out_split_dir.mkdir(parents=True, exist_ok=True)
        out_split_file = out_split_dir / f"{split}.txt"
        if args.append and out_split_file.exists():
            existing_ids = load_split_ids(out_split_file)
            merged_ids = merge_ids_keep_order(existing_ids, new_ids)
            save_split_ids(out_split_file, merged_ids)
        else:
            save_split_ids(out_split_file, new_ids)

        print(
            f"[split={split}] input_graphs={len(graph_paths)} generated_ids={len(new_ids)} skipped_no_label={skipped_no_label}"
        )


if __name__ == "__main__":
    main()
