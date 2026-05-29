import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from v2.utils.instance_label_utils import adj_to_instance_payload


def _labels_from_seg(seg: Any) -> List[int]:
    if isinstance(seg, dict):
        return [int(label) for _, label in sorted(seg.items(), key=lambda item: int(item[0]))]
    if isinstance(seg, list):
        return [int(label) for label in seg]
    raise ValueError("seg must be a dict or list")


def _extract_mfinstseg_label_dict(data: Any) -> Optional[Dict[str, Any]]:
    if isinstance(data, dict) and "seg" in data:
        return data
    if isinstance(data, list):
        if len(data) == 2 and isinstance(data[1], dict) and "seg" in data[1]:
            return data[1]
        if data and isinstance(data[0], list) and len(data[0]) == 2:
            item = data[0][1]
            if isinstance(item, dict) and "seg" in item:
                return item
    return None


def load_train_label(path: Path) -> Tuple[List[int], Optional[Dict[str, Any]], str]:
    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    label_dict = _extract_mfinstseg_label_dict(data)
    if label_dict is not None:
        labels = _labels_from_seg(label_dict["seg"])
        instance_payload = None
        if "inst" in label_dict:
            instance_payload = adj_to_instance_payload(label_dict["inst"], labels, background_labels={0})
        return labels, instance_payload, "mfinstseg_seg_inst"

    if isinstance(data, list):
        return [int(label) for label in data], None, "train_label_list"
    raise ValueError(f"unsupported train label format: {path}")


def build_full_label_payload(
    labels: List[int],
    source_path: Path,
    author: str,
    source_format: str,
    instance_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    labels = [int(label) for label in labels]
    payload = {
        "labels": labels,
        "labels_base": list(labels),
        "version_id": 1,
        "versions": [],
        "domains": {
            "geometry": {
                "face": list(labels),
            }
        },
        "source_format": source_format,
        "source_label_path": str(source_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "generated_by": author,
    }
    if instance_payload is not None:
        instance = {
            "face_instance": [int(value) for value in instance_payload["face_instance"]],
            "instances": [dict(item) for item in instance_payload["instances"]],
        }
        payload["domains"]["instance"] = instance
        payload["domains"]["instance_base"] = {
            "face_instance": list(instance["face_instance"]),
            "instances": [dict(item) for item in instance["instances"]],
        }
    return payload


def collect_label_files(labels_dir: Path) -> List[Path]:
    return sorted(labels_dir.glob("*.json")) + sorted(labels_dir.glob("*.pkl"))


def convert_labels(
    *,
    labels_dir: Path,
    output_dir: Path,
    overwrite: bool,
    author: str,
) -> None:
    if not labels_dir.exists():
        raise FileNotFoundError(f"train labels dir not found: {labels_dir}")

    label_files = collect_label_files(labels_dir)
    if not label_files:
        raise FileNotFoundError(f"no json/pkl train labels found in: {labels_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    for label_path in label_files:
        labels, instance_payload, source_format = load_train_label(label_path)
        payload = build_full_label_payload(
            labels,
            source_path=label_path,
            author=author,
            source_format=source_format,
            instance_payload=instance_payload,
        )

        out_path = output_dir / f"{label_path.stem}.json"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"output label exists: {out_path}")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

        items.append(
            {
                "sample_id": label_path.stem,
                "source_label_path": str(label_path),
                "label_path": str(out_path),
                "version_id": 1,
                "num_faces": len(labels),
                "source_format": source_format,
            }
        )

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "source_labels_dir": str(labels_dir),
        "labels_full_dir": str(output_dir),
        "count": len(items),
        "items": items,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"output manifest exists: {manifest_path}")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create labels_full files from pure list train labels.")
    parser.add_argument("--labels-dir", required=True, help="Directory containing pure list train labels.")
    parser.add_argument("--output-dir", required=True, help="Output labels_full directory.")
    parser.add_argument("--author", default="generated", help="Author name recorded in full label payload.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing full label files.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_labels(
        labels_dir=Path(args.labels_dir),
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        author=args.author,
    )


if __name__ == "__main__":
    main()
