import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List


def load_label_list(path: Path) -> List[int]:
    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"train label must be a list: {path}")
    return [int(label) for label in data]


def build_full_label_payload(labels: List[int], source_path: Path, author: str) -> Dict[str, Any]:
    labels = [int(label) for label in labels]
    return {
        "labels": labels,
        "labels_base": list(labels),
        "version_id": 1,
        "versions": [],
        "domains": {
            "geometry": {
                "face": list(labels),
            }
        },
        "source_format": "train_label_list",
        "source_label_path": str(source_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "generated_by": author,
    }


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
        labels = load_label_list(label_path)
        payload = build_full_label_payload(labels, source_path=label_path, author=author)

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
