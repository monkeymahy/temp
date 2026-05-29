import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from v2.utils.data_utils import (
    load_json_or_pkl,
    normalize_label_payload,
    rollback_payload,
)
from v2.utils.instance_label_utils import (
    face_instance_to_adj,
    normalize_instance_payload,
    rollback_instance_payload,
    validate_instance_struct,
)


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"samples": data}
    if isinstance(data, dict):
        return data
    raise ValueError("manifest.json format not supported")


def _extract_samples(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples = manifest.get("samples")
    if isinstance(samples, list):
        return samples
    if isinstance(manifest.get("items"), list):
        return manifest.get("items")
    if isinstance(manifest.get("data"), list):
        return manifest.get("data")
    raise ValueError("manifest does not contain samples list")


def _sample_id_from_entry(entry: Dict[str, Any]) -> str:
    for key in ("sample_id", "id", "name", "file"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return Path(value).stem if "." in value else value
    raise ValueError("sample entry missing sample_id")


def _resolve_label_path(labels_full_dir: Path, entry: Dict[str, Any], sample_id: str) -> Path:
    for key in ("label_path", "path"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return Path(value)
    json_path = labels_full_dir / f"{sample_id}.json"
    if json_path.exists():
        return json_path
    pkl_path = labels_full_dir / f"{sample_id}.pkl"
    if pkl_path.exists():
        return pkl_path
    return json_path


def _as_train_label_list(labels: Any, sample_id: str) -> List[int]:
    if not isinstance(labels, list):
        raise ValueError(f"labels for sample {sample_id} must be a list")
    return [int(label) for label in labels]


def _load_class_mapping(mapping_path: Optional[Path]) -> Optional[Dict[int, int]]:
    if mapping_path is None:
        return None
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {int(key): int(value) for key, value in data.items()}
    if isinstance(data, list):
        return {idx: int(value) for idx, value in enumerate(data)}
    raise ValueError("class mapping must be a dict or list")


def _apply_class_mapping(labels: List[int], mapping: Optional[Dict[int, int]]) -> List[int]:
    if mapping is None:
        return [int(label) for label in labels]
    return [int(mapping.get(int(label), int(label))) for label in labels]


def _seg_list_to_dict(labels: List[int]) -> Dict[str, int]:
    return {str(idx): int(label) for idx, label in enumerate(labels)}


def _build_train_payload(
    *,
    task_mode: str,
    labels: List[int],
    instance_payload: Optional[Dict[str, Any]],
    class_mapping: Optional[Dict[int, int]],
    sample_id: str,
) -> Any:
    mapped_labels = _apply_class_mapping(labels, class_mapping)
    if task_mode == "seg_only":
        return mapped_labels
    if task_mode != "seg_inst":
        raise ValueError(f"unsupported task_mode: {task_mode}")
    if instance_payload is None:
        raise ValueError(f"instance label missing for sample {sample_id}")
    errors = validate_instance_struct(instance_payload, len(mapped_labels))
    if errors:
        raise ValueError(f"invalid instance label for {sample_id}: {'; '.join(errors)}")
    return [
        [
            sample_id,
            {
                "seg": _seg_list_to_dict(mapped_labels),
                "inst": face_instance_to_adj(instance_payload["face_instance"]),
            },
        ]
    ]


def export_labels(
    *,
    labels_full_dir: Path,
    manifest_path: Optional[Path],
    export_root: Path,
    export_id: str,
    use_manifest_versions: bool,
    task_mode: str,
    class_mapping: Optional[Dict[int, int]],
    background_labels: List[int],
) -> None:
    if manifest_path is not None:
        manifest = _load_manifest(manifest_path)
        samples = _extract_samples(manifest)
    else:
        samples = None

    labels_dir = export_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    exported_items: List[Dict[str, Any]] = []
    if samples is None:
        label_paths = sorted(labels_full_dir.glob("*.json")) + sorted(labels_full_dir.glob("*.pkl"))
        entries = [
            {"sample_id": path.stem, "label_path": str(path)}
            for path in label_paths
        ]
    else:
        entries = samples

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        sample_id = _sample_id_from_entry(entry)
        label_path = _resolve_label_path(labels_full_dir, entry, sample_id)
        if not label_path.exists():
            raise FileNotFoundError(f"label file not found: {label_path}")
        label_data = load_json_or_pkl(label_path)
        payload = normalize_label_payload(label_data)

        if use_manifest_versions:
            target_version_id = entry.get("version_id", payload.get("version_id", 1))
            try:
                target_version_id = int(target_version_id)
            except Exception:
                target_version_id = int(payload.get("version_id", 1))

            if target_version_id < int(payload.get("version_id", target_version_id)):
                labels = rollback_payload(payload, target_version_id)
                instance_payload = rollback_instance_payload(payload, target_version_id)
            else:
                labels = payload.get("labels")
                instance_payload = normalize_instance_payload(
                    label_data,
                    num_faces=len(labels) if isinstance(labels, list) else None,
                    allow_empty=(task_mode == "seg_only"),
                )
        else:
            target_version_id = int(payload.get("version_id", 1))
            labels = payload.get("labels")
            instance_payload = normalize_instance_payload(
                label_data,
                num_faces=len(labels) if isinstance(labels, list) else None,
                allow_empty=(task_mode == "seg_only"),
            )

        if labels is None:
            raise ValueError(f"labels missing for sample {sample_id}")

        out_labels = _as_train_label_list(labels, sample_id)
        data_to_save = _build_train_payload(
            task_mode=task_mode,
            labels=out_labels,
            instance_payload=instance_payload,
            class_mapping=class_mapping,
            sample_id=sample_id,
        )

        out_path = labels_dir / f"{sample_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=True, indent=2)

        exported_items.append(
            {
                "sample_id": sample_id,
                "version_id": target_version_id,
                "label_path": str(out_path),
                "full_label_path": str(label_path),
            }
        )

    manifest_out = {
        "export_id": export_id,
        "task_mode": task_mode,
        "label_format": "list" if task_mode == "seg_only" else "mfinstseg_seg_inst",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "source_manifest": str(manifest_path) if manifest_path is not None else None,
        "labels_full_dir": str(labels_full_dir),
        "class_mapping": class_mapping,
        "background_labels": [int(label) for label in background_labels],
        "count": len(exported_items),
        "items": exported_items,
    }
    with open(export_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, ensure_ascii=True, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export training labels from labels_full.")
    parser.add_argument("--labels-full-dir", required=True, help="Path to labels_full directory.")
    parser.add_argument("--manifest", default=None, help="Path to export manifest json.")
    parser.add_argument("--export-id", default=None, help="Export batch id.")
    parser.add_argument("--output-root", default=None, help="Output labels_train/<export_id> root.")
    parser.add_argument(
        "--task-mode",
        choices=("seg_only", "seg_inst"),
        default="seg_only",
        help="Training label format to export.",
    )
    parser.add_argument("--class-mapping", default=None, help="Optional class mapping json.")
    parser.add_argument(
        "--background-labels",
        default="0",
        help="Comma separated background label ids recorded in manifest.",
    )
    parser.add_argument(
        "--use-manifest-versions",
        action="store_true",
        help="Use version_id from manifest entries instead of latest labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    labels_full_dir = Path(args.labels_full_dir)
    manifest_path = Path(args.manifest) if args.manifest else None
    class_mapping = _load_class_mapping(Path(args.class_mapping)) if args.class_mapping else None
    background_labels = [
        int(item.strip())
        for item in str(args.background_labels).split(",")
        if item.strip()
    ]
    if not labels_full_dir.exists():
        raise FileNotFoundError(f"labels_full dir not found: {labels_full_dir}")
    if manifest_path is not None and not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest_data = _load_manifest(manifest_path) if manifest_path is not None else None
    default_export_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    export_id = args.export_id or (manifest_data.get("export_id") if manifest_data else None) or default_export_id

    if args.output_root:
        export_root = Path(args.output_root)
    else:
        export_root = labels_full_dir.parent / "labels_train" / export_id

    export_labels(
        labels_full_dir=labels_full_dir,
        manifest_path=manifest_path,
        export_root=export_root,
        export_id=export_id,
        use_manifest_versions=args.use_manifest_versions,
        task_mode=args.task_mode,
        class_mapping=class_mapping,
        background_labels=background_labels,
    )


if __name__ == "__main__":
    main()
