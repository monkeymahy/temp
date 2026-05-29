from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np


def _as_int_list(values: Any, *, name: str) -> List[int]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    return [int(value) for value in values]


def _labels_from_seg(seg: Any, num_faces: Optional[int] = None) -> Optional[List[int]]:
    if isinstance(seg, list):
        labels = [int(value) for value in seg]
    elif isinstance(seg, dict):
        if num_faces is None:
            keys = [int(key) for key in seg.keys()]
            num_faces = max(keys) + 1 if keys else 0
        labels = [0 for _ in range(num_faces)]
        for idx in range(num_faces):
            key = str(idx)
            if key not in seg:
                raise ValueError(f"seg missing face index {idx}")
            labels[idx] = int(seg[key])
    else:
        return None

    if num_faces is not None and len(labels) != num_faces:
        raise ValueError(f"seg length {len(labels)} does not match num_faces {num_faces}")
    return labels


def _labels_from_payload(payload: Any, num_faces: Optional[int] = None) -> Optional[List[int]]:
    if isinstance(payload, dict):
        if "seg" in payload:
            return _labels_from_seg(payload["seg"], num_faces)
        if "labels" in payload:
            return _labels_from_seg(payload["labels"], num_faces)
        domains = payload.get("domains")
        if isinstance(domains, dict):
            geometry = domains.get("geometry")
            if isinstance(geometry, dict) and "face" in geometry:
                return _labels_from_seg(geometry["face"], num_faces)
        return None
    if isinstance(payload, list):
        if all(isinstance(item, (int, np.integer)) for item in payload):
            return _labels_from_seg(payload, num_faces)
        if len(payload) == 2 and isinstance(payload[1], dict):
            return _labels_from_payload(payload[1], num_faces)
        if payload and isinstance(payload[0], list) and len(payload[0]) == 2:
            return _labels_from_payload(payload[0][1], num_faces)
    return None


def extract_instance_from_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, list):
        if len(payload) == 2 and isinstance(payload[1], dict):
            return extract_instance_from_payload(payload[1])
        if payload and isinstance(payload[0], list) and len(payload[0]) == 2:
            return extract_instance_from_payload(payload[0][1])
        return None

    if not isinstance(payload, dict):
        return None

    domains = payload.get("domains")
    if isinstance(domains, dict):
        instance = domains.get("instance")
        if isinstance(instance, dict):
            return deepcopy(instance)

    if "inst" in payload:
        return {"inst": deepcopy(payload["inst"])}

    if "face_instance" in payload or "instances" in payload:
        return {
            "face_instance": deepcopy(payload.get("face_instance")),
            "instances": deepcopy(payload.get("instances", [])),
        }

    return None


def normalize_instance_payload(
    payload: Dict[str, Any],
    num_faces: Optional[int] = None,
    *,
    allow_empty: bool = True,
) -> Dict[str, Any]:
    instance = extract_instance_from_payload(payload)
    if instance is None:
        if not allow_empty:
            raise ValueError("instance label is missing")
        if num_faces is None:
            return {"face_instance": [], "instances": []}
        return {"face_instance": [-1 for _ in range(num_faces)], "instances": []}

    if "face_instance" in instance and instance.get("face_instance") is not None:
        face_instance = _as_int_list(instance["face_instance"], name="face_instance")
        if num_faces is not None and len(face_instance) != num_faces:
            raise ValueError(
                f"face_instance length {len(face_instance)} does not match num_faces {num_faces}"
            )
        instances = instance.get("instances")
        if not isinstance(instances, list):
            labels = _labels_from_seg(payload.get("labels") or payload.get("seg"), len(face_instance))
            instances = build_instances_from_face_instance(face_instance, labels or [])
        else:
            instances = normalize_instances(instances)
        result = {"face_instance": face_instance, "instances": instances}
        errors = validate_instance_struct(result, len(face_instance))
        if errors:
            raise ValueError("; ".join(errors))
        return result

    if "inst" in instance:
        seg_labels = _labels_from_payload(payload, num_faces)
        return adj_to_instance_payload(instance["inst"], seg_labels or [], background_labels={0})

    if num_faces is None:
        raise ValueError("cannot normalize instance payload without num_faces")
    return {"face_instance": [-1 for _ in range(num_faces)], "instances": []}


def normalize_instances(instances: Any) -> List[Dict[str, Any]]:
    if not isinstance(instances, list):
        raise ValueError("instances must be a list")
    normalized = []
    for item in instances:
        if not isinstance(item, dict):
            raise ValueError("each instance must be a dict")
        normalized.append(
            {
                "instance_id": int(item["instance_id"]),
                "class_id": int(item["class_id"]),
                "face_indices": sorted(int(idx) for idx in item.get("face_indices", [])),
            }
        )
    return sorted(normalized, key=lambda item: item["instance_id"])


def build_instances_from_face_instance(
    face_instance: Sequence[int],
    seg_labels: Sequence[int],
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for face_idx, instance_id in enumerate(face_instance):
        instance_id = int(instance_id)
        if instance_id >= 0:
            grouped[instance_id].append(face_idx)

    instances = []
    for instance_id, face_indices in sorted(grouped.items()):
        class_id = 0
        if seg_labels:
            votes = Counter(int(seg_labels[idx]) for idx in face_indices)
            class_id = votes.most_common(1)[0][0]
        instances.append(
            {
                "instance_id": int(instance_id),
                "class_id": int(class_id),
                "face_indices": sorted(face_indices),
            }
        )
    return instances


def rebuild_face_instance(instances: Sequence[Dict[str, Any]], num_faces: int) -> List[int]:
    face_instance = [-1 for _ in range(num_faces)]
    for instance in normalize_instances(list(instances)):
        instance_id = int(instance["instance_id"])
        for face_idx in instance["face_indices"]:
            if face_idx < 0 or face_idx >= num_faces:
                raise ValueError(f"face index out of range: {face_idx}")
            if face_instance[face_idx] != -1:
                raise ValueError(f"face {face_idx} appears in more than one instance")
            face_instance[face_idx] = instance_id
    return face_instance


def face_instance_to_adj(face_instance: Sequence[int]) -> List[List[int]]:
    ids = [int(value) for value in face_instance]
    num_faces = len(ids)
    adj = np.zeros((num_faces, num_faces), dtype=np.uint8)
    grouped: Dict[int, List[int]] = defaultdict(list)
    for face_idx, instance_id in enumerate(ids):
        if instance_id >= 0:
            grouped[instance_id].append(face_idx)
    for face_indices in grouped.values():
        for row in face_indices:
            for col in face_indices:
                adj[row, col] = 1
    return adj.tolist()


def adj_to_instance_payload(
    inst_adj: Any,
    seg_labels: Sequence[int],
    background_labels: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    adj = np.asarray(inst_adj, dtype=np.int32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("inst adjacency must be a square matrix")

    num_faces = int(adj.shape[0])
    if len(seg_labels) not in (0, num_faces):
        raise ValueError(f"seg length {len(seg_labels)} does not match inst size {num_faces}")

    bg: Set[int] = {int(v) for v in (background_labels or {0})}
    adj = ((adj > 0) | (adj.T > 0)).astype(np.uint8)

    visited = np.zeros(num_faces, dtype=bool)
    face_instance = [-1 for _ in range(num_faces)]
    instances: List[Dict[str, Any]] = []
    next_instance_id = 0

    for start in range(num_faces):
        if visited[start] or not adj[start].any():
            continue
        queue: deque[int] = deque([start])
        visited[start] = True
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            neighbors = np.where(adj[node] > 0)[0].tolist()
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(int(neighbor))

        if not component:
            continue
        labels = [int(seg_labels[idx]) for idx in component] if seg_labels else []
        if labels and all(label in bg for label in labels):
            continue
        class_id = Counter(labels).most_common(1)[0][0] if labels else 0
        for face_idx in component:
            face_instance[face_idx] = next_instance_id
        instances.append(
            {
                "instance_id": next_instance_id,
                "class_id": int(class_id),
                "face_indices": sorted(int(idx) for idx in component),
            }
        )
        next_instance_id += 1

    return {"face_instance": face_instance, "instances": instances}


def validate_instance_struct(instance: Dict[str, Any], num_faces: int) -> List[str]:
    errors: List[str] = []
    face_instance = instance.get("face_instance")
    instances = instance.get("instances")

    if not isinstance(face_instance, list):
        errors.append("face_instance must be a list")
        return errors
    if len(face_instance) != num_faces:
        errors.append(f"face_instance length {len(face_instance)} != num_faces {num_faces}")

    if not isinstance(instances, list):
        errors.append("instances must be a list")
        return errors

    seen_instance_ids = set()
    expected = [-1 for _ in range(num_faces)]
    for item in instances:
        if not isinstance(item, dict):
            errors.append("instance item must be a dict")
            continue
        try:
            instance_id = int(item["instance_id"])
            face_indices = [int(idx) for idx in item.get("face_indices", [])]
            int(item["class_id"])
        except Exception as exc:
            errors.append(f"invalid instance item: {exc}")
            continue
        if instance_id in seen_instance_ids:
            errors.append(f"duplicated instance_id: {instance_id}")
        seen_instance_ids.add(instance_id)
        if not face_indices:
            errors.append(f"instance {instance_id} has no faces")
        for face_idx in face_indices:
            if face_idx < 0 or face_idx >= num_faces:
                errors.append(f"instance {instance_id} face index out of range: {face_idx}")
                continue
            if expected[face_idx] != -1:
                errors.append(f"face {face_idx} appears in multiple instances")
            expected[face_idx] = instance_id

    for face_idx, instance_id in enumerate(face_instance[:num_faces]):
        instance_id = int(instance_id)
        if instance_id != expected[face_idx]:
            errors.append(
                f"face_instance mismatch at face {face_idx}: {instance_id} != {expected[face_idx]}"
            )
        if instance_id >= 0 and instance_id not in seen_instance_ids:
            errors.append(f"face {face_idx} refers missing instance_id {instance_id}")

    return errors


def validate_inst_adj(inst_adj: Any, num_faces: int) -> List[str]:
    errors = []
    adj = np.asarray(inst_adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        return ["inst adjacency must be a square matrix"]
    if adj.shape[0] != num_faces:
        errors.append(f"inst adjacency size {adj.shape[0]} != num_faces {num_faces}")
    if not np.array_equal(adj, adj.T):
        errors.append("inst adjacency must be symmetric")
    return errors


def build_instance_change_ops(
    old_face_instance: Sequence[int],
    new_face_instance: Sequence[int],
    old_instances: Sequence[Dict[str, Any]],
    new_instances: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if len(old_face_instance) != len(new_face_instance):
        raise ValueError("old and new face_instance lengths differ")
    indices = []
    old_ids = []
    new_ids = []
    for idx, (old_id, new_id) in enumerate(zip(old_face_instance, new_face_instance)):
        if int(old_id) == int(new_id):
            continue
        indices.append(idx)
        old_ids.append(int(old_id))
        new_ids.append(int(new_id))
    if not indices and list(old_instances) == list(new_instances):
        return []
    return [
        {
            "type": "instance_change",
            "indices": indices,
            "old_instance_ids": old_ids,
            "new_instance_ids": new_ids,
            "old_instances": deepcopy(list(old_instances)),
            "new_instances": deepcopy(list(new_instances)),
        }
    ]


def apply_instance_ops(instance: Dict[str, Any], ops: Any, *, use_new: bool) -> Dict[str, Any]:
    updated = deepcopy(instance)
    if not isinstance(ops, list):
        return updated
    id_key = "new_instance_ids" if use_new else "old_instance_ids"
    instances_key = "new_instances" if use_new else "old_instances"
    for op in ops:
        if not isinstance(op, dict) or op.get("type") != "instance_change":
            continue
        indices = op.get("indices", [])
        ids = op.get(id_key, [])
        if isinstance(indices, list) and isinstance(ids, list):
            for face_idx, instance_id in zip(indices, ids):
                face_idx = int(face_idx)
                if 0 <= face_idx < len(updated.get("face_instance", [])):
                    updated["face_instance"][face_idx] = int(instance_id)
        if isinstance(op.get(instances_key), list):
            updated["instances"] = normalize_instances(op[instances_key])
    return updated


def rollback_instance_payload(payload: Dict[str, Any], target_version_id: int) -> Dict[str, Any]:
    domains = payload.get("domains") if isinstance(payload, dict) else None
    instance_current = {}
    if isinstance(domains, dict) and isinstance(domains.get("instance"), dict):
        instance_current = normalize_instance_payload(payload)

    instance_base = None
    if isinstance(domains, dict):
        instance_base = domains.get("instance_base")
    if isinstance(instance_base, dict):
        instance = normalize_instance_payload({"domains": {"instance": instance_base}})
    elif instance_current:
        num_faces = len(instance_current.get("face_instance", []))
        instance = {"face_instance": [-1 for _ in range(num_faces)], "instances": []}
    else:
        return {"face_instance": [], "instances": []}

    target_version_id = int(target_version_id)
    if target_version_id <= 0:
        return instance

    versions = payload.get("versions", [])
    if not isinstance(versions, list):
        return instance

    for record in sorted(
        versions,
        key=lambda item: int(item.get("version_id", 0)) if isinstance(item, dict) else 0,
    ):
        if not isinstance(record, dict):
            continue
        version_id = int(record.get("version_id", 0))
        if version_id > target_version_id:
            break
        instance = apply_instance_ops(instance, record.get("ops"), use_new=True)

    if target_version_id >= int(payload.get("version_id", target_version_id)) and instance_current:
        return instance_current
    return instance
