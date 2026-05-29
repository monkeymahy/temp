from __future__ import annotations

from collections import Counter, deque
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def postprocess_instance_logits(
    inst_logits,
    seg_logits,
    threshold: float = 0.5,
    background_labels: Optional[Iterable[int]] = None,
) -> Dict[str, List]:
    """将模型输出的实例邻接 logits 转成 face_instance 与 instances。"""
    inst_np = _to_numpy(inst_logits)
    seg_np = _to_numpy(seg_logits)
    if inst_np.ndim == 3:
        if inst_np.shape[0] != 1:
            raise ValueError("postprocess_instance_logits expects one sample when logits are batched")
        inst_np = inst_np[0]
    if seg_np.ndim != 2:
        raise ValueError("seg_logits must have shape [num_faces, num_classes]")
    num_faces = seg_np.shape[0]
    if inst_np.shape != (num_faces, num_faces):
        raise ValueError(f"inst logits shape {inst_np.shape} does not match num_faces {num_faces}")

    probabilities = 1.0 / (1.0 + np.exp(-inst_np))
    adj = probabilities >= float(threshold)
    adj = np.logical_or(adj, adj.T)

    pred_labels = np.argmax(seg_np, axis=1).astype(np.int32)
    bg = {int(label) for label in (background_labels or {0})}

    visited = np.zeros(num_faces, dtype=bool)
    face_instance = [-1 for _ in range(num_faces)]
    instances = []
    next_instance_id = 0

    for start in range(num_faces):
        if visited[start] or not adj[start].any():
            continue
        queue = deque([start])
        visited[start] = True
        component = []
        while queue:
            node = queue.popleft()
            component.append(int(node))
            neighbors = np.where(adj[node])[0].tolist()
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(int(neighbor))

        labels = [int(pred_labels[idx]) for idx in component]
        if labels and all(label in bg for label in labels):
            continue
        class_id = Counter(labels).most_common(1)[0][0] if labels else 0
        for face_idx in component:
            face_instance[face_idx] = next_instance_id
        instances.append(
            {
                "instance_id": next_instance_id,
                "class_id": int(class_id),
                "face_indices": sorted(component),
            }
        )
        next_instance_id += 1

    return {"face_instance": face_instance, "instances": instances}

