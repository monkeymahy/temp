# 陈守玉
from decimal import Decimal
import json
import ijson
import os
from pathlib import Path
import tempfile
from tqdm import tqdm


def _json_default(obj):
    """
    Fallback serializer for json.dump(..., default=_json_default)
    Handles Decimal, date/datetime, bytes, Path, set and objects with __dict__.
    """
    if isinstance(obj, Decimal):
        # preserve integer-valued decimals as ints when possible
        try:
            if obj == obj.to_integral():
                return int(obj)
        except Exception:
            pass
        return float(obj)

    return obj


def split_ijson_file(
    path,
    output_dir=None,
    item_prefix="item",
    start_index=0,
    index_width=8,
    compact=True,
    progress_interval=1000,
):
    """
    将一个大的 JSON（通常为一个顶层数组）按元素流式写成多个小的 JSON 文件，避免一次性载入内存。
    参数：
      - path: 待拆分的大 JSON 文件路径（支持本地文件）。
      - output_dir: 输出目录，若为 None，将在输入文件同目录下创建 "<basename>_split" 目录。
      - item_prefix: ijson 的 prefix，用于定位要迭代的条目。默认 "item"（顶层数组元素）。
                     如果你的大 JSON 是 {"data": [...]} 则使用 "data.item"。
      - start_index: 起始编号（默认 0）。
      - index_width: 文件编号的零填宽度（默认 8）。
      - compact: 是否使用紧凑的 JSON 格式（去掉空格）。
      - progress_interval: 每写入多少个文件输出一次进度信息；设为 0 则不打印进度。
    返回：
      - 写入的文件数量（int）。
    注意：
      - 使用 ijson 流式解析，不会把整个文件加载到内存中，适合非常大的文件（如 20GB）。
      - 对每个条目先写入临时文件，完成后原子性替换为目标文件名，避免中断时产生不完整文件。
    """

    in_path = Path(path)
    if output_dir is None:
        output_dir = in_path.parent / f"{in_path.stem}_split"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_prefix = in_path.stem if not item_prefix else in_path.stem
    count = 0
    try:
        with open(in_path, "rb") as f:
            # ijson.items(f, item_prefix) 会按流产出每个元素（假设顶层是数组时 prefix="item"）
            for idx, item in tqdm(
                enumerate(
                    ijson.items(f, item_prefix),
                    start=start_index,
                )
            ):
                filename = f"{base_prefix}_{idx:0{index_width}d}.json"
                final_path = output_dir / filename

                # 使用临时文件保证写入的原子性：写好后重命名或替换为最终文件
                with tempfile.NamedTemporaryFile(
                    "w",
                    delete=False,
                    encoding="utf-8",
                    dir=str(output_dir),
                ) as tmp:
                    if compact:
                        json.dump(
                            obj=item,
                            fp=tmp,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=_json_default,
                        )
                    else:
                        json.dump(
                            obj=item,
                            fp=tmp,
                            ensure_ascii=False,
                            indent=2,
                        )
                    tmp_name = tmp.name
                os.replace(tmp_name, final_path)

                count += 1
                if progress_interval and (count % progress_interval == 0):
                    print(f"written {count} files, latest: {final_path}")

    except KeyboardInterrupt:
        print(f"Interrupted by user. {count} files written so far.")
    except Exception as e:
        print(f"Error after writing {count} files: {e}")
        raise

    return count


if __name__ == "__main__":
    path = "C:/Data/graphs.json"
    data = split_ijson_file(
        path=path,
        output_dir="C:/Data/MFCAD2_split",
        progress_interval=1000,
    )
